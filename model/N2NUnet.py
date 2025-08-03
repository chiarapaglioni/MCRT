import torch
import torch.nn as nn
import torch.nn.functional as F
from model.HistogramEncoder import HistogramEncoder, HistFeatureModulator

import logging
logger = logging.getLogger(__name__)


class ConvBlockLeakyRelu(nn.Module):
    '''
    A block containing a Conv2d followed by a leakyRelu
    '''
    def __init__(self, chanel_in, chanel_out, kernel_size, stride=1, padding=1):
        super(ConvBlockLeakyRelu, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(chanel_in, chanel_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class N2Net(nn.Module):
    def __init__(self, in_channels=9, hist_bins=8, mode="img"):
        super(N2Net, self).__init__()
        self.mode = mode
        self.hist_bins = hist_bins
        self.hist_encoder_out_channels = 8  # fixed size after encoding
        logger.info(f"Initialized N2Net with mode={self.mode}, input_channels={in_channels}")

        # Dynamic input channel size based on mode
        if self.mode == "hist":
            hist_channels = 3 * hist_bins
            self.spatial_in_channels = in_channels - hist_channels
            self.hist_encoder = HistogramEncoder(bins_per_channel=hist_bins, out_features=self.hist_encoder_out_channels)
            logger.info(f"Hist input_channels={hist_channels}")
        else:
            self.spatial_in_channels = in_channels  # full input is spatial

        logger.info(f"Spatial input_channels={self.spatial_in_channels}")

        # ---- Dynamic base width ----
        self.base_width = 48 if self.spatial_in_channels <= 15 else int(self.spatial_in_channels * 1.5)
        final_bw = 64 if self.spatial_in_channels <= 15 else int(self.spatial_in_channels * 2)
        bw = self.base_width
        bw2 = bw * 2

        # TODO: currently removed learnable hist features as they were not helping
        # if self.mode == 'hist':
        #     self.mod_enc_conv01 = HistFeatureModulator(self.hist_encoder_out_channels, bw)
        #     self.mod_enc_conv2 = HistFeatureModulator(self.hist_encoder_out_channels, bw)

        # Change encoder/decoder input channels based on the mode!
        first_encoder_input_channels = self.spatial_in_channels
        first_decoder_input_channels = bw2 + self.spatial_in_channels

        # ---- ENCODER ----
        self.enc_conv01 = nn.Sequential(
            ConvBlockLeakyRelu(first_encoder_input_channels, bw, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(bw, bw, 3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )

        self.enc_conv2 = nn.Sequential(
            ConvBlockLeakyRelu(bw, bw, 3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )

        self.enc_conv3 = nn.Sequential(
            ConvBlockLeakyRelu(bw, bw, 3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )

        self.enc_conv4 = nn.Sequential(
            ConvBlockLeakyRelu(bw, bw, 3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )

        self.enc_conv56 = nn.Sequential(
            ConvBlockLeakyRelu(bw, bw, 3, stride=1, padding=1),
            nn.MaxPool2d(2),
            ConvBlockLeakyRelu(bw, bw, 3, stride=1, padding=1)
        )

        # ---- DECODER ----
        self.dec_conv5ab = nn.Sequential(
            ConvBlockLeakyRelu(bw * 2, bw2, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(bw2, bw2, 3, stride=1, padding=1)
        )

        self.dec_conv4ab = nn.Sequential(
            ConvBlockLeakyRelu(bw2 + bw, bw2, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(bw2, bw2, 3, stride=1, padding=1)
        )

        self.dec_conv3ab = nn.Sequential(
            ConvBlockLeakyRelu(bw2 + bw, bw2, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(bw2, bw2, 3, stride=1, padding=1)
        )

        self.dec_conv2ab = nn.Sequential(
            ConvBlockLeakyRelu(bw2 + bw, bw2, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(bw2, bw2, 3, stride=1, padding=1)
        )

        self.dec_conv1abc = nn.Sequential(
            ConvBlockLeakyRelu(first_decoder_input_channels, final_bw, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(final_bw, final_bw // 2, 3, stride=1, padding=1),
            nn.Conv2d(final_bw // 2, 3, 3, stride=1, padding=1, bias=True)
        )

        self._initialize_weights()

    def forward(self, x, x_hist=None):
        if self.mode == "hist":
            assert x_hist is not None, "Histogram input required in hist mode"
            hist_feat = self.hist_encoder(x_hist)  # [B, D]

        spatial_x = x[:, :self.spatial_in_channels, :, :]
        x = spatial_x  # no concat here

        residual_connection = [x]

        # ---- ENCODER ----
        x = self.enc_conv01(x)
        # TODO: currently removed learnable hist features as they were not helping
        # if self.mode == "hist":
        #     x = self.mod_enc_conv01(x, hist_feat)       # modulate with histogram features
        residual_connection.append(x)

        x = self.enc_conv2(x)
        # if self.mode == "hist":
        #     x = self.mod_enc_conv2(x, hist_feat)        # modulate with histogram features
        residual_connection.append(x)

        x = self.enc_conv3(x)
        residual_connection.append(x)

        x = self.enc_conv4(x)
        residual_connection.append(x)

        x = self.enc_conv56(x)

        # ---- DECODER ----
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, residual_connection.pop()], dim=1)
        x = self.dec_conv5ab(x)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, residual_connection.pop()], dim=1)
        x = self.dec_conv4ab(x)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, residual_connection.pop()], dim=1)
        x = self.dec_conv3ab(x)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, residual_connection.pop()], dim=1)
        x = self.dec_conv2ab(x)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, residual_connection.pop()], dim=1)
        x = self.dec_conv1abc(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)