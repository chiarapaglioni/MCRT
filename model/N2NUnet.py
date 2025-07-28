import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Net(nn.Module):
    def __init__(self, in_channels=9):
        super(Net, self).__init__()
        self.in_channels = in_channels

        # ---- Dynamic base width ----
        self.base_width = 48 if in_channels <= 9 else int(in_channels * 1.5)
        final_bw = 64 if in_channels <= 9 else int(in_channels * 2)
        bw = self.base_width  # base feature size
        bw2 = bw * 2  # used in decoder

        # ---- ENCODER ----
        self.enc_conv01 = nn.Sequential(
            ConvBlockLeakyRelu(in_channels, bw, 3, stride=1, padding=1),
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
            ConvBlockLeakyRelu(bw2 + in_channels, final_bw, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(final_bw, final_bw // 2, 3, stride=1, padding=1),
            nn.Conv2d(final_bw // 2, 3, 3, stride=1, padding=1, bias=True)
        )

        self._initialize_weights()

    def forward(self, x):
        residual_connection = [x]

        # ---- ENCODER ----
        x = self.enc_conv01(x)
        residual_connection.append(x)

        x = self.enc_conv2(x)
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