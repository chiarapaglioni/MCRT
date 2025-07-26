import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CORE CONVOLUTIONAL BLOCK ---
def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.lrelu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.lrelu2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        return x


# --- DOWNSAMPLING LAYER (CONV + POOL) ---
class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.block(x)
        x_pooled = self.pool(x)
        return x_pooled, x  # return pooled and skip connection

# --- UPSAMPLING LAYER (UPSAMPLE + CONCAT + CONV) ---
class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# --- NOISE2NOISE-LIKE UNET ---
class Noise2NoiseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=48):
        super().__init__()

        self.enc0 = ConvBlock(in_channels, features)      # CONV0
        self.pool1 = DownLayer(features, features)        # POOL1
        self.pool2 = DownLayer(features, features)        # POOL2
        self.pool3 = DownLayer(features, features)        # POOL3
        self.pool4 = DownLayer(features, features)        # POOL4
        self.pool5 = DownLayer(features, features)        # POOL5

        self.bottom = ConvBlock(features, features)       # CONV6

        self.up5 = UpLayer(features * 2, features)
        self.up4 = UpLayer(features * 2, features)
        self.up3 = UpLayer(features * 2, features)
        self.up2 = UpLayer(features * 2, features)
        self.up1 = UpLayer(features * 2, features)

        self.final_conv1 = conv3x3(features + in_channels, 64)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = conv3x3(64, 32)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = conv3x3(32, out_channels)  # Linear activation

        self._initialize_weights()  # Apply He initialization

    def forward(self, x):
        input_orig = x

        x0 = self.enc0(x)                      # ENC CONV0
        x1, skip1 = self.pool1(x0)             # POOL1
        x2, skip2 = self.pool2(x1)             # POOL2
        x3, skip3 = self.pool3(x2)             # POOL3
        x4, skip4 = self.pool4(x3)             # POOL4
        x5, skip5 = self.pool5(x4)             # POOL5

        x_bottom = self.bottom(x5)             # CONV6

        x = self.up5(x_bottom, skip5)          # DEC 5
        x = self.up4(x, skip4)                 # DEC 4
        x = self.up3(x, skip3)                 # DEC 3
        x = self.up2(x, skip2)                 # DEC 2
        x = self.up1(x, skip1)                 # DEC 1

        # Final concat with original input
        x = torch.cat([x, input_orig], dim=1)

        x = self.final_relu1(self.final_conv1(x))
        x = self.final_relu2(self.final_conv2(x))
        x = self.final_conv3(x)  # No ReLU — linear output

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)