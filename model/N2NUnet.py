import torch
import torch.nn as nn

# Basic 3x3 convolution layer
def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same")

# CONVBLOCK with LeakyReLU(0.1)
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
         return self.lrelu(self.conv(x))

# UNET
class Noise2NoiseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=48):
        super().__init__()

        # -------- ENCODER --------
        self.enc_conv0 = ConvBlock(in_channels, features)       # ENC_CONV0
        self.enc_conv1 = ConvBlock(features, features)          # ENC_CONV1
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv2 = ConvBlock(features, features)          # ENC_CONV2
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv3 = ConvBlock(features, features)          # ENC_CONV3
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv4 = ConvBlock(features, features)          # ENC_CONV4
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv5 = ConvBlock(features, features)          # ENC_CONV5
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.enc_conv6 = ConvBlock(features, features)          # ENC_CONV6

        # -------- DECODER --------
        self.upsample5 = nn.Upsample(scale_factor=2, mode='nearest')      # UPSAMPLE5
        self.dec_conv5a = ConvBlock(features * 2, features * 2)           # DEC_CONV5A
        self.dec_conv5b = ConvBlock(features * 2, features * 2)           # DEC_CONV5B

        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')      # UPSAMPLE4
        self.dec_conv4a = ConvBlock(features * 3, features * 2)           # DEC_CONV4A (144 in → 96 out)
        self.dec_conv4b = ConvBlock(features * 2, features * 2)           # DEC_CONV4B

        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')      # UPSAMPLE3
        self.dec_conv3a = ConvBlock(features * 3, features * 2)           # DEC_CONV3A
        self.dec_conv3b = ConvBlock(features * 2, features * 2)           # DEC_CONV3B

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')      # UPSAMPLE2
        self.dec_conv2a = ConvBlock(features * 3, features * 2)           # DEC_CONV2A
        self.dec_conv2b = ConvBlock(features * 2, features * 2)           # DEC_CONV2B

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')      # UPSAMPLE1
        self.dec_conv1a = ConvBlock(features * 2 + in_channels, 64)       # DEC_CONV1A
        self.dec_conv1b = ConvBlock(64, 32)                               # DEC_CONV1B
        self.dec_conv1c = conv3x3(32, out_channels)                       # DEC_CONV1C (linear)

        self._initialize_weights()

    def forward(self, x):
        input_x = x

        # ------- ENCODER -------
        x0 = self.enc_conv0(x)                  # ENC_CONV0
        x1 = self.enc_conv1(x0)                 # ENC_CONV1
        x1 = self.pool1(x1)                     # POOL1
        s1 = x1                                 # skip

        x2 = self.enc_conv2(x1)                 # ENC_CONV2
        x2 = self.pool2(x2)                     # POOL2
        s2 = x2                                 # skip

        x3 = self.enc_conv3(x2)                 # ENC_CONV3
        x3 = self.pool3(x3)                     # POOL3
        s3 = x3                                 # skip

        x4 = self.enc_conv4(x3)                 # ENC_CONV4
        x4 = self.pool4(x4)                     # POOL4
        s4 = x4                                 # skip

        x5 = self.enc_conv5(x4)                 # ENC_CONV5
        x5 = self.pool5(x5)                     # POOL5

        x6 = self.enc_conv6(x5)                 # ENC_CONV6

        # ------- DECODER -------
        x = self.upsample5(x6)
        x = torch.cat([x, s4], dim=1)           # CONCAT5
        x = self.dec_conv5a(x)                  # DEC_CONV5A
        x = self.dec_conv5b(x)                  # DEC_CONV5B

        x = self.upsample4(x)
        x = torch.cat([x, s3], dim=1)           # CONCAT4
        x = self.dec_conv4a(x)                  # DEC_CONV4A
        x = self.dec_conv4b(x)                  # DEC_CONV4B

        x = self.upsample3(x)
        x = torch.cat([x, s2], dim=1)           # CONCAT3
        x = self.dec_conv3a(x)                  # DEC_CONV3A
        x = self.dec_conv3b(x)                  # DEC_CONV3B

        x = self.upsample2(x)
        x = torch.cat([x, s1], dim=1)           # CONCAT2
        x = self.dec_conv2a(x)                  # DEC_CONV2A
        x = self.dec_conv2b(x)                  # DEC_CONV2B

        x = self.upsample1(x)
        x = torch.cat([x, input_x], dim=1)      # CONCAT1
        x = self.dec_conv1a(x)                  # DEC_CONV1A
        x = self.dec_conv1b(x)                  # DEC_CONV1B
        x = self.dec_conv1c(x)                  # DEC_CONV1C (no activation)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


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
    def __init__(self):
        super(Net, self).__init__()

        ## ------- ENCODER -------
        self.enc_conv01 = nn.Sequential(
            ConvBlockLeakyRelu(3, 48, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )

        self.enc_conv2 = nn.Sequential(
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )

        self.enc_conv3 = nn.Sequential(
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )

        self.enc_conv4 = nn.Sequential(
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )

        self.enc_conv56 = nn.Sequential(
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
            nn.MaxPool2d(2),
            ConvBlockLeakyRelu(48, 48, 3, stride=1, padding=1),
        )

        ## ------- DECODER -------
        self.dec_conv5ab = nn.Sequential(
            ConvBlockLeakyRelu(96, 96, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(96, 96, 3, stride=1, padding=1),
        )

        self.dec_conv4ab = nn.Sequential(
            ConvBlockLeakyRelu(144, 96, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(96, 96, 3, stride=1, padding=1),
        )

        self.dec_conv3ab = nn.Sequential(
            ConvBlockLeakyRelu(144, 96, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(96, 96, 3, stride=1, padding=1),
        )

        self.dec_conv2ab = nn.Sequential(
            ConvBlockLeakyRelu(144, 96, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(96, 96, 3, stride=1, padding=1),
        )

        self.dec_conv1abc = nn.Sequential(
            ConvBlockLeakyRelu(99, 64, 3, stride=1, padding=1),
            ConvBlockLeakyRelu(64, 32, 3, stride=1, padding=1),
            nn.Conv2d(32, 3, 3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        ## ------- ENCODER -------
        residual_connection = [x]

        x = self.enc_conv01(x)
        residual_connection.append(x)

        x = self.enc_conv2(x)
        residual_connection.append(x)

        x = self.enc_conv3(x)
        residual_connection.append(x)

        x = self.enc_conv4(x)
        residual_connection.append(x)

        x = self.enc_conv56(x)

        ## ------- DECODER -------
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
