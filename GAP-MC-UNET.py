import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn import init

# Basic convolution blocks used in U-Net

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)

# Encoder block with 3 convolutions and optional pooling
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        xskip = self.conv1(x)
        x = F.relu(self.conv2(xskip))
        x = F.relu(self.conv3(x) + xskip)

        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

# Decoder block with upconvolution and merging of encoder features
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(in_channels, out_channels, mode=self.up_mode)
        self.conv1 = conv3x3(out_channels * 2 if merge_mode == 'concat' else out_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        x = torch.cat((from_up, from_down), 1) if self.merge_mode == 'concat' else from_up + from_down
        xskip = self.conv1(x)
        x = F.relu(self.conv2(xskip))
        x = F.relu(self.conv3(x) + xskip)
        return x

# U-Net implementation modified for photon distribution prediction
class UN(pl.LightningModule):
    def __init__(self, levels, channels=32, depth=5, start_filts=64, up_mode='transpose', merge_mode='add'):
        super(UN, self).__init__()
        """
        Args:
            channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """

        self.save_hyperparameters()

        self.levels = levels
        self.channels = channels  # number of time bins (output channels)
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        # Build encoder
        for i in range(depth):
            ins = self.channels * self.levels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = i < depth - 1
            self.down_convs.append(DownConv(ins, outs, pooling))

        # Build decoder
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            self.up_convs.append(UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode))

        # Final conv layer to produce output distribution
        self.conv_final = conv1x1(outs, self.channels)
        self.reset_params()

    # Xavier initialization
    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    # Forward pass with sinusoidal encoding and softmax output over time bins
    def forward(self, x):
        # Sinusoidal Encoding
        stack = None
        factor = 10.0
        for i in range(self.levels):
            scale = torch.sin(x.clone() * (factor ** -i))
            stack = scale if stack is None else torch.cat((stack, scale), 1)
        x = stack

        encoder_outs = []
        for down in self.down_convs:
            x, before_pool = down(x)
            encoder_outs.append(before_pool)

        for i, up in enumerate(self.up_convs):
            x = up(encoder_outs[-(i + 2)], x)

        x = self.conv_final(x)      # [B, T, H, W]
        x = F.softmax(x, dim=1)     # softmax over time bins
        return x

    # Optimizer and LR scheduler configuration
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    # KL divergence loss between predicted and true distributions
    def KLLoss(self, pred, target):
        # torch.clamp -> ensures stability by clamping elements in range [min, max] -> [1e-8, 1.0]
        # needed when probs are equal to zero or undefined NaN
        pred = torch.clamp(pred, 1e-8, 1.0)
        target = torch.clamp(target, 1e-8, 1.0)
        # KL Divergence
        kl = target * torch.log(target / pred)
        return kl.sum(dim=1).mean()  # sum over time, average over batch and spatial dims

    # Training step using KL loss
    def training_step(self, batch, batch_idx):
        X = batch[:, self.channels:, ...]           # [B, T, H, W] noisy input
        target = batch[:, :self.channels, ...]      # [B, T, H, W] clean MC target
        pred = self(X)
        loss = self.KLLoss(pred, target)
        self.log("train_loss", loss)
        return loss

    # Validation step
    def validation_step(self, batch, batch_idx):
        X = batch[:, self.channels:, ...]
        target = batch[:, :self.channels, ...]
        pred = self(X)
        loss = self.KLLoss(pred, target)
        self.log("val_loss", loss)

    # Test step
    def test_step(self, batch, batch_idx):
        X = batch[:, self.channels:, ...]
        target = batch[:, :self.channels, ...]
        pred = self(X)
        loss = self.KLLoss(pred, target)
        self.log("test_loss", loss)
