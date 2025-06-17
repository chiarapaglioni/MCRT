import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np


def conv3x3_3d(in_channels, out_channels, stride=1, padding=1, bias=True):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)

def upconv2x2_3d(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

def conv1x1_3d(in_channels, out_channels):
    return nn.Conv3d(in_channels, out_channels, kernel_size=1)


class DownConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super().__init__()
        self.pooling = pooling
        self.conv1 = conv3x3_3d(in_channels, out_channels)
        self.conv2 = conv3x3_3d(out_channels, out_channels)
        self.conv3 = conv3x3_3d(out_channels, out_channels)
        self.pool = nn.MaxPool3d(2) if pooling else nn.Identity()

    def forward(self, x):
        xskip = self.conv1(x)
        x = F.relu(self.conv2(xskip))
        x = F.relu(self.conv3(x) + xskip)
        before_pool = x
        x = self.pool(x)
        return x, before_pool


class UpConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode='add'):
        super().__init__()
        self.merge_mode = merge_mode
        self.upconv = upconv2x2_3d(in_channels, out_channels)
        self.conv1 = conv3x3_3d(out_channels * 2 if merge_mode == 'concat' else out_channels, out_channels)
        self.conv2 = conv3x3_3d(out_channels, out_channels)
        self.conv3 = conv3x3_3d(out_channels, out_channels)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        x = torch.cat((from_up, from_down), 1) if self.merge_mode == 'concat' else from_up + from_down
        xskip = self.conv1(x)
        x = F.relu(self.conv2(xskip))
        x = F.relu(self.conv3(x) + xskip)
        return x


class GAP3D(pl.LightningModule):
    def __init__(self, levels=10, channels=3, depth=5, start_filts=32, merge_mode='add'):
        super().__init__()
        self.save_hyperparameters()

        self.levels = levels
        self.channels = channels
        self.start_filts = start_filts
        self.depth = depth
        self.merge_mode = merge_mode

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for i in range(depth):
            ins = self.channels * self.levels if i == 0 else outs
            outs = start_filts * (2**i)
            pooling = i < depth - 1
            self.down_convs.append(DownConv3D(ins, outs, pooling=pooling))

        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            self.up_convs.append(UpConv3D(ins, outs, merge_mode=self.merge_mode))

        self.conv_final = conv1x1_3d(outs, channels)

    def forward(self, x):
        # Apply sinusoidal encoding
        stack = None
        
        factor = 10.0
        for i in range (self.levels):
            scale = x.clone()*(factor**(-i))
            scale = torch.sin(scale)
            if stack is None:
                stack = scale
            else:
                stack = torch.cat((stack,scale), dim=1)
        
        x = stack # [B, C*levels, T, D, H, W]

        encoder_outs = []
        for down in self.down_convs:
            x, before_pool = down(x)
            encoder_outs.append(before_pool)

        for i, up in enumerate(self.up_convs):
            x = up(encoder_outs[-(i + 2)], x)

        return self.conv_final(x)

    def photonLoss(self, result, target):
        result = torch.clamp(result, -10, 10)
        expEnergy = torch.exp(result)
        perImage = -torch.mean(result * target, dim=(-1, -2, -3, -4), keepdim=True)
        perImage += torch.log(torch.mean(expEnergy, dim=(-1, -2, -3, -4), keepdim=True)) * torch.mean(target, dim=(-1, -2, -3, -4), keepdim=True)
        return torch.mean(perImage)

    def training_step(self, batch, batch_idx):
        # batch: [B, 2C, T, D, H, W]
        residual = batch[:, :self.channels]
        noisy = batch[:, self.channels:]
        pred = self(noisy)
        loss = self.photonLoss(pred, residual)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        residual = batch[:, :self.channels]
        noisy = batch[:, self.channels:]
        pred = self(noisy)
        loss = self.photonLoss(pred, residual)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        residual = batch[:, :self.channels]
        noisy = batch[:, self.channels:]
        pred = self(noisy)
        loss = self.photonLoss(pred, residual)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }