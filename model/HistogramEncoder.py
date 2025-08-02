import torch
import torch.nn as nn
import torch.nn.functional as F

class HistogramEncoder(nn.Module):
    def __init__(self, bins_per_channel: int = 8, out_features: int = 8):
        super().__init__()
        self.bins_per_channel = bins_per_channel
        self.out_features = out_features

        # 1x1 conv acts like an MLP applied per-pixel
        self.encoder = nn.Sequential(
            nn.Conv2d(3 * bins_per_channel, 32, kernel_size=1),  # per-pixel linear map
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, out_features, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x_hist):
        return self.encoder(x_hist)  # Output: (out_features, H, W)
