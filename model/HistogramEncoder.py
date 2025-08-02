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
        x = self.encoder(x_hist)              # [B, D, H, W]
        x = F.adaptive_avg_pool2d(x, 1)       # [B, D, 1, 1]
        return x.view(x.size(0), -1)          # [B, D]


class HistFeatureModulator(nn.Module):
    def __init__(self, hist_feat_dim, target_channels):
        super().__init__()
        self.gamma_fc = nn.Linear(hist_feat_dim, target_channels)
        self.beta_fc = nn.Linear(hist_feat_dim, target_channels)

    def forward(self, feat_map, hist_feat):
        # feat_map: [B, C, H, W]
        # hist_feat: [B, D]
        gamma = self.gamma_fc(hist_feat).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.beta_fc(hist_feat).unsqueeze(-1).unsqueeze(-1)
        return feat_map * (1 + gamma) + beta
