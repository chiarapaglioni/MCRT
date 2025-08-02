import torch
import torch.nn as nn
import torch.nn.functional as F

class HistogramPatchAggregator(nn.Module):
    def __init__(self, kernel_size=7, sigma_color=0.1, sigma_guidance=0.2):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_guidance = sigma_guidance
        self.padding = kernel_size // 2

    def forward(self, output, guidance):
        """
        output: [B, 3, H, W] - network output
        guidance: [B, Cg, H, W] - e.g. albedo + normal + variance
        """
        B, C, H, W = output.shape
        unfolded_output = F.unfold(output, kernel_size=self.kernel_size, padding=self.padding)  # [B, C * K*K, H*W]
        unfolded_guidance = F.unfold(guidance, kernel_size=self.kernel_size, padding=self.padding)  # [B, Cg * K*K, H*W]

        # Reshape for distance computation
        output_center = output.view(B, C, -1).permute(0, 2, 1).unsqueeze(-1)  # [B, HW, C, 1]
        guidance_center = guidance.view(B, -1, H*W).permute(0, 2, 1).unsqueeze(-1)  # [B, HW, Cg, 1]

        patches_output = unfolded_output.view(B, C, self.kernel_size*self.kernel_size, H*W).permute(0, 3, 1, 2)  # [B, HW, C, K*K]
        patches_guidance = unfolded_guidance.view(B, -1, self.kernel_size*self.kernel_size, H*W).permute(0, 3, 1, 2)  # [B, HW, Cg, K*K]

        # Distance in output space (color)
        color_dist = ((patches_output - output_center)**2).sum(2)  # [B, HW, K*K]
        guidance_dist = ((patches_guidance - guidance_center)**2).sum(2)  # [B, HW, K*K]

        weights = torch.exp(-color_dist / self.sigma_color**2 - guidance_dist / self.sigma_guidance**2)  # [B, HW, K*K]

        # Weighted average
        weighted_output = (patches_output * weights.unsqueeze(2)).sum(-1) / (weights.sum(-1, keepdim=True) + 1e-6)  # [B, HW, C]
        result = weighted_output.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]

        return result