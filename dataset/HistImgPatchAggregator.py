import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchAggregator(nn.Module):
    """
    PatchAggregator performs patch-based weighted averaging where weights are
    determined by similarity in:
    - Output (e.g., predicted HDR image)
    - One or more feature maps (e.g., spatial features, histogram bins)

    This generalized aggregator enables smoothing based on either spatial guidance (e.g., normals),
    histogram guidance (e.g., color histograms), or a combination of both.

    Parameters:
    - kernel_size: Size of the local window (e.g., 7x7).
    - sigma_color: Controls sensitivity to output (e.g., RGB) differences.
    - sigma_guidance: Controls sensitivity to guidance feature differences (can be multiple).
    
    Inputs:
    - output:   [B, 3, H, W] - The predicted image to be smoothed.
    - features: list of (tensor, sigma) tuples, where:
                tensor: [B, Cf, H, W] guidance features (e.g., spatial, histogram),
                sigma:  scalar controlling weight sensitivity for that feature.

    Output:
    - result: [B, 3, H, W] - Smoothed result.
    """
    def __init__(self, kernel_size=7, sigma_color=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.sigma_color = sigma_color

    def forward(self, output, features=None):
        """
        output:   [B, 3, H, W]
        features: Optional list of (tensor, sigma) pairs, e.g.:
                  [(x_spatial, 0.2), (x_hist, 0.3)]
        """
        B, C, H, W = output.shape

        # Unfold the output image into patches
        unfolded_output = F.unfold(output, kernel_size=self.kernel_size, padding=self.padding)
        patches_output = unfolded_output.view(B, C, self.kernel_size**2, H*W).permute(0, 3, 1, 2)       # [B, HW, C, K*K]
        output_center = output.view(B, C, -1).permute(0, 2, 1).unsqueeze(-1)                            # [B, HW, C, 1]

        # Color (output) distance
        color_dist = ((patches_output - output_center) ** 2).sum(2)  # [B, HW, K*K]
        total_dist = color_dist / (self.sigma_color ** 2 + 1e-8)

        # Add each guidance feature's distance term
        if features is not None:
            for feat, sigma in features:
                Cf = feat.size(1)
                unfolded_feat = F.unfold(feat, kernel_size=self.kernel_size, padding=self.padding)      # [B, Cf*K*K, H*W]
                patches_feat = unfolded_feat.view(B, Cf, self.kernel_size**2, H*W).permute(0, 3, 1, 2)  # [B, HW, Cf, K*K]
                center_feat = feat.view(B, Cf, -1).permute(0, 2, 1).unsqueeze(-1)                       # [B, HW, Cf, 1]
                
                # CHI 2 DISTANCE
                # (same as RHF and BCD)
                numerator = (patches_feat - center_feat) ** 2
                denominator = patches_feat + center_feat + 1e-6                     # prevent div-by-zero
                chi2 = (numerator / denominator).sum(2)                             # [B, HW, K*K]
                total_dist += chi2 / (sigma ** 2 + 1e-8)
                
                # EUCLIDEAN DISTANCE
                # feat_dist = ((patches_feat - center_feat) ** 2).sum(2)            # [B, HW, K*K]
                # total_dist += feat_dist / (sigma ** 2 + 1e-8)

        # Compute weights
        weights = torch.exp(-total_dist)  # [B, HW, K*K]

        # Weighted aggregation
        weighted_output = (patches_output * weights.unsqueeze(2)).sum(-1) / (weights.sum(-1, keepdim=True) + 1e-6)  # [B, HW, C]
        result = weighted_output.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]
        return result
