import torch
import numpy as np
from torch.utils.data import Dataset

class BinomDataset3D(Dataset):
    """
    Dataset for [C, T, D, H, W] data.
    Applies per-voxel binomial sampling to simulate noisy input.
    Returns tensor of shape [2C, T, D, H, W]: [residual, noisy]
    """
    def __init__(self, data, patch_size=32, minPSNR=-40, maxPSNR=-5, virtSize=None, augment=True, maxProb=0.99):
        self.data = torch.from_numpy(data.astype(np.int32))  # [C, T, D, H, W]
        self.C, self.T, self.D, self.H, self.W = self.data.shape
        self.patch_size = patch_size
        self.minPSNR = minPSNR
        self.maxPSNR = maxPSNR
        self.maxProb = maxProb
        self.virtSize = virtSize
        self.augment = augment

    def __len__(self):
        return self.virtSize if self.virtSize else self.D * self.H * self.W

    def __getitem__(self, idx):
        # Random crop indices
        z = np.random.randint(0, self.D - self.patch_size + 1)
        y = np.random.randint(0, self.H - self.patch_size + 1)
        x = np.random.randint(0, self.W - self.patch_size + 1)

        # Extract patch [C, T, d, h, w]
        patch = self.data[:, :, z:z+self.patch_size, y:y+self.patch_size, x:x+self.patch_size]

        # Convert to float
        patch = patch.type(torch.float)

        # Compute target mean for PSNR-based binomial level
        mean_val = patch.mean().item() + 1e-5
        uniform = np.random.rand() * (self.maxPSNR - self.minPSNR) + self.minPSNR
        level = (10**(uniform / 10.0)) / mean_val
        level = min(level, self.maxProb)

        # Perform binomial sampling
        binom = torch.distributions.binomial.Binomial(total_count=patch, probs=torch.tensor([level]))
        noisy = binom.sample().squeeze(-1)  # match shape [C, T, d, h, w]

        # Compute residual
        residual = patch - noisy

        # Normalize
        noisy_norm = noisy / (noisy.mean() + 1e-8)
        residual_norm = residual / (residual.mean() + 1e-8)

        # Stack: [2C, T, d, h, w]
        out = torch.cat([residual_norm, noisy_norm], dim=0)

        # Optional 3D flips (for augmentation)
        if self.augment:
            if np.random.rand() < 0.5:
                out = torch.flip(out, dims=[2])  # flip D
            if np.random.rand() < 0.5:
                out = torch.flip(out, dims=[3])  # flip H
            if np.random.rand() < 0.5:
                out = torch.flip(out, dims=[4])  # flip W

        return out