import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

class BinomDataset(torch.utils.data.Dataset):
    '''
    Dataset for time-resolved data: [N, T, D, H, W]
    Splits each sample into binomial input and target, normalized as distributions.

    Parameters:
        data (numpy array): [N, T, D, H, W] integer photon counts
        windowSize (int): spatial crop size (H=W)
        depthSize (int): number of depth slices to crop (D)
        minPSNR (float): minimum pseudo PSNR
        maxPSNR (float): maximum pseudo PSNR
        virtSize (int): virtual dataset size
        augment (bool): whether to use data augmentation
        maxProb (float): max binomial sampling probability
    '''
    def __init__(self, data, windowSize, depthSize, minPSNR, maxPSNR, virtSize=None, augment=True, maxProb=0.99):
        super().__init__()
        self.data = torch.from_numpy(data.astype(np.int32))  # [N, T, D, H, W]
        self.crop = transforms.RandomCrop((depthSize, windowSize, windowSize))
        self.minPSNR = minPSNR
        self.maxPSNR = maxPSNR
        self.windowSize = windowSize
        self.depthSize = depthSize
        self.maxProb = maxProb
        self.virtSize = virtSize if virtSize is not None else data.shape[0]
        self.augment = augment

    def __len__(self):
        return self.virtSize

    def __getitem__(self, idx):
        real_idx = np.random.randint(self.data.shape[0]) if self.virtSize else idx
        img = self.data[real_idx]  # [T, D, H, W]

        # Apply 3D random crop
        d_start = np.random.randint(0, img.shape[1] - self.depthSize + 1)
        h_start = np.random.randint(0, img.shape[2] - self.windowSize + 1)
        w_start = np.random.randint(0, img.shape[3] - self.windowSize + 1)
        img = img[:, d_start:d_start + self.depthSize, h_start:h_start + self.windowSize, w_start:w_start + self.windowSize]

        # Sample binomial level based on pseudo PSNR
        uniform = np.random.rand() * (self.maxPSNR - self.minPSNR) + self.minPSNR
        level = (10 ** (uniform / 10.0)) / (img.float().mean().item() + 1e-5)
        level = min(level, self.maxProb)

        # Binomial sampling
        binom = torch.distributions.binomial.Binomial(total_count=img, probs=torch.tensor([level]))
        img_input = binom.sample()[0]  # [T, D, H, W]
        img_target = img.float()

        # Normalize per voxel over time
        P_input = img_input / (img_input.sum(dim=0, keepdim=True) + 1e-8)
        P_target = img_target / (img_target.sum(dim=0, keepdim=True) + 1e-8)

        out = torch.cat([P_target, P_input], dim=0)  # [2*T, D, H, W]

        return out


# Utility: visualize histogram at a voxel

def plot_voxel_histogram(P_target, P_input, voxel=(0, 0, 0)):
    """
    Plot the predicted vs target distribution at a given voxel.
    P_target, P_input: tensors [T, D, H, W]
    voxel: (d, h, w) index
    """
    T = P_target.shape[0]
    d, h, w = voxel
    target_curve = P_target[:, d, h, w].cpu().numpy()
    input_curve = P_input[:, d, h, w].cpu().numpy()

    plt.figure(figsize=(6, 3))
    plt.plot(range(T), target_curve, label='Target (MC)', marker='o')
    plt.plot(range(T), input_curve, label='Input (Noisy)', marker='x')
    plt.xlabel("Time bin")
    plt.ylabel("Photon Probability")
    plt.title(f"Photon Distribution at voxel (d={d}, h={h}, w={w})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
