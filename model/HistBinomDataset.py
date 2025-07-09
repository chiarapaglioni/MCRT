import os
import random
import torch
import numpy as np
import tifffile
from torch.utils.data import Dataset

class HistogramBinomDataset(Dataset):
    """
    This dataset loads histograms, noisy images (low spp) and optionally clean ground truth (high spp).
    The following features are supported: 
    - optional binomial splitting based on binomial_prob
    - optionally applies data augmentation 
    - random crops are sampled randomly images/histograms

    Parameters:
        - root_dir (str): Root directory containing 'train' and/or 'test' folders with scene subdirectories.
        - crop_size (int): Size of random crops to extract from each scene (default: 128).
        - mode (str): Dataset mode, 'train' or 'test'. Controls augmentations and clean image loading.
        - binomial_split (bool): Whether to split the histogram with binomial noise simulation (default: False).
        - binomial_prob (float): Probability parameter `p` for binomial splitting (default: 0.5).
        - apply_augmentations (bool): Whether to apply D4 augmentations (only applies if mode='train').
        - virt_size (int): Virtual size of the dataset (number of samples per epoch). Enables infinite sampling.

    Returns:
        sample (dict): A dictionary with the following keys:
            - 'histogram': Either a tensor (3, B, H, W) or a tuple of two tensors if binomial_split=True.
            - 'noisy': Tensor of the noisy RGB image (3, H, W).
            - 'clean' (optional): Tensor of the clean RGB image (3, H, W), only in test mode.
            - 'scene': Scene name from which the crop was extracted.
    """

    def __init__(
        self,
        root_dir,
        crop_size=128,
        mode='train',
        binomial_split=False,
        binomial_prob=0.5,  
        apply_augmentations=True,
        virt_size=1000  
    ):
        self.root_dir = root_dir
        self.mode = mode
        self.crop_size = crop_size
        self.binomial_split = binomial_split
        self.binomial_prob = binomial_prob  
        self.apply_augmentations = apply_augmentations and mode == 'train'
        self.virt_size = virt_size  

        self.scene_dir = os.path.join(root_dir, mode)
        self.scene_names = sorted(os.listdir(self.scene_dir))

        # Store image dimensions for each scene (for random cropping)
        self.scene_shapes = {}  
        for scene in self.scene_names:
            scene_path = os.path.join(self.scene_dir, scene)
            npz_files = [f for f in os.listdir(scene_path) if f.endswith('.npz')]
            assert len(npz_files) == 1
            with np.load(os.path.join(scene_path, npz_files[0])) as data:
                hist = data[list(data.keys())[0]]
            H, W = hist.shape[:2]
            self.scene_shapes[scene] = (H, W)  

    def __len__(self):
        return self.virt_size  

    def __getitem__(self, idx):
        # Select random scene and crop coordinates
        scene = random.choice(self.scene_names)  
        H, W = self.scene_shapes[scene]  
        y = random.randint(0, H - self.crop_size)  
        x = random.randint(0, W - self.crop_size)  
        scene_path = os.path.join(self.scene_dir, scene)

        # Load histogram
        npz_path = os.path.join(scene_path, [f for f in os.listdir(scene_path) if f.endswith('.npz')][0])
        with np.load(npz_path) as data:
            hist = data[list(data.keys())[0]]  # (H, W, 3, B)

        # Load noisy image (spp=32)
        noisy = tifffile.imread(os.path.join(scene_path, [f for f in os.listdir(scene_path) if f.endswith('spp32.tiff')][0]))

        # Load clean image (spp=1500) if in test mode
        clean = None
        if self.mode == 'test':
            clean = tifffile.imread(os.path.join(scene_path, [f for f in os.listdir(scene_path) if f.endswith('spp1500.tiff')][0]))

        # Extract crops
        hist_crop = hist[y:y + self.crop_size, x:x + self.crop_size]
        noisy_crop = noisy[y:y + self.crop_size, x:x + self.crop_size]
        clean_crop = clean[y:y + self.crop_size, x:x + self.crop_size] if clean is not None else None

        # Binomial splitting (optional)
        if self.binomial_split:
            h1 = np.random.binomial(hist_crop.astype(np.int32), self.binomial_prob)  
            h2 = hist_crop.astype(np.int32) - h1
            hist_crop = (h1, h2)

        # Apply augmentations (D4 group)
        if self.apply_augmentations:
            hist_crop, noisy_crop, clean_crop = self.apply_d4_augmentations(hist_crop, noisy_crop, clean_crop)

        # Convert to torch tensors
        if self.binomial_split:
            h1_tensor = torch.tensor(hist_crop[0], dtype=torch.float32).permute(2, 3, 0, 1)
            h2_tensor = torch.tensor(hist_crop[1], dtype=torch.float32).permute(2, 3, 0, 1)
            hist_tensor = (h1_tensor, h2_tensor)
        else:
            hist_tensor = torch.tensor(hist_crop, dtype=torch.float32).permute(2, 3, 0, 1)

        noisy_tensor = torch.tensor(noisy_crop, dtype=torch.float32).permute(2, 0, 1)
        clean_tensor = torch.tensor(clean_crop, dtype=torch.float32).permute(2, 0, 1) if clean_crop is not None else None

        sample = {
            'histogram': hist_tensor,
            'noisy': noisy_tensor,
            'scene': scene
        }
        if clean_tensor is not None:
            sample['clean'] = clean_tensor

        return sample

    def apply_d4_augmentations(self, hist, noisy, clean):
        aug_id = random.randint(0, 7)

        def transform(img):
            if aug_id >= 4:
                img = np.rot90(img, k=aug_id - 4, axes=(0, 1))
            if aug_id % 4 == 1:
                img = np.flip(img, axis=0)
            elif aug_id % 4 == 2:
                img = np.flip(img, axis=1)
            elif aug_id % 4 == 3:
                img = np.flip(img, axis=(0, 1))
            return img.copy()

        if self.binomial_split:
            h1, h2 = hist
            return (transform(h1), transform(h2)), transform(noisy), transform(clean) if clean is not None else None
        else:
            return transform(hist), transform(noisy), transform(clean) if clean is not None else None