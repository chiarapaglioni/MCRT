import os
import torch
import random
import tifffile
import numpy as np
from torch.utils.data import Dataset

class HistogramDenoisingDataset(Dataset):
    def __init__(
        self,
        root_dir,
        crop_size=128,
        stride=None,
        mode='train',
        binomial_split=False,
        apply_augmentations=True
    ):
        self.root_dir = root_dir
        self.mode = mode
        self.crop_size = crop_size
        self.stride = stride if stride is not None else crop_size
        self.binomial_split = binomial_split
        self.apply_augmentations = apply_augmentations and mode == 'train'

        self.scene_dir = os.path.join(root_dir, mode)
        self.scene_names = sorted(os.listdir(self.scene_dir))  # subfolders per scene

        self.samples = []  # list of (scene_name, x, y)

        for scene in self.scene_names:
            scene_path = os.path.join(self.scene_dir, scene)
            npz_files = [f for f in os.listdir(scene_path) if f.endswith('.npz')]
            assert len(npz_files) == 1, f"Expected 1 .npz file in {scene_path}, found {len(npz_files)}"
            npz_path = os.path.join(scene_path, npz_files[0])

            with np.load(npz_path) as data:
                hist = data[list(data.keys())[0]]  # assumes only one entry or the first is used

            H, W = hist.shape[:2]
            for y in range(0, H - crop_size + 1, self.stride):
                for x in range(0, W - crop_size + 1, self.stride):
                    self.samples.append((scene, x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene, x, y = self.samples[idx]
        scene_path = os.path.join(self.scene_dir, scene)

        # Load histogram
        npz_files = [f for f in os.listdir(scene_path) if f.endswith('.npz')]
        npz_path = os.path.join(scene_path, npz_files[0])
        with np.load(npz_path) as data:
            hist = data[list(data.keys())[0]]  # shape: (H, W, 3, B)

        # Load noisy TIFF (spp=32)
        noisy_files = [f for f in os.listdir(scene_path) if f.endswith('spp32.tiff')]
        assert len(noisy_files) == 1, f"Expected 1 spp32 file in {scene_path}, found {len(noisy_files)}"
        noisy = tifffile.imread(os.path.join(scene_path, noisy_files[0]))  # (H, W, 3)

        # Load clean TIFF (spp=1500) only for test
        clean = None
        if self.mode == 'test':
            clean_files = [f for f in os.listdir(scene_path) if f.endswith('spp1500.tiff')]
            assert len(clean_files) == 1, f"Expected 1 spp1500 file in {scene_path}, found {len(clean_files)}"
            clean = tifffile.imread(os.path.join(scene_path, clean_files[0]))

        # Crop selection
        hist_crop = hist[y:y + self.crop_size, x:x + self.crop_size]
        noisy_crop = noisy[y:y + self.crop_size, x:x + self.crop_size]
        clean_crop = clean[y:y + self.crop_size, x:x + self.crop_size] if clean is not None else None

        # Binomial histogram split (optional)
        if self.binomial_split:
            h1 = np.random.binomial(hist_crop.astype(np.int32), 0.5)
            h2 = hist_crop.astype(np.int32) - h1
            hist_crop = (h1, h2)

        # Data augmentation (optional)
        if self.apply_augmentations:
            hist_crop, noisy_crop, clean_crop = self.apply_d4_augmentations(hist_crop, noisy_crop, clean_crop)

        # Convert to torch tensors
        if self.binomial_split:
            h1_tensor = torch.tensor(hist_crop[0], dtype=torch.float32).permute(2, 3, 0, 1)  # (3, B, H, W)
            h2_tensor = torch.tensor(hist_crop[1], dtype=torch.float32).permute(2, 3, 0, 1)
            hist_tensor = (h1_tensor, h2_tensor)
        else:
            hist_tensor = torch.tensor(hist_crop, dtype=torch.float32).permute(2, 3, 0, 1)  # (3, B, H, W)

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