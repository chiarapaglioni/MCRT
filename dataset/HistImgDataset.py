import os
import torch
import random
import tifffile
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from dataset.HistogramGenerator import generate_histograms
from utils.utils import standardize_image


class HistogramBinomDataset(Dataset):
    def __init__(self, root_dir: str, crop_size: int = 128, mode: str = 'hist',
                data_augmentation: bool = True, virt_size: int = 1000,
                low_spp: int = 32, high_spp: int = 4500, hist_bins: int = 8,
                clean: bool = False, cached_dir: str = None, debug: bool = False, device: str = None):
        self.root_dir = root_dir
        self.mode = mode
        self.crop_size = crop_size
        self.data_augmentation = data_augmentation
        self.virt_size = virt_size
        self.low_spp = low_spp
        self.high_spp = high_spp
        self.hist_bins = hist_bins
        self.clean = clean
        self.cached_dir = cached_dir
        self.debug = debug
        self.device = device or torch.device("cpu")

        self.spp1_images = {}
        self.noisy_images = {}
        self.clean_images = {}
        self.scene_paths = {}  # full path for each logical scene

        # Search for .tiff files inside all subfolders
        scene_keys = []
        for subdir in sorted(os.listdir(self.root_dir)):
            full_subdir = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(full_subdir):
                continue
            for fname in os.listdir(full_subdir):
                if fname.endswith(f"spp1x{self.low_spp}.tiff"):
                    key = fname.split(f"_spp")[0]  # e.g., 'scene1-A'
                    scene_keys.append((key, full_subdir))

        # De-duplicate and store
        self.scene_names = sorted(set(key for key, _ in scene_keys))
        assert self.scene_names, f"No scenes found in {self.root_dir}"

        print(f"{len(self.scene_names)} scenes: ", self.scene_names)

        if self.cached_dir and not os.path.exists(self.cached_dir):
            os.makedirs(self.cached_dir)

        # Load and Normalise Data
        for key, folder in scene_keys:
            #  Get Images Paths
            spp1_file = next((f for f in os.listdir(folder) if f.startswith(key) and f.endswith(f"spp1x{self.low_spp}.tiff")), None)
            noisy_file = next((f for f in os.listdir(folder) if f.startswith(key) and f.endswith(f"spp{self.low_spp}.tiff")), None)
            clean_file = next((f for f in os.listdir(folder) if f.startswith(key) and f.endswith(f"spp{self.high_spp}.tiff")), None)

            assert spp1_file and noisy_file, f"Missing files for scene: {key} in {folder}"

            spp1_path = os.path.join(folder, spp1_file)
            noisy_path = os.path.join(folder, noisy_file)

            spp1_img = tifffile.imread(spp1_path)  # (low_spp, H, W, 3)
            noisy_img = tifffile.imread(noisy_path)  # (H, W, 3)

            # Standardize/Normalise Images
            spp1_img = standardize_image(spp1_img)
            noisy_img = standardize_image(noisy_img)

            self.scene_paths[key] = folder
            self.spp1_images[key] = spp1_img
            self.noisy_images[key] = torch.from_numpy(noisy_img).permute(2, 0, 1).float()

            if self.clean and clean_file:
                clean_path = os.path.join(folder, clean_file)
                clean_img = tifffile.imread(clean_path)
                clean_img = standardize_image(clean_img)
                self.clean_images[key] = torch.from_numpy(clean_img).permute(2, 0, 1).float()


    def __len__(self):
        return self.virt_size


    def __getitem__(self, idx, seed=None, crop_coords=None):
        if seed is not None:
            random.seed(seed)  # make shuffle deterministic

        scene = self.scene_names[idx % len(self.scene_names)]
        spp1_img = self.spp1_images[scene]
        noisy_tensor = self.noisy_images[scene]
        clean_tensor = self.clean_images.get(scene, None)

        # Consistent sample selection
        indices = list(range(self.low_spp))
        random.shuffle(indices)
        input_indices = indices[:-1]
        target_index = indices[-1]

        input_samples = spp1_img[input_indices]  # (N-1, H, W, 3)
        target_sample = spp1_img[target_index]   # (H, W, 3)

        # HIST MODE
        if self.mode == 'hist':
            full_hist, _ = generate_histograms(input_samples, self.hist_bins, self.device)
            full_hist = full_hist.astype(np.float32)
            full_hist /= (np.sum(full_hist, axis=-1, keepdims=True) + 1e-8)

            full_mean = input_samples.mean(axis=0)[..., None]
            full_var = input_samples.var(axis=0)[..., None]

            features = np.concatenate([full_hist, full_mean, full_var], axis=-1)

            # Normalize Histiogram
            hist = features[..., :self.hist_bins]
            mean = features[..., self.hist_bins:self.hist_bins+1]
            var = features[..., self.hist_bins+1:self.hist_bins+2]
            hist /= (np.sum(hist, axis=-1, keepdims=True) + 1e-8)
            features = np.concatenate([hist, mean, var], axis=-1)

            input_tensor = torch.from_numpy(np.transpose(features, (2, 3, 0, 1))).float()

        # IMG MODE
        else:
            input_avg = input_samples.mean(axis=0)
            input_tensor = torch.from_numpy(input_avg).permute(2, 0, 1).float()

        # Target
        target_tensor = torch.from_numpy(target_sample).permute(2, 0, 1).float()

        # CROP
        if self.crop_size:
            if crop_coords is None:
                i, j, h, w = transforms.RandomCrop.get_params(target_tensor, output_size=(self.crop_size, self.crop_size))
            else:
                i, j, h, w = crop_coords

            if self.mode == 'hist':
                input_tensor = input_tensor[:, :, i:i+h, j:j+w]
            else:
                input_tensor = input_tensor[:, i:i+h, j:j+w]

            target_tensor = target_tensor[:, i:i+h, j:j+w]
            noisy_tensor = noisy_tensor[:, i:i+h, j:j+w]
            if clean_tensor is not None:
                clean_tensor = clean_tensor[:, i:i+h, j:j+w]

        # FINAL DICT
        return {
            "input": input_tensor,
            "target": target_tensor,
            "noisy": noisy_tensor,
            "clean": clean_tensor if clean_tensor is not None else None,
            "scene": scene,
            "crop_coords": (i, j, h, w) if self.crop_size else None
        }
