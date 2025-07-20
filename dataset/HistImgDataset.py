import os
import torch
import random
import tifffile
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from dataset.HistogramGenerator import generate_histograms
from utils.utils import standardize_per_image

# Logger
import logging
logger = logging.getLogger(__name__)


class HistogramBinomDataset(Dataset):
    def __init__(self, root_dir: str, crop_size: int = 128, mode: str = 'hist',
                data_augmentation: bool = True, virt_size: int = 1000,
                low_spp: int = 32, high_spp: int = 4500, hist_bins: int = 8,
                clean: bool = False, cached_dir: str = None, debug: bool = False, 
                device: str = None, hist_regeneration: bool = False, scene_names=None):
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
        self.hist_regeneration = hist_regeneration

        self.image_mean_std = {}  # add this in __init__

        self.spp1_images = {}
        self.hist_features = {}
        self.noisy_images = {}
        self.clean_images = {}
        self.scene_paths = {}
        self.scene_sample_indices = {}

        if self.cached_dir and not os.path.exists(self.cached_dir):
            os.makedirs(self.cached_dir)

        # Search for .tiff files inside subfolders
        scene_keys = []
        for subdir in sorted(os.listdir(self.root_dir)):
            full_subdir = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(full_subdir):
                continue
            for fname in os.listdir(full_subdir):
                if fname.endswith(f"spp1x{self.low_spp}.tiff"):
                    key = fname.split("_spp")[0]
                    scene_keys.append((key, full_subdir))

        all_scenes = sorted(set(key for key, _ in scene_keys))

        if scene_names is not None:
            # Filter scene_keys to only those in scene_names list
            scene_names_set = set(scene_names)
            scene_keys = [(key, folder) for key, folder in scene_keys if key in scene_names_set]
            self.scene_names = sorted(scene_names_set.intersection(all_scenes))
        else:
            self.scene_names = all_scenes
        assert self.scene_names, f"No scenes found in {self.root_dir}"

        logger.info(f"{len(self.scene_names)} scenes: {self.scene_names}")

        for key, folder in scene_keys:
            spp1_file = next((f for f in os.listdir(folder) if f.startswith(key) and f.endswith(f"spp1x{self.low_spp}.tiff")), None)
            noisy_file = next((f for f in os.listdir(folder) if f.startswith(key) and f.endswith(f"spp{self.low_spp}.tiff")), None)
            clean_file = next((f for f in os.listdir(folder) if f.startswith(key) and f.endswith(f"spp{self.high_spp}.tiff")), None)

            assert spp1_file and noisy_file, f"Missing files for scene: {key} in {folder}"
            
            # Load spp1xN images
            spp1_path = os.path.join(folder, spp1_file)
            spp1_img = tifffile.imread(spp1_path)               # (low_spp, H, W, 3)
            
            # Load sppN image
            noisy_path = os.path.join(folder, noisy_file)
            noisy_img = tifffile.imread(noisy_path)             # (H, W, 3)
            self.noisy_images[key] = torch.from_numpy(noisy_img).permute(2, 0, 1).float()

            self.scene_paths[key] = folder
            
            # Load Clean Images
            if self.clean and clean_file:
                clean_path = os.path.join(folder, clean_file)
                clean_img = tifffile.imread(clean_path)
                self.clean_images[key] = torch.from_numpy(clean_img).permute(2, 0, 1).float()

            # Select random input(N-1)/target(1) indices for this scene
            indices = list(range(self.low_spp))
            random.shuffle(indices)
            input_idx = indices[:-1]
            target_idx = indices[-1]
            self.scene_sample_indices[key] = (input_idx, target_idx)

            # Histogram Caching (hist mode only)
            if self.mode == 'hist':
                hist_filename = f"{key}_spp{self.low_spp}_bins{self.hist_bins}_hist.npz"
                cache_path = os.path.join(self.cached_dir, hist_filename) if self.cached_dir else None

                if cache_path and os.path.exists(cache_path) and not self.hist_regeneration:
                    cached = np.load(cache_path)
                    self.hist_features[key] = cached['features']
                else:
                    logger.info(f"Generating Histogram: {hist_filename}")
                    input_samples_raw = spp1_img[input_idx]  # shape: (N-1, H, W, 3)

                    # generate histogram - shape: (H, W, 3, B)
                    hist, _ = generate_histograms(input_samples_raw, self.hist_bins, self.device)
                    hist = hist.astype(np.float32)

                    hist_sum = np.sum(hist, axis=-1, keepdims=True)
                    hist /= (hist_sum + 1e-8)

                    self.hist_features[key] = hist
                    if cache_path:
                        np.savez_compressed(cache_path, features=hist)

            # NORMALIZATION
            # Convert to tensor with channels first: (N, H, W, 3) -> (N, 3, H, W)
            spp1_tensor = torch.from_numpy(spp1_img).permute(0, 3, 1, 2).float()

            spp1_tensor_std, mean, std = standardize_per_image(spp1_tensor)
            self.spp1_images[key] = spp1_tensor_std

            # Later, store per scene:
            self.image_mean_std[key] = (mean, std)



    def __len__(self):
        return self.virt_size

    def __getitem__(self, idx, crop_coords=None):
        scene = self.scene_names[idx % len(self.scene_names)]
        spp1_img = self.spp1_images[scene]
        noisy_tensor = self.noisy_images[scene]
        clean_tensor = self.clean_images.get(scene, None)

        # Get Input and Target Samples
        input_idx, target_idx = self.scene_sample_indices[scene]
        input_samples = spp1_img[input_idx]  # tensor: (N-1, C, H, W)
        target_sample = spp1_img[target_idx] # tensor: (C, H, W)

        mean, std = self.image_mean_std[scene]

        if self.mode == 'hist':
            features = self.hist_features[scene]  # (H, W, 3, bins)

            # Compute on-the-fly mean & var from input_samples
            input_samples_unstd = input_samples * std + mean
            input_samples_np = input_samples_unstd.permute(0, 2, 3, 1).numpy()

            mean_img = input_samples_np.mean(axis=0, keepdims=True)[..., None]  # (1, H, W, 3, 1)
            var_img = input_samples_np.var(axis=0, keepdims=True)[..., None]

            mean_img = mean_img.squeeze(0)  # (H, W, 3, 1)
            var_img = var_img.squeeze(0)

            concat_features = np.concatenate([features, mean_img, var_img], axis=-1)  # (H, W, 3, bins+2)
            input_tensor = torch.from_numpy(np.transpose(concat_features, (2, 3, 0, 1))).float()  # (3, bins+2, H, W)

        else:
            input_avg = input_samples.mean(dim=0)  # average (3, H, W)
            input_tensor = input_avg

        target_tensor = target_sample

        # CROP
        if self.crop_size:
            if crop_coords is None:
                i, j, h, w = transforms.RandomCrop.get_params(target_tensor, output_size=(self.crop_size, self.crop_size))
            else:
                i, j, h, w = crop_coords

            input_tensor = input_tensor[:, :, i:i+h, j:j+w] if self.mode == 'hist' else input_tensor[:, i:i+h, j:j+w]
            target_tensor = target_tensor[:, i:i+h, j:j+w]
            noisy_tensor = noisy_tensor[:, i:i+h, j:j+w]
            if clean_tensor is not None:
                clean_tensor = clean_tensor[:, i:i+h, j:j+w]

        # DATA AUGMENTATION
        if self.data_augmentation:
            # Random horizontal flip
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-1])
                target_tensor = torch.flip(target_tensor, dims=[-1])
                noisy_tensor = torch.flip(noisy_tensor, dims=[-1])
                if clean_tensor is not None:
                    clean_tensor = torch.flip(clean_tensor, dims=[-1])

            # Random vertical flip
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-2])
                target_tensor = torch.flip(target_tensor, dims=[-2])
                noisy_tensor = torch.flip(noisy_tensor, dims=[-2])
                if clean_tensor is not None:
                    clean_tensor = torch.flip(clean_tensor, dims=[-2])

        return {
            "input": input_tensor,
            "target": target_tensor,
            "noisy": noisy_tensor,
            "clean": clean_tensor if clean_tensor is not None else None,
            "scene": scene,
            "crop_coords": (i, j, h, w) if self.crop_size else None,
            "image_mean": mean,  # shape (3, 1, 1)
            "image_std": std,    # shape (3, 1, 1)
        }