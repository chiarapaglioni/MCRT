import os
import torch
import random
import tifffile
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from dataset.HistogramGenerator import generate_histograms
from utils.utils import standardize_per_image, boxcox_transform, boxcox_and_standardize

import logging
logger = logging.getLogger(__name__)

class HistogramBinomDataset(Dataset):
    def __init__(self, root_dir: str, crop_size: int = 128, mode: str = 'hist',
                 data_augmentation: bool = True, virt_size: int = 1000,
                 low_spp: int = 32, high_spp: int = 4500, hist_bins: int = 8,
                 clean: bool = False, cached_dir: str = None, debug: bool = False, 
                 device: str = None, hist_regeneration: bool = False, scene_names=None, 
                 supervised: bool = False, global_mean = None, global_std = None):
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

        self.image_mean_std = {}  # (scene) -> (mean, std)
        self.global_mean = global_mean
        self.global_std = global_std

        self.supervised = supervised
        if self.supervised:
            logger.info("Supervised (noise2clean) mode")
        else:
            logger.info("Self-Supervised (noise2noise) mode")

        self.spp1_images = {}     # (scene) -> tensor (N, 3, H, W)
        self.hist_features = {}   # (scene) -> np.array (H, W, 3, bins)
        self.noisy_images = {}    # (scene) -> tensor (3, H, W)
        self.clean_images = {}    # (scene) -> tensor (3, H, W)
        self.scene_paths = {}     # (scene) -> folder path
        self.scene_sample_indices = {}  # (scene) -> (list of input idx, target idx)

        if self.cached_dir and not os.path.exists(self.cached_dir):
            os.makedirs(self.cached_dir)

        # Scan scenes and files
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
            spp1_img = tifffile.imread(spp1_path)             # (N=low_spp, H, W, 3)

            # Load noisy image
            noisy_path = os.path.join(folder, noisy_file)
            noisy_img = tifffile.imread(noisy_path)            # (H, W, 3)
            self.noisy_images[key] = torch.from_numpy(noisy_img).permute(2, 0, 1).float()  # (3, H, W)

            self.scene_paths[key] = folder

            # Load clean image if available
            if self.clean and clean_file:
                clean_path = os.path.join(folder, clean_file)
                clean_img = tifffile.imread(clean_path)        # (H, W, 3)
                self.clean_images[key] = torch.from_numpy(clean_img).permute(2, 0, 1).float()  # (3, H, W)

            # Randomly shuffle indices to pick input and target samples from spp1_img
            indices = list(range(self.low_spp))
            random.shuffle(indices)
            input_idx = indices[:-1]
            target_idx = indices[-1]
            self.scene_sample_indices[key] = (input_idx, target_idx)

            # Histogram caching (hist mode only)
            if self.mode == 'hist':
                hist_filename = f"{key}_spp{self.low_spp}_bins{self.hist_bins}_hist.npz"
                cache_path = os.path.join(self.cached_dir, hist_filename) if self.cached_dir else None

                if cache_path and os.path.exists(cache_path) and not self.hist_regeneration:
                    cached = np.load(cache_path)
                    self.hist_features[key] = cached['features']   # (H, W, 3, bins)
                else:
                    logger.info(f"Generating Histogram: {hist_filename}")
                    input_samples_raw = spp1_img[input_idx]       # (N-1, H, W, 3)

                    # generate_histograms returns (H, W, 3, bins), bins is hist_bins
                    hist, _ = generate_histograms(input_samples_raw, self.hist_bins, self.device)
                    hist = hist.astype(np.float32)

                    hist_sum = np.sum(hist, axis=-1, keepdims=True)  # (H, W, 3, 1)
                    hist /= (hist_sum + 1e-8)

                    self.hist_features[key] = hist
                    if cache_path:
                        np.savez_compressed(cache_path, features=hist)

            # Convert spp1 images to tensor (N, 3, H, W)
            spp1_tensor = torch.from_numpy(spp1_img).permute(0, 3, 1, 2).float()  # (N, 3, H, W)

            if self.mode == 'hist':
                # Standardize per image for hist mode
                spp1_tensor_std, mean, std = standardize_per_image(spp1_tensor)  # same shape, mean/std: (3, 1, 1)
                self.spp1_images[key] = spp1_tensor_std
                self.image_mean_std[key] = (mean, std)
            else:
                # Store raw tensor, normalization will be dynamic for img mode
                self.spp1_images[key] = spp1_tensor
                self.image_mean_std[key] = (None, None)

    def __len__(self):
        return self.virt_size

    def __getitem__(self, idx, crop_coords=None):
        scene = self.scene_names[idx % len(self.scene_names)]

        spp1_img = self.spp1_images[scene]           # (N, 3, H, W)
        noisy_tensor = self.noisy_images[scene]      # (3, H, W)
        clean_tensor = self.clean_images.get(scene, None)  # (3, H, W) or None

        input_idx, target_idx = self.scene_sample_indices[scene]
        input_samples = spp1_img[input_idx]           # (N-1, 3, H, W)
        target_sample = spp1_img[target_idx]          # (3, H, W)

        # Per Scene Mean --> Currently not used
        # mean, std = self.image_mean_std[scene]

        # Hyperparameters for boxcox transform
        lambda_val = 0.5
        epsilon = 1e-6

        if self.mode == 'hist':
            features = self.hist_features[scene]  # (H, W, 3, bins)

            input_np = input_samples.permute(0, 2, 3, 1).numpy()
            mean_img = input_np.mean(axis=0)   # (H, W, 3)
            var_img = input_np.var(axis=0)     # (H, W, 3)

            mean_std_norm, mean_mean, mean_std = boxcox_and_standardize(
                torch.from_numpy(mean_img).float(), global_mean=self.global_mean, global_std=self.global_std
            )
            var_std_norm, var_mean, var_std = boxcox_and_standardize(
                torch.from_numpy(var_img).float(), global_mean=self.global_mean, global_std=self.global_std
            )

            mean_std_norm = mean_std_norm.numpy()[..., None]
            var_std_norm = var_std_norm.numpy()[..., None]

            concat_features = np.concatenate([features, mean_std_norm, var_std_norm], axis=-1)
            input_tensor = torch.from_numpy(np.transpose(concat_features, (2, 3, 0, 1))).float()

            target_tensor, _, _ = boxcox_and_standardize(
                target_sample, global_mean=self.global_mean, global_std=self.global_std
            )

            mean = mean_mean
            std = mean_std

        else:
            input_bc = boxcox_transform(input_samples, lmbda=lambda_val, epsilon=epsilon)
            input_cuboid = input_bc.permute(0, 2, 3, 1).reshape(-1, 3)

            # global mean
            if self.global_mean is not None and self.global_std is not None:
                cuboid_mean = self.global_mean.view(3)
                cuboid_std = self.global_std.view(3) + 1e-8
            # cuboid mean over the batch
            else:
                cuboid_mean = input_cuboid.mean(dim=0)
                cuboid_std = input_cuboid.std(dim=0) + 1e-8

            norm = (input_cuboid - cuboid_mean) / cuboid_std
            norm_img = norm.view(input_samples.shape[0], input_samples.shape[2], input_samples.shape[3], 3)
            norm_img = norm_img.permute(0, 3, 1, 2)  # (N-1, 3, H, W)

            # average tensor for image mode
            input_tensor = norm_img.mean(dim=0)
            mean = cuboid_mean.view(3, 1, 1)
            std = cuboid_std.view(3, 1, 1)

            # TODO: try only tonemapping input + using MSE realtive 
            # Boxcox transform target sample and normalize
            if self.supervised and clean_tensor is not None:
                clean_bc = boxcox_transform(clean_tensor, lmbda=lambda_val, epsilon=epsilon)    # (3, H, W)
                target_tensor = (clean_bc - mean) / std                                         # (3, H, W)
            else:
                target_bc = boxcox_transform(target_sample, lmbda=lambda_val, epsilon=epsilon)  # (3, H, W)
                target_tensor = (target_bc - mean) / std                                        # (3, H, W)

        # CROP
        if self.crop_size:
            if crop_coords is None:
                i, j, h, w = transforms.RandomCrop.get_params(target_tensor, output_size=(self.crop_size, self.crop_size))
            else:
                i, j, h, w = crop_coords

            input_tensor = input_tensor[..., i:i+h, j:j+w]    # crop spatial dims
            target_tensor = target_tensor[..., i:i+h, j:j+w]
            noisy_tensor = noisy_tensor[..., i:i+h, j:j+w]
            if clean_tensor is not None:
                clean_tensor = clean_tensor[..., i:i+h, j:j+w]

        # DATA AUGMENTATION: random horizontal/vertical flips
        if self.data_augmentation:
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-1])    # horizontal flip
                target_tensor = torch.flip(target_tensor, dims=[-1])
                noisy_tensor = torch.flip(noisy_tensor, dims=[-1])
                if clean_tensor is not None:
                    clean_tensor = torch.flip(clean_tensor, dims=[-1])
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-2])    # vertical flip
                target_tensor = torch.flip(target_tensor, dims=[-2])
                noisy_tensor = torch.flip(noisy_tensor, dims=[-2])
                if clean_tensor is not None:
                    clean_tensor = torch.flip(clean_tensor, dims=[-2])

        return {
            "input": input_tensor,          # (3, crop_size, crop_size) or (3, H, W)
            "target": target_tensor,        # (3, crop_size, crop_size) or (3, H, W)
            "noisy": noisy_tensor,          # (3, crop_size, crop_size) or (3, H, W)
            "clean": clean_tensor if clean_tensor is not None else None,  # (3, crop_size, crop_size) or None
            "scene": scene,
            "crop_coords": (i, j, h, w) if self.crop_size else None,              
            "lambda": lambda_val,           # (3, 1, 1) or None
            "epsilon": epsilon,             # (3, 1, 1) or None
            "mean": mean,                   # (3, 1, 1) or None
            "std": std                      # (3, 1, 1) or None
        }