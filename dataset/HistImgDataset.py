import os
import torch
import random
import tifffile
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from dataset.HistogramGenerator import generate_histograms

import logging
logger = logging.getLogger(__name__)

class HistogramBinomDataset(Dataset):
    def __init__(self, root_dir: str, crop_size: int = 128, mode: str = 'hist',
                 data_augmentation: bool = True, virt_size: int = 1000,
                 low_spp: int = 32, high_spp: int = 4500, hist_bins: int = 8,
                 clean: bool = False, cached_dir: str = None, debug: bool = False, 
                 device: str = None, hist_regeneration: bool = False, scene_names=None, 
                 supervised: bool = False, global_mean = None, global_std = None, 
                 tonemap: str = None, target_split: int = 1, run_mode: str = None):
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
        self.tonemap = tonemap
        self.target_split = target_split
        self.run_mode = run_mode

        # mean/std for normalisation
        # try normalising by only diviing by the mean per image
        self.image_mean_std = {}  # (scene) -> (mean, std)
        self.global_mean = global_mean
        self.global_std = global_std

        self.supervised = supervised
        if self.supervised:
            logger.info("Supervised (noise2clean) mode")
        else:
            logger.info("Self-Supervised (noise2noise) mode")

        self.spp1_images = {}           # (scene) -> tensor (N, 3, H, W)
        self.hist_features = {}         # (scene) -> np.array (H, W, 3, bins)
        self.noisy_images = {}          # (scene) -> tensor (3, H, W)
        self.clean_images = {}          # (scene) -> tensor (3, H, W)
        self.scene_paths = {}           # (scene) -> folder path
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
            input_idx = indices[:self.target_split]
            target_idx = indices[-self.target_split:]
            self.scene_sample_indices[key] = (input_idx, target_idx)

            # Histogram caching (hist mode only)
            if self.mode == 'hist':
                hist_filename = f"{key}_spp{self.low_spp}_bins{self.hist_bins}_hist.npz"
                cache_path = os.path.join(self.cached_dir, hist_filename) if self.cached_dir else None

                if cache_path and os.path.exists(cache_path) or not self.hist_regeneration:
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
            spp1_array = np.transpose(spp1_img, (0, 3, 1, 2)).astype(np.float32)  # (N, 3, H, W)
            self.spp1_images[key] = spp1_array

    def __len__(self):
        return self.virt_size

    def __getitem__(self, idx=None, crop_coords=None):
        if self.run_mode == "train":        # training
            scene = random.choice(self.scene_names)
        else:                               # eval
            scene = self.scene_names[idx % len(self.scene_names)]
            
        spp1_img = self.spp1_images[scene]                  # (low_spp, H, W, 3)
        noisy_tensor = self.noisy_images[scene]             # (3, H, W)
        clean_tensor = self.clean_images.get(scene, None)   # (3, H, W)

        input_idx, target_idx = self.scene_sample_indices[scene]
        input_samples = spp1_img[input_idx]           # (low_spp-N, 3, H, W)
        target_sample = spp1_img[target_idx]          # (N, 3, H, W)
        
        if self.mode == "hist":
            hist_norm = self.hist_features[scene]                           # (H, W, 3, bins)

            input_samples_tensor = torch.from_numpy(input_samples).float()  # (N-1, 3, H, W)
            mean = input_samples_tensor.mean(dim=0)                         # (3, H, W)
            std = input_samples_tensor.std(dim=0)                           # (3, H, W)

            # Histogram from numpy → torch, (H, W, 3, bins) → (3, bins, H, W)
            hist_torch = torch.from_numpy(hist_norm).permute(2, 3, 0, 1).float()
            # input_tensor = torch.cat([hist_torch, mean_t, std_t], dim=1)  # (3, bins+2, H, W)
            # TODO: only train network to map normalised hist to noisy image
            input_tensor = hist_torch
        else:
            input_avg = input_samples.mean(axis=0)                      # (3, H, W)
            input_tensor = torch.from_numpy(input_avg).float()          # (3, H, W)

        
        # TARGET
        if self.supervised:
            target_tensor = clean_tensor   
        else: 
            target_avg = target_sample.mean(axis=0)                     # (3, H, W)
            target_tensor = torch.from_numpy(target_avg).float()        # (3, H, W) 


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
            # Remove or comment this part out to avoid upside-down flips
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-2])    # vertical flip
                target_tensor = torch.flip(target_tensor, dims=[-2])
                noisy_tensor = torch.flip(noisy_tensor, dims=[-2])
                if clean_tensor is not None:
                    clean_tensor = torch.flip(clean_tensor, dims=[-2])


        return {
            "input": input_tensor,                                          # (3, crop_size, crop_size) or (3, H, W)
            "target": target_tensor,                                        # (3, crop_size, crop_size) or (3, H, W)
            "noisy": noisy_tensor,                                          # (3, crop_size, crop_size) or (3, H, W)
            "clean": clean_tensor if clean_tensor is not None else None,    # (3, crop_size, crop_size) or None
            "scene": scene,
            "crop_coords": (i, j, h, w) if self.crop_size else None,              
        }