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


# IMAGE DENOISING IMAGE DATASET
class ImageDataset(Dataset):
    def __init__(self, root_dir: str, crop_size: int = 128, mode: str = 'hist',
                 data_augmentation: bool = True, virt_size: int = 1000,
                 low_spp: int = 32, high_spp: int = 4500, hist_bins: int = 8,
                 clean: bool = False, aov: bool = False, cached_dir: str = None, debug: bool = False, 
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
        self.aov = aov
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
        self.albedo_images = {}         # (scene) -> tensor (3, H, W)
        self.normal_images = {}         # (scene) -> tensor (3, H, W)
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
            albedo_file = next((f for f in os.listdir(folder) if f.endswith("albedo.tiff")), None)
            normal_file = next((f for f in os.listdir(folder) if f.endswith("normal.tiff")), None)

            assert spp1_file and noisy_file, f"Missing files for scene: {key} in {folder}"

            # Load spp1xN images
            spp1_path = os.path.join(folder, spp1_file)
            spp1_img = tifffile.imread(spp1_path)                                                  # (N=low_spp, H, W, 3)

            # NOISY
            noisy_path = os.path.join(folder, noisy_file)
            noisy_img = tifffile.imread(noisy_path)                                                # (H, W, 3)
            self.noisy_images[key] = torch.from_numpy(noisy_img).permute(2, 0, 1).float()          # (3, H, W)

            self.scene_paths[key] = folder

            # CLEAN
            if self.clean and clean_file:
                clean_path = os.path.join(folder, clean_file)
                clean_img = tifffile.imread(clean_path)                                             # (H, W, 3)
                self.clean_images[key] = torch.from_numpy(clean_img).permute(2, 0, 1).float()       # (3, H, W)

            # AOV
            if self.aov:
                # ALBEDO
                albedo_path = os.path.join(folder, albedo_file)
                albedo_img = tifffile.imread(albedo_path)                                           # (H, W, 3)
                self.albedo_images[key] = torch.from_numpy(albedo_img).permute(2, 0, 1).float()     # (3, H, W)
                # NORMAL
                normal_path = os.path.join(folder, normal_file)
                normal_img = tifffile.imread(normal_path)                                           # (H, W, 3)
                self.normal_images[key] = torch.from_numpy(normal_img).permute(2, 0, 1).float()     # (3, H, W)

            # Convert spp1 images to tensor (N, 3, H, W)
            spp1_array = np.transpose(spp1_img, (0, 3, 1, 2)).astype(np.float32)  # (N, 3, H, W)
            self.spp1_images[key] = spp1_array

    def __len__(self):
        return self.virt_size

    def __getitem__(self, idx=None, crop_coords=None):
        if self.run_mode == "train":        # TRAINING
            scene = random.choice(self.scene_names)
        else:                               # EVAL
            scene = self.scene_names[idx % len(self.scene_names)]
            
        spp1_img = self.spp1_images[scene]                      # (low_spp, H, W, 3)
        noisy_tensor = self.noisy_images[scene]                 # (3, H, W)
        clean_tensor = self.clean_images.get(scene, None)       # (3, H, W)
        albedo_tensor = self.albedo_images.get(scene, None)     # (3, H, W)
        normal_tensor = self.normal_images.get(scene, None)     # (3, H, W)

        # Randomly shuffle indices to pick input and target samples from spp1_img
        indices = list(range(self.low_spp))
        random.shuffle(indices)
        input_idx = indices[:self.target_split]
        target_idx = indices[-self.target_split:]
        input_samples = spp1_img[input_idx]                         # (low_spp-N, 3, H, W)
        target_sample = spp1_img[target_idx]                        # (N, 3, H, W)

        # INPUT (average of input samples)
        input_avg = input_samples.mean(axis=0)                      # (3, H, W)
        input_tensor = torch.from_numpy(input_avg).float()          # (3, H, W)

        if self.aov:
            # Concatenate albedo and normal tensors
            input_tensor = torch.cat([input_tensor, albedo_tensor, normal_tensor], dim=0)  # (9, H, W)

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
    


# IMAGE DENOISING - HISTOGRAM DATASET
class HistogramDataset(Dataset):
    def __init__(self, root_dir: str, crop_size: int = 128, mode: str = 'hist',
                 data_augmentation: bool = True, virt_size: int = 1000,
                 low_spp: int = 32, high_spp: int = 4500, hist_bins: int = 8,
                 clean: bool = False, aov: bool = False, cached_dir: str = None, debug: bool = False, 
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
        self.aov = aov
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
        self.albedo_images = {}         # (scene) -> tensor (3, H, W)
        self.normal_images = {}         # (scene) -> tensor (3, H, W)
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
            albedo_file = next((f for f in os.listdir(folder) if f.endswith(f"albedo.tiff")), None)
            normal_file = next((f for f in os.listdir(folder) if f.endswith(f"normal.tiff")), None)

            assert spp1_file and noisy_file, f"Missing files for scene: {key} in {folder}"

            # Load spp1xN images
            spp1_path = os.path.join(folder, spp1_file)
            spp1_img = tifffile.imread(spp1_path)                                                  # (N=low_spp, H, W, 3)

            # NOISY
            noisy_path = os.path.join(folder, noisy_file)
            noisy_img = tifffile.imread(noisy_path)                                                # (H, W, 3)
            self.noisy_images[key] = torch.from_numpy(noisy_img).permute(2, 0, 1).float()          # (3, H, W)

            self.scene_paths[key] = folder

            # CLEAN
            if self.clean and clean_file:
                clean_path = os.path.join(folder, clean_file)
                clean_img = tifffile.imread(clean_path)                                             # (H, W, 3)
                self.clean_images[key] = torch.from_numpy(clean_img).permute(2, 0, 1).float()       # (3, H, W)

            # AOV
            if self.aov:
                # ALBEDO
                albedo_path = os.path.join(folder, albedo_file)
                albedo_img = tifffile.imread(albedo_path)                                           # (H, W, 3)
                self.albedo_images[key] = torch.from_numpy(albedo_img).permute(2, 0, 1).float()     # (3, H, W)
                # NORMAL
                normal_path = os.path.join(folder, normal_file)
                normal_img = tifffile.imread(normal_path)                                           # (H, W, 3)
                self.normal_images[key] = torch.from_numpy(normal_img).permute(2, 0, 1).float()     # (3, H, W)

            # Randomly shuffle indices to pick input and target samples from spp1_img
            # TODO: less random than ImageDataset because we only select indices randomly per image
            indices = list(range(self.low_spp))
            random.shuffle(indices)
            input_idx = indices[:self.target_split]
            target_idx = indices[-self.target_split:]
            self.scene_sample_indices[key] = (input_idx, target_idx)

            # HISTOGRAM GENERATION
            hist_filename = f"{key}_spp{self.low_spp}_bins{self.hist_bins}_hist.npz"
            cache_path = os.path.join(self.cached_dir, hist_filename) if self.cached_dir else None
            if cache_path and os.path.exists(cache_path) or not self.hist_regeneration:
                cached = np.load(cache_path)
                self.hist_features[key] = cached['features']        # (H, W, 3, bins)
            else:
                logger.info(f"Generating Histogram: {hist_filename}")
                input_samples_raw = spp1_img[input_idx]                                         # (N-1, H, W, 3)
                hist, _ = generate_histograms(input_samples_raw, self.hist_bins, self.device)   # (H, W, 3, bins)
                self.hist_features[key] = hist.astype(np.float32)
                if cache_path:
                    np.savez_compressed(cache_path, features=hist)

            spp1_array = np.transpose(spp1_img, (0, 3, 1, 2)).astype(np.float32)  # (N, 3, H, W)
            self.spp1_images[key] = spp1_array

    def __len__(self):
        return self.virt_size

    def __getitem__(self, idx=None, crop_coords=None):
        if self.run_mode == "train":        # TRAINING
            scene = random.choice(self.scene_names)
        else:                               # EVAL
            scene = self.scene_names[idx % len(self.scene_names)]
            
        spp1_img = self.spp1_images[scene]                      # (low_spp, H, W, 3)
        noisy_tensor = self.noisy_images[scene]                 # (3, H, W)
        clean_tensor = self.clean_images.get(scene, None)       # (3, H, W)
        albedo_tensor = self.albedo_images.get(scene, None)     # (3, H, W)
        normal_tensor = self.normal_images.get(scene, None)     # (3, H, W)

        input_idx, target_idx = self.scene_sample_indices[scene]
        input_samples = spp1_img[input_idx]                     # (low_spp-N, 3, H, W)
        target_sample = spp1_img[target_idx]                    # (N, 3, H, W)
        
        # Normalise histogram
        hist_norm = self.hist_features[scene]                   # (H, W, 3, bins)
        hist_sum = np.sum(hist_norm, axis=-1, keepdims=True)    # (H, W, 3, 1)
        hist_norm /= (hist_sum + 1e-8)

        # TODO: also concatenate mean and std to histograms ???
        input_samples_tensor = torch.from_numpy(input_samples).float()  # (N-1, 3, H, W)
        mean = input_samples_tensor.mean(dim=0)                         # (3, H, W)
        std = input_samples_tensor.std(dim=0)                           # (3, H, W)

        # Histogram from numpy → torch, (H, W, 3, bins) → (3, bins, H, W)
        hist_torch = torch.from_numpy(hist_norm).permute(2, 3, 0, 1).float()
        # input_tensor = torch.cat([hist_torch, mean_t, std_t], dim=1)  # (3, bins+2, H, W)
        input_tensor = hist_torch

        if self.aov:
            # Concatenate albedo and normal tensors
            input_tensor = torch.cat([input_tensor, albedo_tensor, normal_tensor], dim=0)  # (3*bins + 6, H, W)
    
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
    


# GENERATIVA ACCUMULATION - HISTOGRAM BINOM DATASET
class HistogramBinomDataset(Dataset):
    def __init__(self, root_dir: str, crop_size: int = 128,
                 data_augmentation: bool = True, virt_size: int = 1000,
                 hist_bins: int = 8, clean: bool = True, low_spp: int = 32, 
                 high_spp: int = 4500, cached_dir: str = None,
                 debug: bool = False, mode: str = None, device: str = None,
                 hist_regeneration: bool = False, scene_names=None, 
                 target_sample: int = 1):
        
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.data_augmentation = data_augmentation
        self.virt_size = virt_size
        self.hist_bins = hist_bins
        self.clean = clean
        self.low_spp = low_spp
        self.high_spp = high_spp
        self.cached_dir = cached_dir
        self.debug = debug
        self.device = device
        logger.info(f"Using device Data Loader: {self.device}")
        self.hist_regeneration = hist_regeneration
        self.target_sample = target_sample

        self.hist_features = {}      # input histograms (from spp1 samples)
        self.target_histograms = {}  # target histograms (from clean image)
        self.clean_images = {}       # clean images for PSNR
        self.bin_edges = {}
        self.scene_paths = {}

        if self.cached_dir and not os.path.exists(self.cached_dir):
            os.makedirs(self.cached_dir)

        # Find all scenes and spp1 files
        scene_keys = []
        for subdir in sorted(os.listdir(self.root_dir)):
            full_subdir = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(full_subdir):
                continue
            for fname in os.listdir(full_subdir):
                if fname.endswith(f"spp1x{self.low_spp}.tiff"):
                    key = fname.split(f"_spp")[0]
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
        logger.info(f"Input Histogram Counts: {self.low_spp - self.target_sample}")
        logger.info(f"Target Histogram Counts: {self.target_sample}")
        logger.info(f"Histogram Bins: {self.hist_bins}")

        for key, folder in scene_keys:
            # Load spp1 samples (32 samples, shape: (32, H, W, 3))
            spp1_file = next((f for f in os.listdir(folder) if f.startswith(key) and f.endswith(f"spp1x{self.low_spp}.tiff")), None)
            assert spp1_file is not None, f"Missing spp1 file for scene {key}"
            spp1_path = os.path.join(folder, spp1_file)
            spp1_samples = tifffile.imread(spp1_path)  # shape: (32, H, W, 3)

            # Load clean image for target
            clean_file = next((f for f in os.listdir(folder) if f.startswith(key) and f.endswith(f"spp{self.high_spp}.tiff")), None)
            assert clean_file is not None, f"Missing clean file for scene {key}"
            clean_path = os.path.join(folder, clean_file)
            clean_img = tifffile.imread(clean_path)  # shape: (H, W, 3)
            self.clean_images[key] = torch.from_numpy(clean_img).permute(2, 0, 1).float()

            self.scene_paths[key] = folder

            # HISTOGRAM CACHING PATHS
            hist_cache = None
            if self.cached_dir:
                hist_cache = os.path.join(self.cached_dir, f"{key}_gen_hist_{self.hist_bins}bins.npz")

            # Split spp1 samples into input and target sets
            assert spp1_samples.shape[0] > self.target_sample, f"target_sample={self.target_sample} must be < total spp1 samples={spp1_samples.shape[0]}"

            # INPUT HISTOGRAM
            if hist_cache and os.path.exists(hist_cache) and not self.hist_regeneration:
                cached = np.load(hist_cache)
                hist = cached['features']
                bin_edges = cached['bin_edges']
            else:
                logger.info(f"Computing histogram for scene {key}")
                hist, bin_edges = generate_histograms(spp1_samples, self.hist_bins, self.device)
                # input_hist, bin_edges = generate_histograms_with_zero_bin(input_samples, self.hist_bins, self.device)
                logger.info(f"Generated histogram of shape {hist.shape}")
                hist = hist.astype(np.float32)
                bin_edges = bin_edges.astype(np.float32)
                if hist_cache:
                    np.savez_compressed(hist_cache, features=hist, bin_edges=bin_edges)

            self.hist_features[key] = torch.from_numpy(hist).float()
            self.bin_edges[key] = torch.from_numpy(bin_edges).float()

            # Proportion of samples to allocate to target
            p = self.target_sample / (self.low_spp + 1e-8)
            self.p = torch.tensor(min(p, 1.0))  # Ensure in [0, 1]


    def __len__(self):
        return self.virt_size

    def __getitem__(self, idx, crop_coords=None):
        scene = self.scene_names[idx % len(self.scene_names)]
        hist = self.hist_features[scene]
        clean_tensor = self.clean_images[scene]
        bin_edges_tensor = self.bin_edges[scene]

        # Create binomial distribution with per-bin counts
        binom = torch.distributions.Binomial(total_count=hist, probs=self.p)

        # Sample the number of target samples per bin + compute input hist based on that
        target_hist = binom.sample()
        input_hist = hist - target_hist

        # NORMALISATION
        target_hist = target_hist / (target_hist.sum(dim=-1, keepdim=True) + 1e-8)          # shape (H, W, 3, B)
        input_hist = input_hist / (input_hist.sum(dim=-1, keepdim=True) + 1e-8)             # shape (H, W, 3, B)

        input_tensor = input_hist.permute(2, 3, 0, 1).contiguous().float()                  # shape (3, B, H, W)
        target_tensor = target_hist.permute(2, 3, 0, 1).contiguous().float()                # shape (3, B, H, W)

        if self.crop_size:
            if crop_coords is None:
                i, j, h, w = transforms.RandomCrop.get_params(target_tensor, output_size=(self.crop_size, self.crop_size))
            else:
                i, j, h, w = crop_coords
            
            input_tensor = input_tensor[:, :, i:i+h, j:j+w]
            target_tensor = target_tensor[:, :, i:i+h, j:j+w]
            clean_tensor = clean_tensor[:, i:i+h, j:j+w]

        if self.data_augmentation:
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-1])
                target_tensor = torch.flip(target_tensor, dims=[-1])
                clean_tensor = torch.flip(clean_tensor, dims=[-1])

            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-2])
                target_tensor = torch.flip(target_tensor, dims=[-2])
                clean_tensor = torch.flip(clean_tensor, dims=[-2])

        return {
            "input_hist": input_tensor,
            "target_hist": target_tensor,
            "clean": clean_tensor,
            "scene": scene,
            "bin_edges": bin_edges_tensor,
            "crop_coords": (i, j, h, w) if self.crop_size else None
        }
