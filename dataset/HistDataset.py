import os
import torch
import random
import tifffile
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from dataset.HistogramGenerator import generate_histograms
from dataset.HistogramGeneratorZero import generate_histograms_with_zero_bin


import logging
logger = logging.getLogger(__name__)

class HistogramDataset(Dataset):
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
