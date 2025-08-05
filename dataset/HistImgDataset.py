import os
import time
import torch
import random
import tifffile
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import os
import torchvision.transforms.functional as TF
import numpy as np
import imageio
# Utils
from utils.utils import apply_tonemap, local_variance, sample_crop_coords_from_variance, load_patches, save_patches, load_or_compute_histograms, compute_covariance_matrix, chi_square_distance, compute_local_histogram_affinity_chi2
from dataset.HistogramGenerator import generate_histograms_torch, generate_hist_statistics

import logging
logger = logging.getLogger(__name__)


# IMAGE DENOISING IMAGE DATASET
class ImageDataset(Dataset):
    def __init__(self, root_dir: str, crop_size: int = 128, mode: str = 'img',
                 data_augmentation: bool = True, crops_per_scene: int = 1000,
                 low_spp: int = 32, high_spp: int = 4500, hist_bins: int = 8,
                 clean: bool = False, aov: bool = False, cached_dir: str = None, 
                 debug: bool = False, device: str = None, scene_names=None, 
                 supervised: bool = False, tonemap: str = None, target_split: int = 1, 
                 run_mode: str = None, use_cached_crops = False):
        self.root_dir = root_dir
        self.mode = mode
        self.crop_size = crop_size
        self.data_augmentation = data_augmentation
        self.low_spp = low_spp
        self.high_spp = high_spp
        self.hist_bins = hist_bins
        self.clean = clean
        self.aov = aov
        self.cached_dir = cached_dir
        self.debug = debug
        self.device = device or torch.device("cpu")
        self.tonemap = tonemap
        self.target_split = target_split
        self.run_mode = run_mode

        self.use_cached_crops = use_cached_crops
        self.crops_per_scene = crops_per_scene

        self.supervised = supervised
        if self.supervised:
            logger.info("Supervised (noise2clean) mode")
        else:
            logger.info("Self-Supervised (noise2noise) mode")

        self.spp1_images = {}           # (scene) -> tensor (N, 3, H, W)
        self.hist_features = {}         # (scene) -> tensor (H, W, 3, bins)
        self.noisy_images = {}          # (scene) -> tensor (3, H, W)
        self.clean_images = {}          # (scene) -> tensor (3, H, W)
        self.albedo_images = {}         # (scene) -> tensor (3, H, W)
        self.normal_images = {}         # (scene) -> tensor (3, H, W)
        self.scene_paths = {}           # (scene) -> folder path
        self.scene_sample_indices = {}  # (scene) -> (list of input idx, target idx)
        self.variance_heatmaps = {}     # (scene) -> torch.Tensor (H_patch, W_patch)
        self.patches = []

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
            spp1_img = np.array(tifffile.imread(spp1_path))                                        # (N=low_spp, H, W, 3)
            self.spp1_images[key] = torch.from_numpy(spp1_img).permute(0, 3, 1, 2).float()         # (N, 3, H, W)

            # NOISY
            noisy_path = os.path.join(folder, noisy_file)
            noisy_img = np.array(tifffile.imread(noisy_path))                                      # (H, W, 3)
            self.noisy_images[key] = torch.from_numpy(noisy_img).permute(2, 0, 1).float()          # (3, H, W)

            self.scene_paths[key] = folder

            # CLEAN
            if self.clean and clean_file:
                clean_path = os.path.join(folder, clean_file)
                clean_img = np.array(tifffile.imread(clean_path))                                   # (H, W, 3)
                self.clean_images[key] = torch.from_numpy(clean_img).permute(2, 0, 1).float()       # (3, H, W)

            # AOV
            if self.aov:
                # ALBEDO
                albedo_path = os.path.join(folder, albedo_file)
                albedo_img = np.array(tifffile.imread(albedo_path))                         # (H, W, 3)
                albedo_tensor = torch.from_numpy(albedo_img).permute(2, 0, 1).float()       # (3, H, W)
                albedo_tensor = apply_tonemap(albedo_tensor, tonemap="log")                 # TONEMAPPING
                self.albedo_images[key] = albedo_tensor
                # NORMAL
                normal_path = os.path.join(folder, normal_file)
                normal_img = np.array(tifffile.imread(normal_path))                         # (H, W, 3)
                normal_tensor = torch.from_numpy(normal_img).permute(2, 0, 1).float()       # (3, H, W)
                normal_tensor = (normal_tensor + 1.0) * 0.5                                 # NORMALISATION
                self.normal_images[key] = normal_tensor

            # VARIANCE SAMPLING MAP
            heatmap_path = os.path.join(folder, f"{key}_variance_heatmap.png")
            # use odd window size to cover whole image (TODO: make it customisable!)
            # the smaller the window size the more the details captured
            varmap = local_variance(self.noisy_images[key], window_size=15, save_path=heatmap_path, cmap='viridis')
            logger.info(f"Sampling Map Shape: {varmap.shape}")
            self.variance_heatmaps[key] = varmap
            
            # TODO: currently 1 min x 100 patches per scene --> could speed it up ?
            patch_cache_path = os.path.join(self.cached_dir, f"{key}_{crops_per_scene}_{self.crop_size}.pkl")

            if self.use_cached_crops and os.path.exists(patch_cache_path):
                logger.info(f"Loading cached patches from {patch_cache_path}")
                scene_patches = load_patches(patch_cache_path)
            else:
                logger.info(f"Generating patches for scene {key}")
                scene_patches = []
                start_time = time.time()

                for _ in range(self.crops_per_scene):
                    i, j, h, w = sample_crop_coords_from_variance(varmap, self.crop_size)
                    spp1_patch = self.spp1_images[key][:, :, i:i+h, j:j+w]         
                    noisy_patch = self.noisy_images[key][:, i:i+h, j:j+w]          
                    clean_patch = self.clean_images[key][:, i:i+h, j:j+w] if self.clean and key in self.clean_images else None
                    albedo_patch = self.albedo_images[key][:, i:i+h, j:j+w] if self.aov and key in self.albedo_images else None
                    normal_patch = self.normal_images[key][:, i:i+h, j:j+w] if self.aov and key in self.normal_images else None

                    scene_patches.append({
                        "scene": key,
                        "spp1": spp1_patch,
                        "noisy": noisy_patch,
                        "clean": clean_patch,
                        "albedo": albedo_patch,
                        "normal": normal_patch,
                        "crop_coords": (i, j, h, w)
                    })

                elapsed = time.time() - start_time
                logger.info(f"Finished generating {self.crops_per_scene} crops for scene '{key}' in {elapsed:.2f} seconds!")

                # if self.cached_dir:
                #     save_patches(scene_patches, patch_cache_path)
                #     logger.info(f"Saved {len(scene_patches)} patches to {patch_cache_path}")

            self.patches.extend(scene_patches)

        logger.info(f"Total patches collected: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)

    # TODO: add the coord option that has been discared
    def __getitem__(self, idx):
        patch = self.patches[idx]

        spp1_patch = patch["spp1"]                  # Tensor (N, 3, crop_size, crop_size)
        noisy_patch = patch["noisy"]                # Tensor (3, crop_size, crop_size)
        clean_patch = patch.get("clean", None)      # Tensor (3, crop_size, crop_size) or None
        albedo_patch = patch.get("albedo", None)    # Tensor (3, crop_size, crop_size) or None
        normal_patch = patch.get("normal", None)    # Tensor (3, crop_size, crop_size) or None

        # Shuffle spp1 patch indices to split input and target sets (Noise2Noise style)
        indices = list(range(spp1_patch.shape[0]))
        random.shuffle(indices)
        half = len(indices) // 2
        input_idx = indices[:half]
        target_idx = indices[half:]

        # INPUT 
        if self.mode == "stat":
            # INPUT FEATURES
            rgb_stats = generate_hist_statistics(spp1_patch[input_idx], return_channels='luminance')    # STACK tensor
            # rgb_stats = generate_hist_statistics(noisy_patch)                                         # NOISY tensor

            mean_img = rgb_stats['mean']                                        # (3, H, W)
            mean_img = apply_tonemap(mean_img, tonemap="log") 
            rel_var = rgb_stats['relative_variance'].permute(2, 0, 1).float()   # (1, H, W)
            rel_var = apply_tonemap(rel_var, tonemap="log") 

            # Compose input by concatenating mean + relative variance along channel dim
            input_tensor = torch.cat([mean_img, rel_var], dim=0)                # (3, H, W) or # (4, H, W)
        else: 
            # NOISY tensor
            # input_tensor = noisy_patch.clone()
            # input_tensor = apply_tonemap(input_tensor, tonemap="log")

            # STACK tensor: the first N input samples from spp1_img
            input_samples = spp1_patch[input_idx]                       # (N, H, W, 3)
            input_tensor = input_samples.mean(dim=0)                    # (3, H, W)
            input_tensor = apply_tonemap(input_tensor, tonemap="log")

        # TARGET
        if self.supervised:
            target_tensor = clean_patch   
        else:
            # NOISY tensor
            # target_tensor = noisy_patch                               # (3, H, W)
            # STACK tensor: the last N input samples from spp1_img
            target_tensor = spp1_patch[target_idx].mean(dim=0)          # (3, H, W)

        # AOV
        if self.aov:
            # Concatenate along channel dimension: input + albedo + normal
            to_concat = [input_tensor]
            if albedo_patch is not None:
                to_concat.append(albedo_patch)
            if normal_patch is not None:
                to_concat.append(normal_patch)
            input_tensor = torch.cat(to_concat, dim=0)  # e.g. ((3 or 4) + 3 + 3 = (9 or 10), H, W)

        # DATA AUGMENTATION: random horizontal and vertical flips (only in train mode)
        if self.data_augmentation and self.run_mode == "train":
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-1])
                target_tensor = torch.flip(target_tensor, dims=[-1])
                noisy_patch = torch.flip(noisy_patch, dims=[-1])
                if clean_patch is not None:
                    clean_patch = torch.flip(clean_patch, dims=[-1])
                if albedo_patch is not None:
                    albedo_patch = torch.flip(albedo_patch, dims=[-1])
                if normal_patch is not None:
                    normal_patch = torch.flip(normal_patch, dims=[-1])
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-2])
                target_tensor = torch.flip(target_tensor, dims=[-2])
                noisy_patch = torch.flip(noisy_patch, dims=[-2])
                if clean_patch is not None:
                    clean_patch = torch.flip(clean_patch, dims=[-2])
                if albedo_patch is not None:
                    albedo_patch = torch.flip(albedo_patch, dims=[-2])
                if normal_patch is not None:
                    normal_patch = torch.flip(normal_patch, dims=[-2])

        return {
            "input": input_tensor,                                          # (3 or 9, H, W)
            "target": target_tensor,                                        # (3, H, W)
            "noisy": noisy_patch,                                           # (3, H, W)
            "clean": clean_patch if clean_patch is not None else None,      # (3, H, W) or None
            "scene": patch.get("scene", None),
            "crop_coords": patch.get("crop_coords", None),
        }


# IMAGE DENOISING - CROP HISTOGRAM DATASET
# TODO: maybe here instead of an histogram of discrete counts give an actual histogram of the radiance (also faster to generate on GPU)
class HistogramDataset(Dataset):
    def __init__(self, root_dir: str, crop_size: int = 128, mode: str = 'hist',
                 data_augmentation: bool = True, crops_per_scene: int = 1000,
                 low_spp: int = 32, high_spp: int = 4500, hist_bins: int = 8,
                 clean: bool = False, aov: bool = False, cached_dir: str = None, debug: bool = False,
                 device: str = None, scene_names=None, supervised: bool = False, tonemap: str = None, 
                 target_split: int = 1, run_mode: str = None, use_cached_crops: bool = False, log_bins: bool = False):
        
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.data_augmentation = data_augmentation
        self.crops_per_scene = crops_per_scene
        self.low_spp = low_spp
        self.high_spp = high_spp
        self.hist_bins = hist_bins
        self.clean = clean
        self.aov = aov
        self.cached_dir = cached_dir
        self.debug = debug
        self.device = device or torch.device("cpu")
        self.tonemap = tonemap
        self.target_split = target_split
        self.run_mode = run_mode
        self.use_cached_crops = use_cached_crops
        self.supervised = supervised
        self.log_bins = log_bins
        if self.supervised:
            logger.info("Supervised (noise2clean) mode")
        else:
            logger.info("Self-Supervised (noise2noise) mode")

        self.spp1_images = {}           # (scene) -> tensor (N, 3, H, W)
        self.hist_features = {}         # (scene) -> tensor (H, W, 3, bins)
        self.noisy_images = {}          # (scene) -> tensor (3, H, W)
        self.clean_images = {}          # (scene) -> tensor (3, H, W)
        self.albedo_images = {}         # (scene) -> tensor (3, H, W)
        self.normal_images = {}         # (scene) -> tensor (3, H, W)
        self.scene_paths = {}           # (scene) -> folder path
        self.scene_sample_indices = {}  # (scene) -> (list of input idx, target idx)
        self.bin_edges = {}             # (scene) -> tensor (bins+1,)
        self.patches = []

        # Scan scenes and collect file paths only
        for subdir in sorted(os.listdir(self.root_dir)):
            full_subdir = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(full_subdir):
                continue
            for fname in os.listdir(full_subdir):
                if fname.endswith(f"spp1x{self.low_spp}.tiff"):
                    key = fname.split("_spp")[0]
                    # PATHS
                    spp1_file = next((f for f in os.listdir(full_subdir) if f.startswith(key) and f.endswith(f"spp1x{self.low_spp}.tiff")), None)
                    noisy_file = next((f for f in os.listdir(full_subdir) if f.startswith(key) and f.endswith(f"spp{self.low_spp}.tiff")), None)
                    clean_file = next((f for f in os.listdir(full_subdir) if f.startswith(key) and f.endswith(f"spp{self.high_spp}.tiff")), None)
                    albedo_file = next((f for f in os.listdir(full_subdir) if f.endswith("albedo.tiff")), None)
                    normal_file = next((f for f in os.listdir(full_subdir) if f.endswith("normal.tiff")), None)
                    self.scene_paths[key] = {
                        "folder": full_subdir,
                        "spp1": os.path.join(full_subdir, spp1_file),
                        "noisy": os.path.join(full_subdir, noisy_file),
                        "clean": os.path.join(full_subdir, clean_file) if clean_file else None,
                        "albedo": os.path.join(full_subdir, albedo_file) if albedo_file else None,
                        "normal": os.path.join(full_subdir, normal_file) if normal_file else None,
                    }

                    # Load spp1 image
                    spp1_img = tifffile.imread(self.scene_paths[key]["spp1"])
                    spp1_tensor = torch.from_numpy(spp1_img).permute(0, 3, 1, 2).float()  # (N,3,H,W)
                    self.spp1_images[key] = spp1_tensor
                
                    # For each scene, sample patch coords based on variance or some heuristic
                    # You can load only noisy image for variance calculation here or skip and cache patch coords externally.
                    noisy_img = tifffile.imread(self.scene_paths[key]["noisy"])
                    noisy_tensor = torch.from_numpy(noisy_img).permute(2, 0, 1).float()
                    self.noisy_images[key] = noisy_tensor
                    varmap = local_variance(noisy_tensor, window_size=15)

                    for _ in range(self.crops_per_scene):
                        i, j, h, w = sample_crop_coords_from_variance(varmap, self.crop_size)
                        self.patches.append({"scene": key, "coords": (i, j, h, w)})
                    logger.info(f"Generated patches of scene {key}!")

        logger.info(f"Total patches: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        scene = patch_info["scene"]
        i, j, h, w = patch_info["coords"]

        paths = self.scene_paths[scene]

        # NOISY
        spp1_img = self.spp1_images[scene]                          # (N, H, W, 3)
        noisy_tensor = self.noisy_images[scene][:, i:i+h, j:j+w]    # (H, W, 3)

        # CLEAN
        clean_tensor = None
        if self.clean and paths["clean"] is not None:
            clean_img = tifffile.imread(paths["clean"])
            clean_tensor = torch.from_numpy(clean_img).permute(2, 0, 1).float()[:, i:i+h, j:j+w]

        # AOV
        albedo_tensor, normal_tensor = None, None
        if self.aov:
            if paths["albedo"]:
                albedo_img = tifffile.imread(paths["albedo"])
                albedo_tensor = torch.from_numpy(albedo_img).permute(2, 0, 1).float()[:, i:i+h, j:j+w]
                albedo_tensor = apply_tonemap(albedo_tensor, tonemap="log")
            if paths["normal"]:
                normal_img = tifffile.imread(paths["normal"])
                normal_tensor = torch.from_numpy(normal_img).permute(2, 0, 1).float()[:, i:i+h, j:j+w]
                normal_tensor = (normal_tensor + 1.0) * 0.5

        # HISTOGRAMS
        hist, bin_edges = load_or_compute_histograms(
            key=scene,
            spp1_tensor=spp1_img,
            hist_bins=self.hist_bins,
            device=self.device,
            cached_dir=self.cached_dir,
            log_binning=self.log_bins
        )
        hist_tensor = hist[:, i:i+h, j:j+w, :]  # (3, H, W, bins)

        # Input/target split (random shuffle)
        indices = list(range(self.low_spp))
        random.shuffle(indices)
        input_idx = indices[:self.target_split]
        target_idx = indices[self.target_split:]
        input_samples_tensor = spp1_img[input_idx][:, :, i:i+h, j:j+w]      # (N, 3, H, W)
        target_samples = spp1_img[target_idx].mean(dim=0)[:, i:i+h, j:j+w]  # (3, H, W)

        # Histogram processing
        hist_torch = hist_tensor.permute(0, 3, 1, 2).contiguous()  # (3, bins, H, W)
        input_tensor = hist_torch.reshape(-1, h, w)

        # Mean and relative variance
        rgb_stats = generate_hist_statistics(input_samples_tensor, return_channels='luminance')
        mean_img = apply_tonemap(rgb_stats["mean"], tonemap="log")
        rel_var = apply_tonemap(rgb_stats["relative_variance"], tonemap="log")

        input_tensor = torch.cat([input_tensor, mean_img, rel_var], dim=0)

        if self.aov:
            if albedo_tensor is not None:
                input_tensor = torch.cat([input_tensor, albedo_tensor], dim=0)
            if normal_tensor is not None:
                input_tensor = torch.cat([input_tensor, normal_tensor], dim=0)

        # Data augmentation as before
        if self.data_augmentation and self.run_mode == "train":
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-1])
                target_samples = torch.flip(target_samples, dims=[-1])
                noisy_tensor = torch.flip(noisy_tensor, dims=[-1])
                if clean_tensor is not None:
                    clean_tensor = torch.flip(clean_tensor, dims=[-1])
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-2])
                target_samples = torch.flip(target_samples, dims=[-2])
                noisy_tensor = torch.flip(noisy_tensor, dims=[-2])
                if clean_tensor is not None:
                    clean_tensor = torch.flip(clean_tensor, dims=[-2])

        target_tensor = clean_tensor if self.supervised else target_samples

        return {
            "input": input_tensor,
            "target": target_tensor,
            "noisy": noisy_tensor,
            "clean": clean_tensor,
            "scene": scene,
            "bin_edges": bin_edges,
            "crop_coords": (i, j, h, w),
        }


# GENERATIVE ACCUMULATION - HISTOGRAM BINOM DATASET
class HistogramBinomDataset(Dataset):
    def __init__(self, root_dir: str, crop_size: int = 128,
                 data_augmentation: bool = True, virt_size: int = 1000,
                 hist_bins: int = 8, clean: bool = True, low_spp: int = 32, 
                 high_spp: int = 4500, cached_dir: str = None,
                 debug: bool = False, mode: str = None, device: str = None, scene_names=None, 
                 target_sample: int = 1, stat: bool = True, log_bins = True):
        
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.data_augmentation = data_augmentation
        self.virt_size = virt_size
        self.hist_bins = hist_bins
        self.clean = clean
        self.stat = stat
        self.low_spp = low_spp
        self.high_spp = high_spp
        self.cached_dir = cached_dir
        self.debug = debug
        self.device = device
        logger.info(f"Using device Data Loader: {self.device}")
        self.target_sample = target_sample
        self.log_bins = log_bins

        self.clean_images = {}          # clean images for PSNR
        self.scene_paths = {}
        self.cached_data = {}

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
            spp1_samples = tifffile.imread(spp1_path)                                                  # (low_spp, H, W, 3)
            spp1_samples = torch.from_numpy(spp1_samples).permute(0, 3, 1, 2).float()         # (N, 3, H, W)
            
            # hist shape: (3, H, W, B)
            # edges shape: (3, H, W, B)
            hist, bin_edges = load_or_compute_histograms(
                key=key,
                spp1_tensor=spp1_samples,
                hist_bins=self.hist_bins,
                device=self.device,
                cached_dir=self.cached_dir,
                log_binning=self.log_bins,
                normalize=False
            ) 

            # generate statistics and tone map them otherwise they'll be in HDR range !!!
            stats = generate_hist_statistics(spp1_samples, return_channels='all')
            pixel_mean = stats['mean']                                                      # (3, H, W)
            rel_var = stats['relative_variance']                                            # (3, H, W)
            cov_matrix = compute_covariance_matrix(spp1_samples)                            # (6, H, W)

            # logger.info(f"Mean Min {pixel_mean.min()} - Max {pixel_mean.max()} - Mean {pixel_mean.mean()} - Var {pixel_mean.var()}")
            # logger.info(f"Var Min {rel_var.min()} - Max {rel_var.max()} - Mean {rel_var.mean()} - Var {rel_var.var()}")
            # logger.info(f"Cov Min {cov_matrix.min()} - Max {cov_matrix.max()} - Mean {cov_matrix.mean()} - Var {cov_matrix.var()}")

            pixel_mean = apply_tonemap(pixel_mean, tonemap="log") 
            rel_var = apply_tonemap(rel_var, tonemap="log") 
            # cov_matrix = apply_tonemap(cov_matrix, tonemap="log")         do not apply log on cov matrix ! might be negative !

            # logger.info(f"Mean Min {pixel_mean.min()} - Max {pixel_mean.max()} - Mean {pixel_mean.mean()} - Var {pixel_mean.var()}")
            # logger.info(f"Var Min {rel_var.min()} - Max {rel_var.max()} - Mean {rel_var.mean()} - Var {rel_var.var()}")
            # logger.info(f"Cov Min {cov_matrix.min()} - Max {cov_matrix.max()} - Mean {cov_matrix.mean()} - Var {cov_matrix.var()}")

            self.cached_data[key] = {
                "hist": hist, "bin_edges": bin_edges, "mean": pixel_mean, "rel_var": rel_var, "cov_mat": cov_matrix
            }

            # Load clean image for target
            clean_file = next((f for f in os.listdir(folder) if f.startswith(key) and f.endswith(f"spp{self.high_spp}.tiff")), None)
            assert clean_file is not None, f"Missing clean file for scene {key}"
            clean_path = os.path.join(folder, clean_file)

            self.scene_paths[key] = {
                "folder": folder,
                "spp1_path": spp1_path,
                "clean_path": clean_path
            }

            # Proportion of samples to allocate to target
            p = self.target_sample / (self.low_spp + 1e-8)
            self.p = torch.tensor(min(p, 1.0))  # Ensure in [0, 1]


    def __len__(self):
        return self.virt_size

    def __getitem__(self, idx, crop_coords=None):
        scene = self.scene_names[idx % len(self.scene_names)] 
        scene_cache = self.cached_data[scene]
        hist = scene_cache['hist']
        bin_edges = scene_cache['bin_edges']

        # CHI2 MATRIX
        hist_norm = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)
        affinity_map = compute_local_histogram_affinity_chi2(hist_norm, scene, cache_dir="maps")  # (1, H, W)

        # CROP
        if self.crop_size:
            _, H, W, _ = hist.shape
            if crop_coords is None:
                i, j, h, w = transforms.RandomCrop.get_params(torch.empty((H, W)), output_size=(self.crop_size, self.crop_size))
            else:
                i, j, h, w = crop_coords
            hist = hist[:, i:i+h, j:j+w, :]                   # (3, H, W, B)
            affinity_crop = affinity_map[:, i:i+h, j:j+w]                   # (3, H, W, B)

        # CLEAN
        clean_tensor = None
        if self.clean:
            clean_path = self.scene_paths[scene]["clean_path"]
            clean_img = tifffile.imread(clean_path)
            clean_tensor = torch.from_numpy(clean_img).permute(2, 0, 1).float()
            if self.crop_size:
                clean_tensor = clean_tensor[:, i:i+h, j:j+w]

        # Randomize target_sample each time (currently discarded because it adds more noise)
        # target_sample = random.choice([8, 12, , 20, 24])
        # p = torch.tensor(min(self.target_sample / (self.low_spp + 1e-8), 1.0))              # Compute binomial sampling probability
        binom = torch.distributions.Binomial(total_count=hist, probs=self.p)                     # Generate binomial input and target histograms
        target_hist = binom.sample()
        input_hist = hist - target_hist

        # NORMALISATION
        target_hist = target_hist / (target_hist.sum(dim=-1, keepdim=True) + 1e-8)          # shape (3, H, W, B)
        input_hist = input_hist / (input_hist.sum(dim=-1, keepdim=True) + 1e-8)             # shape (3, H, W, B)

        input_tensor = input_hist.permute(0, 3, 1, 2).contiguous().float()                  # shape (3, B, H, W)
        target_tensor = target_hist.permute(0, 3, 1, 2).contiguous().float()                # shape (3, B, H, W)

        # Optionally save affinity map as image (rescale to 0-255)
        # if self.debug:
        #     os.makedirs("maps", exist_ok=True)
        #     affinity_np = (affinity_map.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        #     filename = f"{scene}_affinity.png"
        #     imageio.imwrite(os.path.join("maps", filename), affinity_np)

        # CHI2 MATRIX
        affinity_expanded = affinity_crop.repeat(3, 1, 1).unsqueeze(1)  # (3, 1, H, W)
        input_tensor = torch.cat([input_tensor, affinity_expanded], dim=1)

        # MEAN and RELATIVE VARIANCE from samples
        if self.stat:
            pixel_mean_expanded = scene_cache['mean'][:, i:i+h, j:j+w].unsqueeze(1)             # (3, 1, H, W)
            pixel_var_expanded = scene_cache['rel_var'][:, i:i+h, j:j+w].unsqueeze(1)           # (3, 1, H, W)
            # cov_tensor = scene_cache['cov_mat'][:, i:i+h, j:j+w]                                # (6, H, W)
            # cov_tensor = cov_tensor.unsqueeze(0).repeat(3, 1, 1, 1)                             # (3, 6, H, W)

            input_tensor = torch.cat([input_tensor, pixel_mean_expanded, pixel_var_expanded], dim=1)    # (3, B+1+6, H, W)

        # DATA AUGMENTATION
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
            "bin_edges": bin_edges,
            "crop_coords": (i, j, h, w) if self.crop_size else None
        }


# GENERATIVE ACCUMULATION - CROP HISTOGRAM BINOM DATASET
class CropHistogramBinomDataset(Dataset):
    """"
        Gives better results as the range is more restricted so more balanced histogram for 128x128 crops and  bins
        BUT more difficul for model to learn as range chanegs for very crop
    """
    def __init__(self, root_dir: str, crop_size: int = 128,
                 data_augmentation: bool = True, virt_size: int = 1000,
                 hist_bins: int = 8, clean: bool = True, low_spp: int = 32, 
                 high_spp: int = 4500, cached_dir: str = None,
                 debug: bool = False, mode: str = None, device: str = None, scene_names=None, 
                 target_sample: int = 1, stat: bool = True):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.data_augmentation = data_augmentation
        self.virt_size = virt_size
        self.hist_bins = hist_bins
        self.clean = clean
        self.stat = stat
        self.low_spp = low_spp
        self.high_spp = high_spp
        self.cached_dir = cached_dir
        self.debug = debug
        self.device = device
        logger.info(f"Using device Data Loader: {self.device}")
        self.target_sample = target_sample

        self.hist_features = {}      # input histograms (from spp1 samples)
        self.clean_images = {}       # clean images for PSNR
        self.bin_edges = {}
        self.scene_paths = {}
        self.spp1_samples = {}
        self.global_bin_counts = torch.zeros(self.hist_bins)

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
            self.spp1_samples[key] = torch.from_numpy(spp1_samples)

            # Load clean image for target
            clean_file = next((f for f in os.listdir(folder) if f.startswith(key) and f.endswith(f"spp{self.high_spp}.tiff")), None)
            assert clean_file is not None, f"Missing clean file for scene {key}"
            clean_path = os.path.join(folder, clean_file)
            clean_img = tifffile.imread(clean_path)  # shape: (H, W, 3)
            self.clean_images[key] = torch.from_numpy(clean_img).permute(2, 0, 1).float()

            self.scene_paths[key] = folder

            # Split spp1 samples into input and target sets
            assert spp1_samples.shape[0] > self.target_sample, f"target_sample={self.target_sample} must be < total spp1 samples={spp1_samples.shape[0]}"

    def __len__(self):
        return self.virt_size

    def __getitem__(self, idx, crop_coords=None):
        scene = self.scene_names[idx % len(self.scene_names)]
        clean_tensor = self.clean_images[scene]                 # (3, H, W)
        spp1_samples = self.spp1_samples[scene]                 # (low_spp, H, W, 3)

        # CROP
        if self.crop_size:
            _, H, W, _ = spp1_samples.shape
            if crop_coords is None:
                i, j, h, w = transforms.RandomCrop.get_params(torch.empty((H, W)), output_size=(self.crop_size, self.crop_size))
            else:
                i, j, h, w = crop_coords
            spp1_samples = spp1_samples[:, i:i+h, j:j+w, :]    # (low_spp, crop_H, crop_W, 3)
            clean_tensor = clean_tensor[:, i:i+h, j:j+w]

        # ACCUMULATE HISTOGRAM HERE PER CROP !!!! 
        hist, bin_edges_tensor = generate_histograms_torch(spp1_samples, self.hist_bins, self.device, log_binning=True)

        # BINOMIAL SPLIT
        # target_sample = random.choice([4, 8, 12, , 20])                     
        # discarded because even if it augments it confuses the model (uncomment if confidence is added)
        p = torch.tensor(min(self.target_sample / (self.low_spp + 1e-8), 1.0))       # Binom Probability 
        binom = torch.distributions.Binomial(total_count=hist, probs=p)         # Binom Sampling
        target_hist = binom.sample()
        input_hist = hist - target_hist

        # CONFIDENCE: number of samples = "how much to trust the network" (DISCARDED because not using random target samples)
        confidence = input_hist.sum(dim=-1)  # shape: (H, W, 3)

        # NORMALIZATION
        target_hist = target_hist / (target_hist.sum(dim=-1, keepdim=True) + 1e-8)
        input_hist = input_hist / (input_hist.sum(dim=-1, keepdim=True) + 1e-8)
        input_tensor = input_hist.permute(2, 3, 0, 1).contiguous().float()      # (3, B, H, W)
        target_tensor = target_hist.permute(2, 3, 0, 1).contiguous().float()    # (3, B, H, W)

        # Add confidence as an extra channel per RGB component
        # confidence = confidence.permute(2, 0, 1).unsqueeze(1)                    # (3, 1, H, W)
        # input_tensor = torch.cat([input_tensor, confidence], dim=1)              # (3, B+1, H, W) --> across bins may ruin model predictions because it sees confidence as extra counts

        # MEAN
        # Compute mean image from input samples (excluding target_sample)
        if self.stat:
            input_sample_count = self.low_spp - self.target_sample
            input_samples = spp1_samples[:input_sample_count]  # (N, H, W, 3)
            mean_img = input_samples.mean(dim=0).permute(2, 0, 1).float()  # (3, H, W)
            mean_img = mean_img.unsqueeze(1)  # (3, 1, H, W)
            input_tensor = torch.cat([input_tensor, mean_img], dim=1)  # (3, B+1, H, W)

        # DATA AUMENTATION
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


class AdaptiveSamplingDataset(Dataset):
    """
    Dataset that loads input features (histograms or stats) and the importance map target.
    """
    def __init__(self, root_dir: str, crop_size: int = 128, virt_size: int = 1000,
                 hist_bins: int = 8, mode: str = 'hist', clean: bool = True,
                 low_spp: int = 32, high_spp: int = 4500, cached_dir: str = None,
                 debug: bool = False, device: str = 'cpu', target_sample: int = 1, 
                 scene_names=None, log_bins: bool = True):

        self.root_dir = root_dir
        self.crop_size = crop_size
        self.virt_size = virt_size
        self.hist_bins = hist_bins
        self.mode = mode
        self.clean = clean
        self.low_spp = low_spp
        self.high_spp = high_spp
        self.cached_dir = cached_dir
        self.debug = debug
        self.device = device
        self.target_sample = target_sample
        self.log_bins = log_bins

        self.scene_names = scene_names or []
        self.cached_data = {}

        self._load_dataset()

    def _load_dataset(self):
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
        if not self.scene_names:
            self.scene_names = all_scenes

        for key, folder in scene_keys:
            if key not in self.scene_names:
                continue

            spp1_file = next((f for f in os.listdir(folder) if f.startswith(key) and f.endswith(f"spp1x{self.low_spp}.tiff")), None)
            spp1_path = os.path.join(folder, spp1_file)
            spp1 = tifffile.imread(spp1_path)  # (low_spp, H, W, 3)
            spp1_tensor = torch.from_numpy(spp1).permute(0, 3, 1, 2).float()  # (N, 3, H, W)

            spp1_file = next((f for f in os.listdir(folder) if f.startswith(key) and f.endswith(f"spp1x{self.low_spp}.tiff")), None)
            spp1_path = os.path.join(folder, spp1_file)
            spp1 = tifffile.imread(spp1_path)  # (low_spp, H, W, 3)
            spp1_tensor = torch.from_numpy(spp1).permute(0, 3, 1, 2).float()  # (N, 3, H, W)

            hist, bin_edges = load_or_compute_histograms(
                key=key,
                spp1_tensor=spp1_tensor,
                hist_bins=self.hist_bins,
                device=self.device,
                cached_dir=self.cached_dir,
                log_binning=self.log_bins,
                normalize=False
            )

            hist_norm = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)
            chi2_map = compute_local_histogram_affinity_chi2(hist_norm, key)

            cache = {"hist": hist, "bin_edges": bin_edges, "chi2": chi2_map}

            if self.mode == 'stat':
                stats = generate_hist_statistics(spp1_tensor, return_channels='all')
                mean = apply_tonemap(stats['mean'], tonemap='log')
                var = apply_tonemap(stats['relative_variance'], tonemap='log')
                cache.update({'mean': mean, 'var': var})

            self.cached_data[key] = cache

    def __len__(self):
        return self.virt_size

    def __getitem__(self, idx):
        scene = self.scene_names[idx % len(self.scene_names)]
        cache = self.cached_data[scene]

        hist = cache['hist']  # (3, H, W, B)
        chi2 = cache['chi2']  # (1, H, W)

        _, H, W, B = hist.shape
        i, j = random.randint(0, H - self.crop_size), random.randint(0, W - self.crop_size)

        # Crop everything
        hist_crop = hist[:, i:i+self.crop_size, j:j+self.crop_size, :]  # (3, crop, crop, B)
        chi2_crop = chi2[:, i:i+self.crop_size, j:j+self.crop_size]     # (1, crop, crop)

        if self.mode == 'hist':
            hist_norm = hist_crop / (hist_crop.sum(dim=-1, keepdim=True) + 1e-8)
            hist_tensor = hist_norm.permute(0, 3, 1, 2)  # (3, B, H, W)
            chi2_tensor = chi2_crop.repeat(3, 1, 1).unsqueeze(1)  # (3, 1, H, W)
            x = torch.cat([hist_tensor, chi2_tensor], dim=1)     # (3, B+1, H, W)

        elif self.mode == 'stat':
            mean = cache['mean'][:, i:i+self.crop_size, j:j+self.crop_size].unsqueeze(1)  # (3, 1, H, W)
            var = cache['var'][:, i:i+self.crop_size, j:j+self.crop_size].unsqueeze(1)    # (3, 1, H, W)
            chi2_tensor = chi2_crop.repeat(3, 1, 1).unsqueeze(1)  # (3, 1, H, W)
            x = torch.cat([mean, var, chi2_tensor], dim=1)       # (3, 3, H, W)

        x = x.view(-1, self.crop_size, self.crop_size)  # flatten channel group -> (C, H, W)

        # Fake target for now: uniform importance (can replace with e.g. |error| or |variance|)
        y = torch.ones((self.crop_size, self.crop_size))  # (H, W)

        return {
            'input': x.float(),
            'target': y.float(),
            'scene': scene,
        }