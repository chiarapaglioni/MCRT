import os
import time
import torch
import random
import tifffile
import numpy as np
import matplotlib.pyplot as plt
# Torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import RandomCrop
# Utils
from utils.utils import apply_tonemap, local_variance, sample_crop_coords_from_variance, load_patches, load_or_compute_histograms, compute_covariance_matrix, compute_fixed_log_edges_from_scenes
from dataset.HistogramGenerator import generate_hist_statistics

import logging
logger = logging.getLogger(__name__)


# IMAGE DENOISING IMAGE DATASET
class ImageDataset(Dataset):
    def __init__(self, root_dir: str, crop_size: int = 128, mode: str = 'img',
                 data_augmentation: bool = True, crops_per_scene: int = 1000,
                 low_spp: int = 32, high_spp: int = 4500, hist_bins: int = 8,
                 clean: bool = False, aov: bool = False, cached_dir: str = None, 
                 debug: bool = False, device: str = None, supervised: bool = False, 
                 tonemap: str = None, target_split: int = 1, run_mode: str = None, 
                 use_cached_crops: bool = False, input_tonemap: str = 'log',
                 stat: bool = True):
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
        self.input_tonemap = input_tonemap
        self.stat = stat

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
        self.scene_paths = {}           # (scene) -> folder path
        self.scene_sample_indices = {}  # (scene) -> (list of input idx, target idx)
        self.variance_heatmaps = {}     # (scene) -> torch.Tensor (H_patch, W_patch)
        self.scene_paths = {}
        self.patches = []

        if self.cached_dir and not os.path.exists(self.cached_dir):
            os.makedirs(self.cached_dir)

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

                    assert spp1_file and noisy_file, f"Missing files for scene: {key} in {full_subdir}"

                    # Load spp1xN images
                    spp1_path = os.path.join(full_subdir, spp1_file)
                    spp1_img = np.array(tifffile.imread(spp1_path))                                        # (N=low_spp, H, W, 3)
                    self.spp1_images[key] = torch.from_numpy(spp1_img).permute(0, 3, 1, 2).float()         # (N, 3, H, W)

                    # NOISY
                    noisy_path = os.path.join(full_subdir, noisy_file)
                    noisy_img = np.array(tifffile.imread(noisy_path))                                      # (H, W, 3)
                    self.noisy_images[key] = torch.from_numpy(noisy_img).permute(2, 0, 1).float()          # (3, H, W)

                    # VARIANCE SAMPLING MAP
                    heatmap_path = os.path.join(full_subdir, f"{key}_variance_heatmap.png")
                    # use odd window size to cover whole image (TODO: make it customisable!)
                    # the smaller the window size the more the details captured
                    varmap = local_variance(self.noisy_images[key], window_size=15, save_path=heatmap_path, cmap='viridis')
                    logger.info(f"Sampling Map Shape: {varmap.shape}")
                    self.variance_heatmaps[key] = varmap
                    
                    patch_cache_path = os.path.join(self.cached_dir, f"{key}_{crops_per_scene}_{self.crop_size}.pkl")

                    if self.use_cached_crops and os.path.exists(patch_cache_path):
                        logger.info(f"Loading cached patches from {patch_cache_path}")
                        scene_patches = load_patches(patch_cache_path)
                    else:
                        logger.info(f"Generating patches for scene {key}")
                        scene_patches = []
                        start_time = time.time()

                        # random crops because if selected based on variance it overfits!
                        for crop in range(self.crops_per_scene):
                            # i, j, h, w = sample_crop_coords_from_variance(varmap, self.crop_size)
                            i, j, h, w = RandomCrop.get_params(varmap, output_size=(self.crop_size, self.crop_size))
                            patch_var = varmap[i:i+h, j:j+w]
                            patch_mean = patch_var.mean().item()
                            scene_patches.append({
                                "scene": key,
                                "crop_coords": (i, j, h, w),
                                "variance": patch_mean
                            })

                            if crop%10 == 0: 
                                fname = f"{key}_patch_{i}_{j}_varmap.png"
                                save_path = os.path.join("maps", fname)
                                
                                plt.imsave(save_path, patch_var, cmap="viridis")
                                logger.info(f"Saved patch variance map to {save_path}")
                        elapsed = time.time() - start_time
                        logger.info(f"Finished generating {self.crops_per_scene} crops for scene '{key}' in {elapsed:.2f} seconds!")

                    self.patches.extend(scene_patches)

        logger.info(f"Total patches collected: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)
    
    def get_sampling_weights(self, normalize=True, min_clip=1e-6):
        variances = np.array([p["variance"] for p in self.patches])
        variances = np.clip(variances, min_clip, None)
        if normalize:
            variances = variances / variances.sum()
        return variances

    def __getitem__(self, idx, coords=None):
        patch = self.patches[idx]
        scene = patch["scene"]
        paths = self.scene_paths[scene]
        if coords==None:
            i, j, h, w = patch["crop_coords"]
        else: 
            i, j, h, w = coords

        spp1_patch = self.spp1_images[scene][:, :, i:i+h, j:j+w]         # (N, 3, h, w)
        noisy_patch = self.noisy_images[scene][:, i:i+h, j:j+w]          # (3, h, w)

        # Shuffle spp1 patch indices to split input and target sets (Noise2Noise style)
        indices = list(range(spp1_patch.shape[0]))
        random.shuffle(indices)
        half = len(indices) // 2
        input_idx = indices[:half]
        target_idx = indices[half:]

        # INPUT 
        if self.stat:
            # INPUT FEATURES
            rgb_stats = generate_hist_statistics(spp1_patch[input_idx], return_channels='hdr')      # STACK tensor
            mean_img = rgb_stats['mean']                                                            # (3, H, W)
            mean_img = apply_tonemap(mean_img, tonemap=self.input_tonemap) 
            rel_var = rgb_stats['relative_variance'] 
            # rel_var = rgb_stats['var']                                                            # (3, H, W)
            rel_var = apply_tonemap(rel_var, tonemap=self.input_tonemap) 
            input_tensor = torch.cat([mean_img, rel_var], dim=0)                                    # (3, H, W) or # (6, H, W)
        else: 
            # STACK tensor: the first N input samples from spp1_img
            input_samples = spp1_patch[input_idx]                                                   # (N, H, W, 3)
            input_tensor = input_samples.mean(dim=0)                                                # (3, H, W)
            input_tensor = apply_tonemap(input_tensor, tonemap=self.input_tonemap)

        # CLEAN
        clean_tensor = None
        if self.clean and paths["clean"] is not None:
            clean_img = tifffile.imread(paths["clean"])
            clean_tensor = torch.from_numpy(clean_img).permute(2, 0, 1).float()[:, i:i+h, j:j+w]

        # TARGET
        if self.supervised:
            target_tensor = clean_tensor   
        else:
            target_tensor = spp1_patch[target_idx].mean(dim=0)          # (3, H, W)

        # AOV
        albedo_tensor, normal_tensor = None, None
        if self.aov:
            if paths["albedo"]:
                albedo_img = tifffile.imread(paths["albedo"])
                albedo_tensor = torch.from_numpy(albedo_img).permute(2, 0, 1).float()[:, i:i+h, j:j+w]
                albedo_tensor = apply_tonemap(albedo_tensor, tonemap=self.input_tonemap)
                input_tensor = torch.cat([input_tensor, albedo_tensor], dim=0)
            if paths["normal"]:
                normal_img = tifffile.imread(paths["normal"])
                normal_tensor = torch.from_numpy(normal_img).permute(2, 0, 1).float()[:, i:i+h, j:j+w]
                normal_tensor = (normal_tensor + 1.0) * 0.5
                input_tensor = torch.cat([input_tensor, normal_tensor], dim=0)

        # DATA AUGMENTATION: random horizontal and vertical flips (only in train mode)
        if self.data_augmentation and self.run_mode == "train":
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-1])
                target_tensor = torch.flip(target_tensor, dims=[-1])
                noisy_patch = torch.flip(noisy_patch, dims=[-1])
                if clean_tensor is not None:
                    clean_tensor = torch.flip(clean_tensor, dims=[-1])
                if albedo_tensor is not None:
                    albedo_tensor = torch.flip(albedo_tensor, dims=[-1])
                if normal_tensor is not None:
                    normal_tensor = torch.flip(normal_tensor, dims=[-1])
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-2])
                target_tensor = torch.flip(target_tensor, dims=[-2])
                noisy_patch = torch.flip(noisy_patch, dims=[-2])
                if clean_tensor is not None:
                    clean_tensor = torch.flip(clean_tensor, dims=[-2])
                if albedo_tensor is not None:
                    albedo_tensor = torch.flip(albedo_tensor, dims=[-2])
                if normal_tensor is not None:
                    normal_tensor = torch.flip(normal_tensor, dims=[-2])

        return {
            "input": input_tensor,                                          # (3 or 9, H, W)
            "target": target_tensor,                                        # (3, H, W)
            "noisy": noisy_patch,                                           # (3, H, W)
            "clean": clean_tensor,
            "scene": patch.get("scene", None),
            "crop_coords": patch.get("crop_coords", None),
        }


# IMAGE DENOISING - CROP HISTOGRAM DATASET
class HistogramDataset(Dataset):
    def __init__(self, root_dir: str, crop_size: int = 128, mode: str = 'hist',
                 data_augmentation: bool = True, crops_per_scene: int = 1000,
                 low_spp: int = 32, high_spp: int = 4500, hist_bins: int = 8,
                 clean: bool = False, aov: bool = False, cached_dir: str = None, 
                 debug: bool = False, device: str = None, scene_names=None, 
                 supervised: bool = False, tonemap: str = None, target_split: int = 1, 
                 run_mode: str = None, use_cached_crops: bool = False, log_bins: bool = False,
                 stat: bool = False, input_tonemap: str = 'log'):
        
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
        self.stat = stat
        self.input_tonemap = input_tonemap
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

                    patch_cache_path = os.path.join(self.cached_dir, f"{key}_{crops_per_scene}_{self.crop_size}.pkl")

                    if self.use_cached_crops and os.path.exists(patch_cache_path):
                        logger.info(f"Loading cached patches from {patch_cache_path}")
                        scene_patches = load_patches(patch_cache_path)
                    else:
                        logger.info(f"Generating patches for scene {key}")
                        scene_patches = []
                        start_time = time.time()

                        # random crops because if selected based on variance it overfits!
                        for _ in range(self.crops_per_scene):
                            # i, j, h, w = sample_crop_coords_from_variance(varmap, self.crop_size)
                            i, j, h, w = RandomCrop.get_params(varmap, output_size=(self.crop_size, self.crop_size))
                            patch_var = varmap[i:i+h, j:j+w].mean().item()
                            scene_patches.append({
                                "scene": key,
                                "crop_coords": (i, j, h, w),
                                "variance": patch_var
                            })
                        elapsed = time.time() - start_time
                        logger.info(f"Finished generating {self.crops_per_scene} crops for scene '{key}' in {elapsed:.2f} seconds!")

                    self.patches.extend(scene_patches)

        logger.info(f"Total patches: {len(self.patches)}")

    def __len__(self):
        return len(self.patches)
    
    def get_sampling_weights(self, normalize=True, min_clip=1e-6):
        variances = np.array([p["variance"] for p in self.patches])
        variances = np.clip(variances, min_clip, None)
        if normalize:
            variances = variances / variances.sum()
        return variances

    def __getitem__(self, idx, coords=None):
        patch_info = self.patches[idx]
        scene = patch_info["scene"]
        paths = self.scene_paths[scene]
        if coords==None:
            i, j, h, w = patch_info["crop_coords"]
        else: 
            i, j, h, w = coords

        # NOISY
        spp1_img = self.spp1_images[scene]                          # (N, H, W, 3)
        noisy_tensor = self.noisy_images[scene][:, i:i+h, j:j+w]    # (H, W, 3)

        # CLEAN
        clean_tensor = None
        if self.clean and paths["clean"] is not None:
            clean_img = tifffile.imread(paths["clean"])
            clean_tensor = torch.from_numpy(clean_img).permute(2, 0, 1).float()[:, i:i+h, j:j+w]

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

        # STAT
        if self.stat:
            rgb_stats = generate_hist_statistics(input_samples_tensor, return_channels='hdr')
            mean_img = apply_tonemap(rgb_stats["mean"], tonemap=self.input_tonemap)
            rel_var = apply_tonemap(rgb_stats["relative_variance"], tonemap=self.input_tonemap)
            input_tensor = torch.cat([input_tensor, mean_img, rel_var], dim=0)

        # AOV
        albedo_tensor, normal_tensor = None, None
        if self.aov:
            if paths["albedo"]:
                albedo_img = tifffile.imread(paths["albedo"])
                albedo_tensor = torch.from_numpy(albedo_img).permute(2, 0, 1).float()[:, i:i+h, j:j+w]
                albedo_tensor = apply_tonemap(albedo_tensor, tonemap="log")
                input_tensor = torch.cat([input_tensor, albedo_tensor], dim=0)
            if paths["normal"]:
                normal_img = tifffile.imread(paths["normal"])
                normal_tensor = torch.from_numpy(normal_img).permute(2, 0, 1).float()[:, i:i+h, j:j+w]
                normal_tensor = (normal_tensor + 1.0) * 0.5
                input_tensor = torch.cat([input_tensor, normal_tensor], dim=0)

        # DATA AUGMENTATION
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
                 high_spp: int = 4500, cached_dir: str = None, aov: bool = False,
                 debug: bool = False, mode: str = None, device: str = None, scene_names=None, 
                 target_sample: int = 1, stat: bool = True, log_bins = True, p_val: float = 0.5):
        
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.data_augmentation = data_augmentation
        self.virt_size = virt_size
        self.hist_bins = hist_bins
        self.clean = clean
        self.stat = stat
        self.aov = aov
        self.low_spp = low_spp
        self.high_spp = high_spp
        self.cached_dir = cached_dir
        self.debug = debug
        self.device = device
        logger.info(f"Using device Data Loader: {self.device}")
        self.target_sample = target_sample
        self.log_bins = log_bins
        self.p_val = p_val

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

        self.fixed_edges = compute_fixed_log_edges_from_scenes(
            scene_index=scene_keys,
            low_spp=self.low_spp,
            bins=self.hist_bins,
            device=self.device if self.device is not None else "cpu",
            p_lo=0.005, p_hi=0.995,
            max_pixels_per_channel=2_000_000,
        )  # (3, bins+1)
        logger.info(f"Fixed edges ready (per-channel): shape={tuple(self.fixed_edges.shape)}")
        logger.info(f"Fixed edges ready (per-channel): R - Min: {self.fixed_edges[0].min()}  Max: {self.fixed_edges[0].max()}")
        logger.info(f"Fixed edges ready (per-channel): G - Min: {self.fixed_edges[1].min()}  Max: {self.fixed_edges[1].max()}")
        logger.info(f"Fixed edges ready (per-channel): B - Min {self.fixed_edges[2].min()}  Max: {self.fixed_edges[2].max()}")

        for key, folder in scene_keys:
            # Load spp1 samples (32 samples, shape: (32, H, W, 3))
            spp1_file = next((f for f in os.listdir(folder) if f.startswith(key) and f.endswith(f"spp1x{self.low_spp}.tiff")), None)
            albedo_file = next((f for f in os.listdir(full_subdir) if f.endswith("albedo.tiff")), None)
            normal_file = next((f for f in os.listdir(full_subdir) if f.endswith("normal.tiff")), None)
            
            spp1_path = os.path.join(folder, spp1_file)
            albedo_path = os.path.join(full_subdir, albedo_file)
            normal_path = os.path.join(full_subdir, normal_file)
            spp1_samples = tifffile.imread(spp1_path)                                         # (low_spp, H, W, 3)
            spp1_samples = torch.from_numpy(spp1_samples).permute(0, 3, 1, 2).float()         # (N, 3, H, W)

            assert spp1_file is not None, f"Missing spp1 file for scene {key}"
            
            # hist shape: (3, H, W, B)
            # edges shape: (3, H, W, B)
            hist, bin_edges = load_or_compute_histograms(
                key=key,
                spp1_tensor=spp1_samples,
                hist_bins=self.hist_bins,
                device=self.device,
                cached_dir=self.cached_dir,
                log_binning=self.log_bins,
                normalize=False,
                fixed_edges=self.fixed_edges
            ) 

            # generate statistics and tone map them otherwise they'll be in HDR range !!!
            stats = generate_hist_statistics(spp1_samples, return_channels='all')
            pixel_mean = stats['mean']                                                      # (3, H, W)
            rel_var = stats['relative_variance']                                            # (3, H, W)
            cov_matrix = compute_covariance_matrix(spp1_samples)                            # (6, H, W)

            # logger.info(f"Mean Min {pixel_mean.min()} - Max {pixel_mean.max()} - Mean {pixel_mean.mean()} - Var {pixel_mean.var()}")
            # logger.info(f"Var Min {rel_var.min()} - Max {rel_var.max()} - Mean {rel_var.mean()} - Var {rel_var.var()}")

            pixel_mean = apply_tonemap(pixel_mean, tonemap="log") 
            rel_var = apply_tonemap(rel_var, tonemap="log") 

            # logger.info(f"Mean Min {pixel_mean.min()} - Max {pixel_mean.max()} - Mean {pixel_mean.mean()} - Var {pixel_mean.var()}")
            # logger.info(f"Var Min {rel_var.min()} - Max {rel_var.max()} - Mean {rel_var.mean()} - Var {rel_var.var()}")

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
                "clean_path": clean_path, 
                "albedo_path": albedo_path,
                "normal_path": normal_path
            }

    def __len__(self):
        return self.virt_size

    def __getitem__(self, idx, crop_coords=None):
        scene = self.scene_names[idx % len(self.scene_names)] 
        scene_cache = self.cached_data[scene]
        hist = scene_cache['hist']
        # bin_edges = scene_cache['bin_edges']

        # CROP
        if self.crop_size:
            _, H, W, _ = hist.shape
            if crop_coords is None:
                i, j, h, w = transforms.RandomCrop.get_params(torch.empty((H, W)), output_size=(self.crop_size, self.crop_size))
            else:
                i, j, h, w = crop_coords
            hist = hist[:, i:i+h, j:j+w, :]                                 # (3, H, W, B)

        # CLEAN
        clean_tensor = None
        if self.clean:
            clean_path = self.scene_paths[scene]["clean_path"]
            clean_img = tifffile.imread(clean_path)
            clean_tensor = torch.from_numpy(clean_img).permute(2, 0, 1).float()
            if self.crop_size:
                clean_tensor = clean_tensor[:, i:i+h, j:j+w]

        # BINOMIAL SPLIT
        p = torch.tensor(self.p_val, device=hist.device, dtype=torch.float32)
        binom = torch.distributions.Binomial(total_count=hist.to(torch.float32), probs=p)
        target_hist = binom.sample()
        input_hist  = hist - target_hist

        # SAVE target counts for loss
        target_counts_raw = target_hist.clone()

        # NORMALISATION
        target_hist = target_hist / (target_hist.sum(dim=-1, keepdim=True) + 1e-8)          # shape (3, H, W, B)
        input_hist = input_hist / (input_hist.sum(dim=-1, keepdim=True) + 1e-8)             # shape (3, H, W, B)
        input_tensor = input_hist.permute(0, 3, 1, 2).contiguous().float()                  # shape (3, B, H, W)
        target_tensor = target_hist.permute(0, 3, 1, 2).contiguous().float()                # shape (3, B, H, W)

        # MEAN and RELATIVE VARIANCE from samples
        if self.stat:
            pixel_mean_expanded = scene_cache['mean'][:, i:i+h, j:j+w].unsqueeze(1)                     # (3, 1, H, W)
            pixel_var_expanded = scene_cache['rel_var'][:, i:i+h, j:j+w].unsqueeze(1)                   # (3, 1, H, W)

            input_tensor = torch.cat([input_tensor, pixel_mean_expanded, pixel_var_expanded], dim=1)    # (3, B+1+6, H, W)

        # AOV
        albedo_tensor, normal_tensor = None, None
        aov_tensors = []
        if self.aov:
            if self.scene_paths[scene]["albedo_path"]:
                albedo_img = tifffile.imread(self.scene_paths[scene]["albedo_path"])
                albedo_tensor = torch.from_numpy(albedo_img).permute(2, 0, 1).float()[:, i:i+h, j:j+w]
                albedo_tensor = apply_tonemap(albedo_tensor, tonemap="reinhard")  # (3, H, W)
                aov_tensors.append(albedo_tensor)
            if self.scene_paths[scene]["normal_path"]:
                normal_img = tifffile.imread(self.scene_paths[scene]["normal_path"])
                normal_tensor = torch.from_numpy(normal_img).permute(2, 0, 1).float()[:, i:i+h, j:j+w]
                normal_tensor = (normal_tensor + 1.0) * 0.5  # Map from [-1,1] to [0,1]
                aov_tensors.append(normal_tensor)

        # DATA AUGMENTATION
        if self.data_augmentation:
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-1])
                target_tensor = torch.flip(target_tensor, dims=[-1])
                clean_tensor = torch.flip(clean_tensor, dims=[-1])
                aov_tensors = [torch.flip(t, dims=[-1]) for t in aov_tensors]
                target_counts_raw = torch.flip(target_counts_raw, dims=[2])
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, dims=[-2])
                target_tensor = torch.flip(target_tensor, dims=[-2])
                clean_tensor = torch.flip(clean_tensor, dims=[-2])
                aov_tensors = [torch.flip(t, dims=[-2]) for t in aov_tensors]
                target_counts_raw = torch.flip(target_counts_raw, dims=[1])

        return {
            "input_hist": input_tensor,                 # (3, B, H, W) normalized
            "target_hist": target_tensor,               # (3, B, H, W) normalized
            "target_counts": target_counts_raw,         # (3, B, H, W) raw counts
            "aov_tensors": aov_tensors,                 # list of normal and albedo (3, H, W) each
            "clean": clean_tensor,                      # (3, H, W)
            "scene": scene,
            "bin_edges": self.fixed_edges,
            "crop_coords": (i, j, h, w) if self.crop_size else None
        }
