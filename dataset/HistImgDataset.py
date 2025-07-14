import os
import torch
import random
import tifffile
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from dataset.HistogramGenerator import generate_histograms


class HistogramBinomDataset(Dataset):
    def __init__(self, root_dir: str, crop_size: int = 128, mode: str = 'hist',
                data_augmentation: bool = True, virt_size: int = 1000,
                low_spp: int = 32, high_spp: int = 4500, hist_bins: int = 8,
                clean: bool = False, cached_dir: str = None):
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

        for key, folder in scene_keys:
            spp1_file = next((f for f in os.listdir(folder) if f.startswith(key) and f.endswith(f"spp1x{self.low_spp}.tiff")), None)
            noisy_file = next((f for f in os.listdir(folder) if f.startswith(key) and f.endswith(f"spp{self.low_spp}.tiff")), None)
            clean_file = next((f for f in os.listdir(folder) if f.startswith(key) and f.endswith(f"spp{self.high_spp}.tiff")), None)

            assert spp1_file and noisy_file, f"Missing files for scene: {key} in {folder}"

            spp1_path = os.path.join(folder, spp1_file)
            noisy_path = os.path.join(folder, noisy_file)

            self.scene_paths[key] = folder
            self.spp1_images[key] = tifffile.imread(spp1_path)  # (low_spp, H, W, 3)
            self.noisy_images[key] = torch.from_numpy(tifffile.imread(noisy_path)).permute(2, 0, 1).float()  # (3, H, W)

            if self.clean and clean_file:
                clean_path = os.path.join(folder, clean_file)
                self.clean_images[key] = torch.from_numpy(tifffile.imread(clean_path)).permute(2, 0, 1).float()


    def __len__(self):
        return self.virt_size

    def __getitem__(self, idx):
        scene = random.choice(self.scene_names)
        spp1_img = self.spp1_images[scene]  # (low_spp, H, W, 3)
        noisy_tensor = self.noisy_images[scene]  # (3, H, W)
        clean_tensor = self.clean_images.get(scene, None)

        indices = list(range(self.low_spp))
        random.shuffle(indices)
        input_indices = indices[:-1]
        target_index = indices[-1]

        input_samples = spp1_img[input_indices]  # (N-1, H, W, 3)
        target_sample = spp1_img[target_index]   # (H, W, 3)

        # DEBUG INPUT-TARGET VALUES
        debug_coords = [(64, 64), (128, 128), (0, 0)]
        for y, x in debug_coords:
            avg_pixel = input_samples[:, y, x, :].mean(axis=0)     # mean across N-1
            target_pixel = target_sample[y, x, :]                  # single target

            print(f"[DEBUG] Scene: {scene} | Pixel ({x},{y})")
            print(f"Input Mean RGB:  {avg_pixel}")
            print(f"Target Sample RGB: {target_pixel}\n")

        # HISTOGRAM MODE
        if self.mode == 'hist':
            if self.cached_dir is None:
                raise RuntimeError("Histogram mode requires 'cached_dir' for caching.")

            cache_path = os.path.join(self.cached_dir, f"{scene}_hist.npz")

            if os.path.exists(cache_path):
                features = np.load(cache_path)['features']  # (H, W, 3, bins+2)
            else:
                # Compute from full spp1_img, not subset
                full_hist, _ = generate_histograms(spp1_img, self.hist_bins)
                full_mean = spp1_img.mean(axis=0)[..., None]
                full_var = spp1_img.var(axis=0)[..., None]
                features = np.concatenate([full_hist, full_mean, full_var], axis=-1)
                np.savez_compressed(cache_path, features=features)
                print(f"[CACHE] Created {cache_path}")

            # Convert to (3, bins+2, H, W)
            input_tensor = torch.from_numpy(np.transpose(features, (2, 3, 0, 1))).float()

        # IMAGE MODE
        else:
            input_avg = input_samples.mean(axis=0)  # (H, W, 3)
            input_tensor = torch.from_numpy(input_avg).permute(2, 0, 1).float()  # (3, H, W)

        target_tensor = torch.from_numpy(target_sample).permute(2, 0, 1).float()  # (3, H, W)

        # RANDOM CROP
        if self.crop_size:
            i, j, h, w = transforms.RandomCrop.get_params(target_tensor, output_size=(self.crop_size, self.crop_size))
            if self.mode == 'hist':
                input_tensor = input_tensor[:, :, i:i+h, j:j+w]  # (3, bins+2, H, W)
            else:
                input_tensor = input_tensor[:, i:i+h, j:j+w]
            target_tensor = target_tensor[:, i:i+h, j:j+w]
            noisy_tensor = noisy_tensor[:, i:i+h, j:j+w]
            if clean_tensor is not None:
                clean_tensor = clean_tensor[:, i:i+h, j:j+w]

        # DATA AUGMENTATION
        if self.data_augmentation:
            if random.random() < 0.5:
                flip_dim = 3 if self.mode == 'hist' else 2
                input_tensor = torch.flip(input_tensor, dims=[flip_dim])
                target_tensor = torch.flip(target_tensor, dims=[2])
                noisy_tensor = torch.flip(noisy_tensor, dims=[2])
                if clean_tensor is not None:
                    clean_tensor = torch.flip(clean_tensor, dims=[2])
            if random.random() < 0.5:
                flip_dim = 2 if self.mode == 'hist' else 1
                input_tensor = torch.flip(input_tensor, dims=[flip_dim])
                target_tensor = torch.flip(target_tensor, dims=[1])
                noisy_tensor = torch.flip(noisy_tensor, dims=[1])
                if clean_tensor is not None:
                    clean_tensor = torch.flip(clean_tensor, dims=[1])

        # FINAL DICTIONARY
        sample = {
            'input': input_tensor,      # (3, bins, H, W) or (3, H, W)
            'target': target_tensor,    # (3, H, W)
            'noisy': noisy_tensor,      # (3, H, W)
            'scene': scene              # (str) name of the scene
        }
        if clean_tensor is not None:
            sample['clean'] = clean_tensor  # (3, H, W)

        return sample