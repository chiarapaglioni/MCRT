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
                 clean: bool = False):
        self.root_dir = root_dir
        self.mode = mode
        self.crop_size = crop_size
        self.data_augmentation = data_augmentation
        self.virt_size = virt_size
        self.low_spp = low_spp
        self.hist_bins = hist_bins
        self.clean = clean

        self.scene_names = sorted(os.listdir(self.root_dir))
        assert self.scene_names, f"No scenes found in {self.root_dir}"

        self.spp1_images = {}
        self.noisy_images = {}
        self.clean_images = {}

        for scene in self.scene_names:
            scene_path = os.path.join(self.root_dir, scene)
            files = os.listdir(scene_path)

            spp1_file = next((f for f in files if f.endswith(f'spp1x{self.low_spp}.tiff')), None)
            noisy_file = next((f for f in files if f.endswith(f'spp{self.low_spp}.tiff')), None)
            clean_file = next((f for f in files if f.endswith(f'spp{high_spp}.tiff')), None)

            assert spp1_file and noisy_file, f"Missing required files in scene: {scene}"

            spp1_path = os.path.join(scene_path, spp1_file)
            noisy_path = os.path.join(scene_path, noisy_file)

            self.spp1_images[scene] = tifffile.imread(spp1_path)  # (low_spp, H, W, 3)
            self.noisy_images[scene] = torch.from_numpy(tifffile.imread(noisy_path)).permute(2, 0, 1).float()  # (3, H, W)

            if self.clean and clean_file:
                clean_path = os.path.join(scene_path, clean_file)
                self.clean_images[scene] = torch.from_numpy(tifffile.imread(clean_path)).permute(2, 0, 1).float()

    def __len__(self):
        return self.virt_size

    def __getitem__(self, idx):
        scene = random.choice(self.scene_names)
        spp1_img = self.spp1_images[scene]  # (low_spp, H, W, 3)
        noisy_tensor = self.noisy_images[scene]  # (3, H, W)
        clean_tensor = self.clean_images.get(scene, None)

        # print("\nSPP1 Shape: ", spp1_img.shape)
        # print("Noisy Shape: ", noisy_tensor.shape)
        # print("Clean Shape: ", clean_tensor.shape)

        indices = list(range(self.low_spp))
        random.shuffle(indices)
        input_indices = indices[:-1]
        target_index = indices[-1]

        input_samples = spp1_img[input_indices]  # (N-1, H, W, 3)
        target_sample = spp1_img[target_index]   # (H, W, 3)

        # print("Input Shape: ", input_samples.shape)
        # print("Target Shape: ", target_sample.shape)

        if self.mode == 'hist':
            input_hist, _ = generate_histograms(input_samples, self.hist_bins)  # (H, W, 3, bins)
            # print("Input Hist Shape: ", input_hist.shape)
            input_hist = np.transpose(input_hist, (2, 3, 0, 1))  # → (3, bins, H, W)
            input_tensor = torch.from_numpy(input_hist).float()

        else:
            input_avg = input_samples.mean(axis=0)  # (H, W, 3)
            # print("Input Image Shape: ", input_avg.shape)
            input_tensor = torch.from_numpy(input_avg).permute(2, 0, 1).float()  # (3, H, W)

        target_tensor = torch.from_numpy(target_sample).permute(2, 0, 1).float()  # (3, H, W)
        # print("Target Shape: ", target_tensor.shape)

        # --- RANDOM CROP ---
        if self.crop_size:
            i, j, h, w = transforms.RandomCrop.get_params(target_tensor, output_size=(self.crop_size, self.crop_size))
            if self.mode == 'hist':
                input_tensor = input_tensor[:, :, i:i+h, j:j+w]  # (3, bins, H, W)
            else:
                input_tensor = input_tensor[:, i:i+h, j:j+w]
            target_tensor = target_tensor[:, i:i+h, j:j+w]
            noisy_tensor = noisy_tensor[:, i:i+h, j:j+w]
            if clean_tensor is not None:
                clean_tensor = clean_tensor[:, i:i+h, j:j+w]

        # --- DATA AUGMENTATION ---
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

        # --- Final dictionary ---
        sample = {
            'input': input_tensor,     # (3, bins, H, W) or (3, H, W)
            'target': target_tensor,   # (3, H, W)
            'noisy': noisy_tensor,
            'scene': scene
        }
        if clean_tensor is not None:
            sample['clean'] = clean_tensor

        return sample
