import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from renderer.SceneRenderer import SceneRenderer
from torch.utils.data import DataLoader, random_split
from dataset.HistogramGenerator import generate_histograms_torch
from dataset.HistImgDataset import AdaptiveSamplingDataset
from utils.utils import compute_psnr
from model.ClassicUNet import UNetImportancePredictor
from torch.utils.data import DataLoader
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


class AdaptiveSampler:
    def __init__(
        self,
        scene_path,
        width=512,
        height=512,
        num_bins=64,
        device='cpu',
        debug=False,
        # new adaptive parameters
        initial_samples=64,
        initial_spp=1,
        extra_samples=64,
        extra_spp=4,
        quantile_threshold=0.8,
        mode='hist',
    ):
        self.renderer = SceneRenderer(scene_path, width, height, debug)
        self.num_bins = num_bins
        self.device = device
        self.debug = debug

        # Store adaptive sampling parameters
        self.initial_samples = initial_samples
        self.initial_spp = initial_spp
        self.extra_samples = extra_samples
        self.extra_spp = extra_spp
        self.quantile_threshold = quantile_threshold
        self.mode = mode

    def initial_render(self, seed_start=0):
        if self.debug:
            print(f"Starting initial render: {self.initial_samples} images @ {self.initial_spp} spp")
        start = time.time()
        images = self.renderer.render_n_images(self.initial_samples, spp=self.initial_spp, seed_start=seed_start)
        duration = time.time() - start
        if self.debug:
            print(f"Initial render done in {duration:.2f} seconds")
        samples_np = np.stack(images, axis=0)  # (N, H, W, 3)
        samples = torch.from_numpy(samples_np).float().to(self.device)
        return samples, duration

    def compute_importance_map(self, samples):
        if self.mode == 'stat':
            # === Standard approach: Mean / Variance-based Importance ===
            variance = torch.var(samples, dim=0).mean(dim=-1)  # (H, W)
            importance_map = variance
            hist = None
            bin_edges = None
        else:
            # === Histogram-based Importance (optionally learned) ===
            hist, bin_edges = generate_histograms_torch(samples, num_bins=self.num_bins, device=self.device)
            if self.learned_model:
                # If model provided, use it to predict importance map
                with torch.no_grad():
                    importance_map = self.learned_model(hist).squeeze(0).cpu()
            else:
                # Default: use max bin count as heuristic
                max_bin_count = hist.max(dim=-1).values
                importance_map = max_bin_count.float().mean(dim=-1).cpu()

        return importance_map.numpy(), hist, bin_edges
    
    def get_adaptive_mask(self, importance_map):
        threshold = np.quantile(importance_map, self.quantile_threshold)
        mask = importance_map > threshold
        if self.debug:
            print(f"Adaptive mask threshold: {threshold:.3f} ({self.quantile_threshold*100:.0f} percentile)")
            print(f"Pixels to resample: {mask.sum()} / {mask.size}")
        return mask

    def render_additional_samples(self, seed_start=0):
        if self.debug:
            print(f"Rendering additional {self.extra_samples} images @ {self.extra_spp} spp")
        start = time.time()
        images = self.renderer.render_n_images(self.extra_samples, spp=self.extra_spp, seed_start=seed_start)
        duration = time.time() - start
        if self.debug:
            print(f"Additional render done in {duration:.2f} seconds")
        samples_np = np.stack(images, axis=0)
        samples = torch.from_numpy(samples_np).float().to(self.device)
        return samples, duration

    def merge_samples(self, old_samples, new_samples, mask=None):
        if mask is None:
            merged = torch.cat([old_samples, new_samples], dim=0)
        else:
            mask = torch.from_numpy(mask).to(self.device).unsqueeze(-1)
            merged = torch.cat([old_samples, new_samples], dim=0)
        return merged

    def adaptive_sampling_loop(self):
        old_samples, t_initial = self.initial_render(seed_start=0)
        importance_map, hist, bin_edges = self.compute_importance_map(old_samples)
        mask = self.get_adaptive_mask(importance_map)
        new_samples, t_extra = self.render_additional_samples(seed_start=self.initial_samples)
        all_samples = self.merge_samples(old_samples, new_samples, mask=mask)
        final_importance, _, _ = self.compute_importance_map(all_samples)

        if self.debug:
            print(f"Adaptive sampling completed: total samples = {all_samples.shape[0]}")
            print(f"Total render time: {t_initial + t_extra:.2f} seconds")

        return {
            'samples': all_samples,
            'importance_map_initial': importance_map,
            'importance_map_final': final_importance,
            'render_time': t_initial + t_extra,
            'mask': mask
        }

def to_numpy_image(tensor):
    """Convert (H,W,3) torch tensor to numpy image"""
    return tensor.detach().cpu().clamp(0, 1).numpy()

def run_adaptive_sampling(config):
    adaptive_sampler = AdaptiveSampler(
        scene_path=config["scene_path"],
        width=config["resolution"]["width"],
        height=config["resolution"]["height"],
        num_bins=config["histogram_bins"],
        debug=config["debug"],
        initial_samples=config["initial_samples"],
        initial_spp=config["initial_spp"],
        extra_samples=config["extra_samples"],
        extra_spp=config["extra_spp"],
        quantile_threshold=config["quantile_threshold"], 
        mode=config['mode']
    )
    device = adaptive_sampler.device
    
    # Run adaptive sampling
    results = adaptive_sampler.adaptive_sampling_loop()

    # Render high-quality ground truth (reference)
    gt_images = adaptive_sampler.renderer.render_n_images(n=1, spp=100, seed_start=999)
    reference = torch.from_numpy(gt_images[0]).float().to(device)  # (H, W, 3)

    # Compute initial and final means
    init_img = results['samples'][:config["initial_samples"]].mean(dim=0)
    final_img = results['samples'].mean(dim=0)

    # Compute PSNR
    psnr_init = compute_psnr(init_img, reference)
    psnr_final = compute_psnr(final_img, reference)

    # Print timing and PSNR
    print("\n=== Summary ===")
    print(f"Total render time: {results['render_time']:.2f} seconds")
    print(f"Initial PSNR: {psnr_init:.2f} dB")
    print(f"Final PSNR:   {psnr_final:.2f} dB")

    # Convert to numpy for plotting
    init_img_np = to_numpy_image(init_img)
    final_img_np = to_numpy_image(final_img)
    ref_img_np = to_numpy_image(reference)
    importance_map = results['importance_map_initial']
    diff_img = np.abs(final_img_np - ref_img_np)

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(init_img_np)
    axes[0].set_title(f"Initial Mean ({config['initial_samples']}x1spp)")

    axes[1].imshow(final_img_np)
    axes[1].set_title("Final Adaptive Result")

    axes[2].imshow(ref_img_np)
    axes[2].set_title("Reference (2048 spp)")

    axes[3].imshow(importance_map, cmap='hot')
    axes[3].set_title("Importance Map")

    axes[4].imshow(diff_img)
    axes[4].set_title("Error Map (|Final - GT|)")

    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


class ImportanceTrainer:
    def __init__(self,
                 model,
                 train_dataset,
                 val_dataset=None,
                 batch_size=8,
                 lr=1e-3,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 checkpoint_dir='./checkpoints',
                 save_best_only=True,
                 loss='mse'):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_dataset
        self.val_loader = val_dataset
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.save_best_only = save_best_only
        self.best_val_loss = float('inf')
        self.loss = loss

        if self.loss == "mse":
            self.criterion = torch.nn.MSELoss()
        elif self.loss == "l1":
            self.criterion = torch.nn.L1Loss()

    def train_epoch(self):
        self.model.train()
        running_loss = 0
        for batch in self.train_loader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        if self.val_loader is None:
            return None
        self.model.eval()
        running_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].unsqueeze(1).to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
        avg_loss = running_loss / len(self.val_loader)
        return avg_loss

    def save_checkpoint(self, epoch, val_loss):
        path = os.path.join(self.checkpoint_dir, f'importance_model_epoch{epoch}_val{val_loss:.4f}.pth')
        torch.save(self.model.state_dict(), path)
        print(f"Saved checkpoint: {path}")

    def fit(self, epochs=10):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}", end='')
            if val_loss is not None:
                print(f" - Val Loss: {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.save_best_only:
                        self.save_checkpoint(epoch, val_loss)
            else:
                print()

def run_adaptive_sampling(config):
    dataset_cfg = config['dataset']
    model_cfg = config['model']

    full_dataset = AdaptiveSamplingDataset(root_dir=dataset_cfg['root_dir'], mode=dataset_cfg['mode'])
    total_len = len(full_dataset)
    val_ratio = config.get('val_split', 0.1)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    train_set, val_set = random_split(full_dataset, [train_len, val_len])

    logger.info(f"Total dataset size: {total_len}")
    logger.info(f"Training set size:  {len(train_set)}")
    logger.info(f"Validation set size: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    logger.info(f"Running Adaptive Sampling with Mode {dataset_cfg['mode'].upper()}")

    model = UNetImportancePredictor(mode=dataset_cfg['mode'], num_bins=dataset_cfg['hist_bins'])

    trainer = ImportanceTrainer(model, train_loader, val_loader, batch_size=config['batch_size'], lr=model_cfg['learning_rate'], loss=config['loss'])

    # Train
    trainer.fit(epochs=20)