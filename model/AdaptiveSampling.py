import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from renderer.SceneRenderer import SceneRenderer
from dataset.HistogramGenerator import generate_histograms_torch
from utils.utils import compute_psnr

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
        quantile_threshold=0.8
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

    def compute_histogram_and_importance(self, samples):
        hist, bin_edges = generate_histograms_torch(samples, num_bins=self.num_bins, device=self.device)
        max_bin_count = hist.max(dim=-1).values
        importance_map = max_bin_count.mean(dim=-1)
        return importance_map.cpu().numpy(), hist, bin_edges

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
        importance_map, hist, bin_edges = self.compute_histogram_and_importance(old_samples)
        mask = self.get_adaptive_mask(importance_map)
        new_samples, t_extra = self.render_additional_samples(seed_start=self.initial_samples)
        all_samples = self.merge_samples(old_samples, new_samples, mask=mask)
        final_importance, _, _ = self.compute_histogram_and_importance(all_samples)

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
        quantile_threshold=config["quantile_threshold"]
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
