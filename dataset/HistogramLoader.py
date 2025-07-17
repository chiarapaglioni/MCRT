import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Datasets
from dataset.HistImgDataset import HistogramBinomDataset
from dataset.HistDataset import HistogramDataset

# Logger
import logging
logger = logging.getLogger(__name__)


def test_data_loader(config):
    dataset_cfg = config['dataset']
    out_mode = config.get("out_mode", "dist")
    hist_bins = dataset_cfg.get("hist_bins", 16)

    # Resolve dataset root
    dataset_cfg['root_dir'] = Path(__file__).resolve().parents[1] / dataset_cfg['root_dir']

    logger.info(f"Data Loader - Mode: {out_mode} !!")

    # Pick dataset based on out_mode
    if out_mode == "dist":
        dataset = HistogramDataset(**dataset_cfg)
        input_key = 'input_hist'
        target_key = 'target_hist'
    elif out_mode == "mean":
        dataset = HistogramBinomDataset(**dataset_cfg)
        input_key = 'input'
        target_key = 'target'
    else:
        raise ValueError(f"Unsupported out_mode: {out_mode}. Must be 'dist' or 'mean'.")

    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 1),
        num_workers=config.get('num_workers', 0),
        shuffle=config.get('shuffle', False)
    )

    for batch in dataloader:
        logger.info(f"Batch keys: {batch.keys()}")
        logger.info(f"Input shape:  {batch[input_key].shape}")
        logger.info(f"Target shape: {batch[target_key].shape}")
        if 'clean' in batch and batch['clean'] is not None:
            logger.info(f"Clean shape: {batch['clean'].shape}")
        logger.info(f"Scene: {batch['scene']}")

        # Plot histograms for a pixel (center)
        input_hist = batch[input_key][0].numpy()   # shape: (C, B, H, W)
        x, y = input_hist.shape[2] // 2, input_hist.shape[3] // 2

        n_plots = 2 if out_mode == "dist" else 1
        fig, axs = plt.subplots(1, n_plots, figsize=(10, 4))
        axs = [axs] if n_plots == 1 else axs

        def plot_hist_bar(ax, hist, title, used_bins):
            r = hist[0, :used_bins, y, x]
            g = hist[1, :used_bins, y, x]
            b = hist[2, :used_bins, y, x]

            bar_width = 0.25
            bin_positions = np.arange(used_bins)

            ax.bar(bin_positions - bar_width, r, width=bar_width, color='red', label='R')
            ax.bar(bin_positions, g, width=bar_width, color='green', label='G')
            ax.bar(bin_positions + bar_width, b, width=bar_width, color='blue', label='B')

            ax.set_title(title)
            ax.set_xlabel("Bin")
            ax.set_ylabel("Counts")
            ax.set_xticks(bin_positions)
            ax.legend()

        if out_mode == "mean":
            used_bins = hist_bins  # Only the histogram part
            plot_hist_bar(axs[0], input_hist, "Input Histogram", used_bins)
        else:  # dist mode
            used_bins = input_hist.shape[1]  # Full histogram range
            target_hist = batch[target_key][0].numpy()
            plot_hist_bar(axs[0], input_hist, "Input Histogram", used_bins)
            plot_hist_bar(axs[1], target_hist, "Target Histogram", used_bins)

        plt.tight_layout()
        plt.show()

        break  # Just test one batch
