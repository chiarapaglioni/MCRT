from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.HistImgDataset import HistogramBinomDataset

def test_data_loader(config):
    # Dataset configuration
    dataset_cfg = config['dataset']

    # Resolve root_dir relative to this script
    dataset_cfg['root_dir'] = Path(__file__).resolve().parents[1] / dataset_cfg['root_dir']

    # Instantiate dataset
    dataset = HistogramBinomDataset(**dataset_cfg)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=config['shuffle'])

    # Iterate over a few batches
    for batch in dataloader:
        print("Keys:", batch.keys())
        print("Input shape:", batch['input'].shape)     # (B, 3, B, H, W) in 'hist' mode
        print("Target shape:", batch['target'].shape)   # same
        print("Noisy shape:", batch['noisy'].shape)     # (B, 3, H, W)
        if 'clean' in batch:
            print("Clean shape:", batch['clean'].shape) # (B, 3, H, W)
        print("Scene:", batch['scene'])

        # Optional: visualize histograms
        if dataset.mode == 'hist':
            input_hist = batch['input'][0]  # shape: (3, bins+2, H, W)
            print("Histogram shape (C, B+2, H, W):", input_hist.shape)

            input_hist_np = input_hist.numpy()

            # Choose a pixel coordinate
            x, y = 50, 50

            num_bins = config['dataset']['hist_bins']  # e.g., 8

            # Extract hist bins and stats
            r_hist = input_hist_np[0, :num_bins, y, x]
            g_hist = input_hist_np[1, :num_bins, y, x]
            b_hist = input_hist_np[2, :num_bins, y, x]

            r_mean = input_hist_np[0, num_bins, y, x]
            g_mean = input_hist_np[1, num_bins, y, x]
            b_mean = input_hist_np[2, num_bins, y, x]

            r_var = input_hist_np[0, num_bins + 1, y, x]
            g_var = input_hist_np[1, num_bins + 1, y, x]
            b_var = input_hist_np[2, num_bins + 1, y, x]

            # Print
            print(f"Pixel ({x},{y}):")
            print(f"  Red   - bins: {r_hist}, mean: {r_mean:.4f}, var: {r_var:.4f}")
            print(f"  Green - bins: {g_hist}, mean: {g_mean:.4f}, var: {g_var:.4f}")
            print(f"  Blue  - bins: {b_hist}, mean: {b_mean:.4f}, var: {b_var:.4f}")

            # Plot
            bins = range(num_bins)
            plt.plot(bins, r_hist, label='Red')
            plt.plot(bins, g_hist, label='Green')
            plt.plot(bins, b_hist, label='Blue')
            plt.xlabel('Bin')
            plt.ylabel('Counts')
            plt.title(f'Histogram bin counts for pixel ({x},{y})')
            plt.legend()
            plt.show()
                
        break