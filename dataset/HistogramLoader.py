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
            input_hist = batch['input'][0]  # shape: (3, B, H, W)

            print("Histogram shape (C, B, H, W):", input_hist.shape)

            # Convert to numpy for inspection
            input_hist_np = input_hist.numpy()

            # Choose a pixel coordinate, for example (x=100, y=150)
            x, y = 50, 50

            # input_hist_np shape: (3, B, H, W)
            # Extract histogram bins counts for that pixel for each channel:
            r_pixel_hist = input_hist_np[0, :, y, x]  # shape (B,)
            g_pixel_hist = input_hist_np[1, :, y, x]
            b_pixel_hist = input_hist_np[2, :, y, x]

            print(f"Red pixel histogram counts at ({x},{y}):", r_pixel_hist)
            print(f"Green pixel histogram counts at ({x},{y}):", g_pixel_hist)
            print(f"Blue pixel histogram counts at ({x},{y}):", b_pixel_hist)

            print("Sum of bins per channel at this pixel:")
            print("Red:", r_pixel_hist.sum())
            print("Green:", g_pixel_hist.sum())
            print("Blue:", b_pixel_hist.sum())

            # Optional: plot histograms for the pixel
            import matplotlib.pyplot as plt

            bins = range(len(r_pixel_hist))  # number of bins

            plt.plot(bins, r_pixel_hist, label='Red')
            plt.plot(bins, g_pixel_hist, label='Green')
            plt.plot(bins, b_pixel_hist, label='Blue')
            plt.xlabel('Bin')
            plt.ylabel('Counts')
            plt.title(f'Histogram bin counts for pixel ({x},{y})')
            plt.legend()
            plt.show()
        
        break