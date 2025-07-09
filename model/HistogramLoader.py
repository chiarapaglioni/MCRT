from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.HistImgDataset import HistogramBinomDataset

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
            input_hist = batch['input'][0]  # (3, B, H, W)
            r_hist = input_hist[0].mean(dim=(1, 2)).numpy()  # Mean over H, W
            g_hist = input_hist[1].mean(dim=(1, 2)).numpy()
            b_hist = input_hist[2].mean(dim=(1, 2)).numpy()

            plt.plot(r_hist, label='R')
            plt.plot(g_hist, label='G')
            plt.plot(b_hist, label='B')
            plt.legend()
            plt.title("Averaged histogram bins (sample 0)")
            plt.show()
        
        break  # Only one batch for testing