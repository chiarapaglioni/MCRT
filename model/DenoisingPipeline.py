# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Path
import os
from pathlib import Path

# Custom
from model.UNet import UNet
from dataset.HistImgDataset import HistogramBinomDataset


# GPU Check
if not torch.cuda.is_available():
    print("GPU not found, code will run on CPU and can be extremely slow!")
else:
    device = torch.device("cuda:0")


def get_data_loaders(config):
    # Resolve root_dir path
    dataset_cfg = config['dataset'].copy()
    dataset_cfg['root_dir'] = Path(__file__).resolve().parents[1] / dataset_cfg['root_dir']

    # Instantiate the full dataset (mode param ignored here)
    full_dataset = HistogramBinomDataset(**dataset_cfg)

    # Compute split sizes
    val_ratio = config.get('val_split', 0.2)
    total_len = len(full_dataset)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    # Split dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Create DataLoaders with differing shuffle flags
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False,
        drop_last=False
    )

    train_img = next(iter(train_loader))

    print("Input shape:", train_img['input'].shape)
    print("Target shape:", train_img['target'].shape)
    print("Noisy shape:", train_img['noisy'].shape)
    if 'clean' in train_img:
        print("Clean shape:", train_img['clean'].shape)

    return train_loader, val_loader


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        hist = batch['input'].to(device)        # B, 3, H, W
        target = batch['target'].to(device)     # 3, H, W

        optimizer.zero_grad()
        pred = model(hist)                      # 3, H, W
        
        # print("Input shape: ", hist.shape)
        # print("Target shape: ", target.shape)
        # print("Pred shape: ", target.shape)

        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0 

    with torch.no_grad():
        for batch in dataloader:
            hist = batch['input'].to(device)         # B, 3, H, W
            target = batch['target'].to(device)      # 3, H, W

            pred = model(hist)
            loss = criterion(pred, target)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_data_loaders(config)

    dataset_cfg = config['dataset']
    model_cfg = config['model']

    print(f"\nDataset Config: {config['dataset']['mode'].upper()} mode | Crop Size: {config['dataset']['crop_size']} | Augmentation: {config['dataset']['data_augmentation']}")
    print(f"Model Config: Depth={config['model']['depth']} | Start Filters={config['model']['start_filters']} | Output: {config['model']['out_mode']}")
    print(f"Training for {config['num_epochs']} epochs | Batch Size: {config['batch_size']} | Val Split: {config['val_split']} | Learning Rate: {config['model']['learning_rate']}\n")

    model = UNet(
        in_channels=model_cfg['in_channels'],
        n_bins=dataset_cfg['hist_bins'],
        out_mode=model_cfg['out_mode'],
        merge_mode=model_cfg['merge_mode'],
        depth=model_cfg['depth'],
        start_filters=model_cfg['start_filters'],
        mode=dataset_cfg['mode']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(model_cfg["learning_rate"]), weight_decay=float(model_cfg["weight_decay"]))
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    num_epochs = config["num_epochs"]
    save_path = config["save_path"]

    print("TRAINING STARTED !")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")