# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# Path
import os
from pathlib import Path
# Time
import time
from datetime import datetime
# Custom
from model.UNet import UNet
from dataset.HistImgDataset import HistogramBinomDataset
from utils.utils import plot_images
# Eval
from skimage.metrics import peak_signal_noise_ratio as psnr


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
        pin_memory=config['pin_memory'],
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=False
    )

    train_img = next(iter(train_loader))

    print("Input shape:", train_img['input'].shape)
    print("Target shape:", train_img['target'].shape)
    print("Noisy shape:", train_img['noisy'].shape)
    if 'clean' in train_img:
        print("Clean shape:", train_img['clean'].shape)

    return train_loader, val_loader


def load_model(model_path, mode, out_mode='mean', hist_bins=16, device='cpu'):
    model = UNet(
        in_channels=3,
        n_bins=hist_bins,
        out_mode=out_mode,
        merge_mode='concat',
        depth=3,
        start_filters=64,
        mode=mode
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_sample(dataset, idx=0, device='cpu'):
    sample = dataset[idx]
    input_tensor = sample['input'].unsqueeze(0).to(device)   # Add batch dimension
    target = sample['target'].to(device)
    noisy = sample['noisy'].to(device)
    clean = sample.get('clean', None)
    if clean is not None:
        clean = clean.to(device)
    return input_tensor, target, noisy, clean


def evaluate_sample(model, input_tensor, clean_tensor):
    with torch.no_grad():
        pred = model(input_tensor).squeeze(0).clamp(0, 1)
        clean = clean_tensor.clamp(0, 1)

        # Fix dimensions
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if clean.dim() == 3:
            clean = clean.unsqueeze(0)

        print("Target shape:     ", clean.shape)
        print("Pred shape:  ", pred.shape)
        # Calculate PSNR
        psnr_val = psnr(clean.cpu().numpy(), pred.cpu().numpy(), data_range=1.0)
    return pred, psnr_val


# TRAINING STEP
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        hist = batch['input'].to(device)        # B, 3, H, W
        target = batch['target'].to(device)     # 3, H, W

        optimizer.zero_grad()
        pred = model(hist)                      # 3, H, W

        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# VALIDATION STEP
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


# TRAINING LOOP
def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data Loaders
    train_loader, val_loader = get_data_loaders(config)

    dataset_cfg = config['dataset']
    model_cfg = config['model']

    print(f"\nDataset Config: {config['dataset']['mode'].upper()} mode | Crop Size: {config['dataset']['crop_size']} | Augmentation: {config['dataset']['data_augmentation']}")
    print(f"Model Config: Depth={config['model']['depth']} | Start Filters={config['model']['start_filters']} | Output: {config['model']['out_mode']}")
    print(f"Training for {config['num_epochs']} epochs | Batch Size: {config['batch_size']} | Val Split: {config['val_split']} | Learning Rate: {config['model']['learning_rate']}\n")

    # Model
    model = UNet(
        in_channels=model_cfg['in_channels'],
        n_bins=dataset_cfg['hist_bins'],
        out_mode=model_cfg['out_mode'],
        merge_mode=model_cfg['merge_mode'],
        depth=model_cfg['depth'],
        start_filters=model_cfg['start_filters'],
        mode=dataset_cfg['mode']
    ).to(device)

    # Optimizer + Loss (MSE for Mean)
    # TODO: add support for Cross Entropy for distribution
    optimizer = optim.Adam(model.parameters(), lr=float(model_cfg["learning_rate"]), weight_decay=float(model_cfg["weight_decay"]))
    criterion = nn.MSELoss()
    
    # Model Name
    date_str = datetime.now().strftime("%Y-%m-%d")
    model_type = "hist2noise" if dataset_cfg["mode"] == "hist" else "noise2noise"
    out_mode = model_cfg["out_mode"]
    bins = dataset_cfg["hist_bins"] if dataset_cfg["mode"] == "hist" else "img"
    filename = f"{date_str}_{model_type}_{out_mode}_bins{bins}.pth"

    save_dir = config.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    best_val_loss = float('inf')
    print("TRAINING STARTED !")

    for epoch in range(config["num_epochs"]):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time
        print(f"[Epoch {epoch+1}/{config['num_epochs']}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| Time: {epoch_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")


def evaluate_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Read evaluation parameters from config
    idx = config["eval"]["idx"]
    seed = config["eval"]["seed"]

    # Dataset config
    dataset_cfg = config["dataset"]

    # Load datasets using config values
    hist_dataset = HistogramBinomDataset(**{**dataset_cfg, "mode": "hist"})
    img_dataset = HistogramBinomDataset(**{**dataset_cfg, "mode": "img"})

    # Get samples
    hist_sample = hist_dataset.__getitem__(idx, seed=seed)
    crop_coords = hist_sample["crop_coords"]
    img_sample = img_dataset.__getitem__(idx, seed=seed, crop_coords=crop_coords)

    # Prepare inputs for model
    hist_input = hist_sample["input"].unsqueeze(0).to(device)
    img_input = img_sample["input"].unsqueeze(0).to(device)
    target = hist_sample["target"].to(device)
    noisy = hist_sample["noisy"].to(device)
    clean = hist_sample.get("clean", None)
    if clean is not None:
        clean = clean.to(device)
    scene = hist_sample["scene"]

    # Load models from checkpoint paths in config
    hist_model = load_model(config["eval"]["hist_checkpoint"], mode="hist", device=device)
    img_model = load_model(config["eval"]["img_checkpoint"], mode="img", device=device)

    # Evaluate models
    hist_pred, hist_psnr = evaluate_sample(hist_model, hist_input, clean)
    img_pred, img_psnr = evaluate_sample(img_model, img_input, clean)

    print(f"\nScene: {scene}")
    print(f"Hist2Noise PSNR:  {hist_psnr:.2f} dB")
    print(f"Noise2Noise PSNR: {img_psnr:.2f} dB")

    # Plot results
    plot_images(noisy, hist_pred, img_pred, target, clean=clean)