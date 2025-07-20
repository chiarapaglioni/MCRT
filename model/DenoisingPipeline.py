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
from utils.utils import plot_images, save_loss_plot, save_psnr_plot, plot_debug_images, compute_psnr, unstandardize_tensor

# Logger
import logging
logger = logging.getLogger(__name__)

# GPU Check
if not torch.cuda.is_available():
    logger.warning("GPU not found, code will run on CPU and can be extremely slow!")
else:
    device = torch.device("cuda:0")


def get_data_loaders(config):
    dataset_cfg = config['dataset'].copy()
    dataset_cfg['root_dir'] = Path(__file__).resolve().parents[1] / dataset_cfg['root_dir']

    full_dataset = HistogramBinomDataset(**dataset_cfg)

    # Split into train/val
    val_ratio = config.get('val_split', 0.2)
    total_len = len(full_dataset)
    val_len   = int(total_len * val_ratio)
    train_len = total_len - val_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,  num_workers=config['num_workers'], pin_memory=config['pin_memory'], drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=config['pin_memory'], drop_last=False)

    # Log one batch to confirm shapes
    sample = next(iter(train_loader))
    logger.info(f"Input shape:  {sample['input'].shape}")
    logger.info(f"Target shape: {sample['target'].shape}")
    logger.info(f"Noisy shape:  {sample['noisy'].shape}")
    if 'clean' in sample and sample['clean'] is not None:
        logger.info(f"Clean shape:  {sample['clean'].shape}")

    return train_loader, val_loader


def load_model(model_config, model_path, mode, hist_bins=16, device='cpu'):
    model = UNet(
        in_channels=model_config['in_channels'],
        n_bins=hist_bins,
        out_mode=model_config['out_mode'],
        merge_mode=model_config['merge_mode'],
        depth=model_config['depth'],
        start_filters=model_config['start_filters'],
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


def evaluate_sample(model, input_tensor, clean_tensor, image_mean, image_std):
    with torch.no_grad():
        pred = model(input_tensor).squeeze(0).clamp(0, 1)
        clean = clean_tensor.clamp(0, 1)

        assert pred.dim() == 3 and clean.dim() == 3

        logger.info(f"Target shape: {clean.shape}")
        logger.info(f"Pred shape:  {pred.shape}")
        # Calculate PSNR
        pred_real = unstandardize_tensor(pred, mean=image_mean, std=image_std).clamp(0, 1)       # H, W, 3
        psnr_val = compute_psnr(pred_real, clean)
    return pred_real, psnr_val


# TRAINING STEP
def train_epoch(model, dataloader, optimizer, criterion, device, epoch=None, debug=True):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        hist = batch['input'].to(device)        # B, 3, H, W
        target = batch['target'].to(device)     # 3, H, W

        optimizer.zero_grad()
        pred = model(hist)                      # 3, H, W

        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Plot the first batch in the first epoch for debugging
        if debug:
            plot_debug_images(batch, preds=pred, epoch=epoch, batch_idx=batch_idx, image_mean=batch['image_mean'][batch_idx], image_std=batch['image_std'][batch_idx])

    return total_loss / len(dataloader)


# VALIDATION STEP
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_psnr = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            hist = batch['input'].to(device)            # B, 3, H, W
            target = batch['target'].to(device)         # B, 3, H, W
            clean = batch['clean'].to(device)           # B, 3, H, W
            image_mean = batch['image_mean'].to(device) 
            image_std = batch['image_std'].to(device) 

            pred = model(hist)                          # B, 3, H, W
            loss = criterion(pred, target)
            total_loss += loss.item()

            for i in range(pred.shape[0]):
                # Unstandardize prediction
                pred_i = unstandardize_tensor(pred[i], image_mean[i], image_std[i]).clamp(0, 1)       # H, W, 3
                clean_i = clean[i].clamp(0, 1)

                total_psnr += compute_psnr(pred_i, clean_i)
                count += 1

    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / count
    return avg_loss, avg_psnr



# TRAINING LOOP
def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data Loaders
    train_loader, val_loader = get_data_loaders(config)

    dataset_cfg = config['dataset']
    model_cfg = config['model']

    logger.info(f"\nDataset Config: {config['dataset']['mode'].upper()} mode | Crop Size: {config['dataset']['crop_size']} | Augmentation: {config['dataset']['data_augmentation']}")
    logger.info(f"Model Config: Depth={config['model']['depth']} | Start Filters={config['model']['start_filters']} | Output: {config['model']['out_mode']}")
    logger.info(f"Training for {config['num_epochs']} epochs | Batch Size: {config['batch_size']} | Val Split: {config['val_split']} | Learning Rate: {config['model']['learning_rate']}\n")

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
    optimizer = optim.Adam(model.parameters(), lr=float(model_cfg["learning_rate"]))
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # TODO: try combined loss MSE * 0.5 + L1 * 0.5
    
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
    logger.info("TRAINING STARTED !")

    # store loss values for plot
    train_losses = []
    val_losses = []
    psnr_values = []

    for epoch in range(config["num_epochs"]):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, debug=dataset_cfg['debug'])
        val_loss, val_psnr = validate_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        psnr_values.append(val_psnr)

        epoch_time = time.time() - start_time
        logger.info(f"[Epoch {epoch+1}/{config['num_epochs']}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} dB "
            f"| Time: {epoch_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")
        
    # save plot loss
    save_loss_plot(train_losses, val_losses, save_dir="plots", filename=f"{date_str}_{dataset_cfg['mode']}_loss_plot.png")
    save_psnr_plot(psnr_values, save_dir="plots", filename=f"{date_str}_{dataset_cfg['mode']}_psnr_plot.png")



def evaluate_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on device: {device}")

    # Read evaluation parameters from config
    idx = config["eval"]["idx"]

    # Dataset config
    dataset_cfg = config["dataset"]

    # Load datasets using config values
    hist_dataset = HistogramBinomDataset(**{**dataset_cfg, "mode": "hist"})
    img_dataset = HistogramBinomDataset(**{**dataset_cfg, "mode": "img"}, hist_regeneration=False)

    # Get samples
    hist_sample = hist_dataset.__getitem__(idx)
    crop_coords = hist_sample["crop_coords"]
    img_sample = img_dataset.__getitem__(idx, crop_coords=crop_coords)

    # Prepare inputs for model
    hist_input = hist_sample["input"].unsqueeze(0).to(device)
    img_input = img_sample["input"].unsqueeze(0).to(device)
    target = hist_sample["target"].to(device)
    noisy = hist_sample["noisy"].to(device)
    clean = hist_sample.get("clean", None)
    if clean is not None:
        clean = clean.to(device)
    scene = hist_sample["scene"]
    mean = hist_sample["image_mean"]
    std = hist_sample["image_std"]

    # Load models from checkpoint paths in config
    hist_model = load_model(config['model'], config["eval"]["hist_checkpoint"], mode="hist", device=device)
    img_model = load_model(config['model'], config["eval"]["img_checkpoint"], mode="img", device=device)

    # Evaluate models
    hist_pred, hist_psnr = evaluate_sample(hist_model, hist_input, clean, image_mean=mean, image_std=std)
    img_pred, img_psnr = evaluate_sample(img_model, img_input, clean, image_mean=mean, image_std=std)
    init_psnr = compute_psnr(noisy, clean)

    logger.info(f"\nScene: {scene}")
    logger.info(f"Noisy Input PSNR:  {init_psnr:.2f} dB")
    logger.info(f"Hist2Noise PSNR:  {hist_psnr:.2f} dB")
    logger.info(f"Noise2Noise PSNR: {img_psnr:.2f} dB")

    # Plot results
    plot_images(noisy, init_psnr, hist_pred, hist_psnr, img_pred, img_psnr, target, clean, save_path=f'plots/denoised_{idx}.png')