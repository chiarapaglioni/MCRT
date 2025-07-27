# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# Path
import os
import random
from pathlib import Path
# Time
import time
from datetime import datetime
# Custom
from model.UNet import UNet
from model.N2NUnet import Noise2NoiseUNet, Net
from model.noise2noise import Noise2Noise
from dataset.HistImgDataset import HistogramBinomDataset
from utils.utils import plot_images, save_loss_plot, save_psnr_plot, plot_debug_images, compute_psnr, compute_global_mean_std, apply_tonemap

# Logger
import logging
logger = logging.getLogger(__name__)

# GPU Check
if not torch.cuda.is_available():
    logger.warning("GPU not found, code will run on CPU and can be extremely slow!")
else:
    device = torch.device("cuda:0")


def get_data_loaders(config, run_mode="train"):
    dataset_cfg = config['dataset'].copy()
    dataset_cfg['root_dir'] = Path(__file__).resolve().parents[1] / dataset_cfg['root_dir']

    gloab_mean, glob_std = compute_global_mean_std(dataset_cfg['root_dir'])
    logger.info(f"DATASET mean {[round(v.item(), 4) for v in gloab_mean.view(-1)]} - std {[round(v.item(), 4) for v in glob_std.view(-1)]}")

    if config['standardisation']=='global':
        full_dataset = HistogramBinomDataset(**dataset_cfg, global_mean=gloab_mean, global_std=glob_std, run_mode=run_mode)

    # Split into train/val
    val_ratio = config.get('val_split', 0.2)
    total_len = len(full_dataset)
    val_len   = int(total_len * val_ratio)
    train_len = total_len - val_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len])

    logger.info(f"Total dataset size: {total_len}")
    logger.info(f"Training set size:  {len(train_ds)}")
    logger.info(f"Validation set size: {len(val_ds)}")

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
    # model = UNet(
    #     in_channels=model_config['in_channels'],
    #     n_bins=hist_bins,
    #     out_mode=model_config['out_mode'],
    #     merge_mode=model_config['merge_mode'],
    #     depth=model_config['depth'],
    #     start_filters=model_config['start_filters'],
    #     mode=mode
    # ).to(device)

    model = Noise2NoiseUNet(
        in_channels=model_config['in_channels'],
        out_channels=3,
        features=model_config['start_filters']
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
        tonemapped_input = apply_tonemap(input_tensor, "reinhard_gamma")
        pred = model(tonemapped_input)
        clean = clean_tensor

        logger.info(f"Target shape: {clean.shape}")      # H, W, 3
        logger.info(f"Pred shape:  {pred.shape}")        # H, W, 3

        psnr_val = compute_psnr(pred.squeeze(0), clean)
    return pred, psnr_val


# TRAINING STEP
def train_epoch(model, dataloader, optimizer, criterion, device, tonemap, epoch=None, debug=True):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        hdr_input = batch['input'].to(device)               # B, 3, H, W or # B, 9, H, W 
        hdr_target = batch['target'].to(device)             # B, 3, H, W

        optimizer.zero_grad()

        # Apply tonemapping to input if tonemap != none
        tonemapped_input = apply_tonemap(hdr_input, tonemap=tonemap) 
        pred = model(tonemapped_input)                      # B, 3, H, W (HDR space)

        # DEBUG (statistics)
        if batch_idx % 10 == 0:
            # Only take RGB channels if input has more than 3 channels
            input_rgb = tonemapped_input[:, :3] if tonemapped_input.shape[1] > 3 else tonemapped_input

            logger.info(f"Input (RGB) Min {input_rgb.min():.4f} - Max {input_rgb.max():.4f} - Mean {input_rgb.mean():.4f} - Var {input_rgb.var():.4f}")
            logger.info(f"Target Min {hdr_target.min():.4f} - Max {hdr_target.max():.4f} - Mean {hdr_target.mean():.4f} - Var {hdr_target.var():.4f}")
            logger.info(f"Pred Min {pred.min():.4f} - Max {pred.max():.4f} - Mean {pred.mean():.4f} - Var {pred.var():.4f}")
            logger.info("-------------------------------------------------------------------")

        # LOSS (in tonemapped space if tonemap != none)
        loss = criterion(apply_tonemap(pred, tonemap=tonemap), apply_tonemap(hdr_target, tonemap=tonemap))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # DEBUG (plot the first batch)
        if debug and batch_idx==0 and epoch>=0:
            plot_debug_images(batch, preds=pred, epoch=epoch, batch_idx=batch_idx, correct=True)

    return total_loss / len(dataloader)


# VALIDATION STEP
def validate_epoch(model, dataloader, criterion, device, tonemap):
    model.eval()
    total_loss = 0
    total_psnr = 0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            hdr_input = batch['input'].to(device)           # B, 3, H, W
            hdr_target = batch['target'].to(device)         # B, 3, H, W
            clean = batch['clean'].to(device)               # B, 3, H, W

            # Apply tonemapping to input if tonemap != none
            tonemapped_input = apply_tonemap(hdr_input, tonemap=tonemap) 
            pred = model(tonemapped_input)                  # B, 3, H, W (HDR space)

            loss = criterion(pred, hdr_target)
            total_loss += loss.item()

            for i in range(pred.shape[0]):
                pred_i = pred[i]
                clean_i = clean[i]
                
                # PSNR in HDR space
                total_psnr += compute_psnr(pred_i, clean_i)
                count += 1

    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / count
    return avg_loss, avg_psnr


class LHDRLoss(nn.Module):
    def __init__(self, epsilon=0.01):
        super(LHDRLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, denoised, target):
        """Computes loss by unpacking render buffer."""
        loss = ((denoised - target) ** 2) / ((denoised + self.epsilon)**2)
        return torch.mean(loss.view(-1))


class RelativeMSELoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        return torch.mean((input - target)**2 / (target + self.epsilon)**2)
    

# TRAINING LOOP
def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data Loaders
    train_loader, val_loader = get_data_loaders(config, run_mode="train")

    dataset_cfg = config['dataset']
    model_cfg = config['model']

    logger.info(f"\nDataset Config: {config['dataset']['mode'].upper()} mode | Crop Size: {config['dataset']['crop_size']} | Augmentation: {config['dataset']['data_augmentation']}")
    logger.info(f"Model Config: Depth={config['model']['depth']} | Start Filters={config['model']['start_filters']} | Output: {config['model']['out_mode']}")
    logger.info(f"Training for {config['num_epochs']} epochs | Batch Size: {config['batch_size']} | Val Split: {config['val_split']} | Learning Rate: {config['model']['learning_rate']}\n")

    # MODEL
    # model = UNet(
    #     in_channels=model_cfg['in_channels'],
    #     n_bins=dataset_cfg['hist_bins'],
    #     out_mode=model_cfg['out_mode'],
    #     merge_mode=model_cfg['merge_mode'],
    #     depth=model_cfg['depth'],
    #     start_filters=model_cfg['start_filters'],
    #     mode=dataset_cfg['mode']
    # ).to(device)

    # model = Noise2NoiseUNet(
    #     in_channels=model_cfg['in_channels'],
    #     out_channels=3,
    #     features=model_cfg['start_filters']
    # ).to(device)

    model = Net(in_channels=model_cfg['in_channels']).to(device)

    # OPTIMIZER
    optimizer = optim.Adam(model.parameters(), lr=float(model_cfg["learning_rate"]))

    # LOSS FUNCTIONS 
    logger.info(f"Using Tonemap: {dataset_cfg['tonemap'].upper()}")
    logger.info(f"Using Loss: {config['loss'].upper()}")
    if config['loss']=='mse':
        criterion = nn.MSELoss()            # when input is same as output
    elif config['loss']=='rmse':
        criterion = RelativeMSELoss()       # when input tone mapping but output isn't
    elif config['loss']=='lhdr':
        criterion = LHDRLoss()              # when input tone mapping but output isn't
    elif config['loss']=='l1':
        criterion = nn.L1Loss()             # like MSE but abs value
    
    # Model Name
    date_str = datetime.now().strftime("%Y-%m-%d")
    model_type = "hist2noise" if dataset_cfg["mode"] == "hist" else "noise2noise"
    out_mode = model_cfg["out_mode"]
    total_epochs = config["num_epochs"]
    loss_func = config['loss']
    bins = dataset_cfg["hist_bins"] if dataset_cfg["mode"] == "hist" else "img"
    filename = f"{date_str}_{model_type}_{out_mode}_bins{bins}_ep{total_epochs}_{loss_func}.pth"

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

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, tonemap=dataset_cfg['tonemap'], epoch=epoch, debug=dataset_cfg['debug'])
        val_loss, val_psnr = validate_epoch(model, val_loader, criterion, device, tonemap=dataset_cfg['tonemap'])

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        psnr_values.append(val_psnr)

        epoch_time = time.time() - start_time
        logger.info(f"[Epoch {epoch+1}/{config['num_epochs']}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} dB "
            f"| Time: {epoch_time:.2f}s")

        # TODO: choose best mode based also on PSNR
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

    # Number of random samples to evaluate
    n_samples = config["eval"].get("n_eval_samples", 5)

    dataset_cfg = config["dataset"]

    hist_dataset = HistogramBinomDataset(**{**dataset_cfg, "mode": "hist"}, run_mode="test")
    img_dataset = HistogramBinomDataset(**{**dataset_cfg, "mode": "img"}, run_mode="test")

    # Randomly select n indices
    total_samples = len(img_dataset)
    selected_indices = random.sample(range(total_samples), n_samples)
    logger.info(f"Randomly selected indices: {selected_indices}")

    # Load models
    # hist_model = load_model(config['model'], config["eval"]["hist_checkpoint"], mode="hist", device=device)
    hist_model = load_model(config['model'], config["eval"]["hist_checkpoint"], mode="img", device=device)
    img_model = load_model(config['model'], config["eval"]["img_checkpoint"], mode="img", device=device)

    for idx in selected_indices:
        logger.info(f"\nEvaluating index: {idx}")

        # Get samples
        hist_sample = hist_dataset.__getitem__(idx)
        crop_coords = hist_sample["crop_coords"]
        img_sample = img_dataset.__getitem__(idx, crop_coords=crop_coords)

        # Prepare inputs
        hist_input = hist_sample["input"].unsqueeze(0).to(device)
        img_input = img_sample["input"].unsqueeze(0).to(device)
        target = hist_sample["target"].to(device)
        noisy = hist_sample["noisy"].to(device)
        clean = hist_sample.get("clean", None)
        if clean is not None:
            clean = clean.to(device)
        scene = hist_sample["scene"]

        # Evaluate models
        # hist_pred, hist_psnr = evaluate_sample(hist_model, hist_input, clean, tonemap=dataset_cfg['tonemap'])
        hist_pred, hist_psnr = evaluate_sample(hist_model, img_input, clean)
        img_pred, img_psnr = evaluate_sample(img_model, img_input, clean)
        init_psnr = compute_psnr(noisy, clean)

        logger.info(f"Scene: {scene}")
        logger.info(f"Noisy Input PSNR:  {init_psnr:.2f} dB")
        logger.info(f"Hist2Noise PSNR:  {hist_psnr:.2f} dB")
        logger.info(f"Noise2Noise PSNR: {img_psnr:.2f} dB")

        # PLOT
        plot_images(
            noisy, init_psnr, 
            hist_pred, hist_psnr, 
            img_pred, img_psnr,
            target, clean,
            save_path=f'plots/denoised_{idx}.png',
            correct=True        # whether to plot tonemapped + gamma corrected images
        )


def train_n2n(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data Loaders
    train_loader, val_loader = get_data_loaders(config)

    n2n = Noise2Noise(config, trainable=True)
    n2n.train(train_loader, val_loader)


def test_n2n(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_cfg = config["dataset"]
    img_dataset = HistogramBinomDataset(**{**dataset_cfg, "mode": "img"})
    img_loader = DataLoader(img_dataset, batch_size=1, shuffle=True,  num_workers=config['num_workers'], pin_memory=config['pin_memory'], drop_last=True)

    n2n = Noise2Noise(config, trainable=False)
    n2n.load_model(config['load_ckpt'])
    n2n.test(img_loader, show=config['show_output'])
