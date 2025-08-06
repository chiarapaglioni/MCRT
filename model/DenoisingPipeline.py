# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
# Path
import os
import random
from pathlib import Path
# Time
import time
from datetime import datetime
# Custom
import numpy as np
from model.UNet import GapUNet
from model.N2NUnet import N2Net
from dataset.HistImgDataset import ImageDataset, HistogramDataset
from dataset.HistImgPatchAggregator import PatchAggregator
from utils.utils import load_model, plot_images, save_loss_plot, save_psnr_plot, plot_debug_images, compute_psnr, compute_global_mean_std, apply_tonemap, plot_debug_aggregation, plot_aggregation_analysis

# Logger
import logging
logger = logging.getLogger(__name__)

# GPU Check
if not torch.cuda.is_available():
    logger.warning("GPU not found, code will run on CPU and can be extremely slow!")
else:
    device = torch.device("cuda:0")



# LOSS FUNCTIONS
class LHDRLoss(nn.Module):
    def __init__(self, epsilon=0.01):
        super(LHDRLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, denoised, target):
        """Computes loss by unpacking render buffer."""
        loss = ((denoised - target) ** 2) / (denoised + self.epsilon)**2
        return torch.mean(loss.view(-1))

class RelativeMSELoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        return torch.mean((input - target)**2 / (target + self.epsilon)**2)
    
class RootMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.sqrt(torch.mean((input - target) ** 2))
    
class SMAPELoss(nn.Module):
    def __init__(self, epsilon=1e-2):
        super(SMAPELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        """
        prediction, target: tensors of shape [batch_size, channels, height, width]
        Expected channels = 3 (RGB)
        """
        numerator = torch.abs(prediction - target)
        denominator = torch.abs(prediction) + torch.abs(target) + self.epsilon
        
        # Calculate per-pixel, per-channel SMAPE
        smape_map = numerator / denominator
        
        # Average over pixels, channels, and batch
        return smape_map.mean()


def get_data_loaders(config, run_mode="train"):
    dataset_cfg = config['dataset'].copy()
    dataset_cfg['root_dir'] = Path(__file__).resolve().parents[1] / dataset_cfg['root_dir']

    gloab_mean, glob_std = compute_global_mean_std(dataset_cfg['root_dir'])
    logger.info(f"DATASET mean {[round(v.item(), 4) for v in gloab_mean.view(-1)]} - std {[round(v.item(), 4) for v in glob_std.view(-1)]}")

    if dataset_cfg['mode']=='img' or dataset_cfg['mode']=='stat':
        full_dataset = ImageDataset(**dataset_cfg, run_mode=run_mode)
    elif dataset_cfg['mode']=='hist':
        full_dataset = HistogramDataset(**dataset_cfg, run_mode=run_mode)

    total_len = len(full_dataset)
    val_ratio = config.get('val_split', 0.1)
    val_len   = int(total_len * val_ratio)
    train_len = total_len - val_len

    # Generate shuffled indices
    indices = np.random.permutation(total_len)
    train_indices = indices[:train_len].tolist()
    val_indices   = indices[train_len:].tolist()

    # Subset the dataset
    train_ds = Subset(full_dataset, train_indices)
    val_ds   = Subset(full_dataset, val_indices)

    logger.info(f"Total dataset size: {total_len}")
    logger.info(f"Training set size:  {train_len}")
    logger.info(f"Validation set size: {val_len}")

    # Get full patch weights and extract only the ones for train_ds
    weights_full = full_dataset.get_sampling_weights()
    train_weights = weights_full[train_indices]
    val_weights = weights_full[val_indices]

    train_sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    val_sampler = WeightedRandomSampler(val_weights, len(val_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        # sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        # sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=False,
    )

    # Log one batch to confirm shapes
    sample = next(iter(train_loader))
    logger.info(f"Input shape:  {sample['input'].shape}")
    logger.info(f"Target shape: {sample['target'].shape}")
    logger.info(f"Noisy shape:  {sample['noisy'].shape}")
    if 'clean' in sample and sample['clean'] is not None:
        logger.info(f"Clean shape:  {sample['clean'].shape}")
    return train_loader, val_loader


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
        pred = model(input_tensor)
        clean = clean_tensor

        logger.info(f"Target shape: {clean.shape}")      # H, W, 3
        logger.info(f"Pred shape:  {pred.shape}")        # H, W, 3

        psnr_val = compute_psnr(pred.squeeze(0), clean)
    return pred, psnr_val


# TRAINING STEP
def train_epoch(model, dataloader, optimizer, criterion, device, tonemap, mode, n_bins, epoch=None, debug=True, plot_every_n=10):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        hdr_input = batch['input'].to(device)               # B, 3, H, W (img) or # B, 9, H, W (aov) or # B, 10, H, W (aov + stat)
        hdr_target = batch['target'].to(device)             # B, 3, H, W

        optimizer.zero_grad()

        if mode == "hist":
            x_hist = hdr_input[:, :3*n_bins]                        # (B, 48, H, W)
            x_spatial = hdr_input[:, 3*n_bins:, :, :]               # (B, 6, H, W)
            pred = model(x_spatial, x_hist)                         # B, 3, H, W (HDR space)
        else:
            pred = model(hdr_input)                                 # B, 3, H, W (HDR space)

        # DEBUG (statistics)
        if batch_idx % 50 == 0:
            # Only take RGB channels if input has more than 3 channels
            input_rgb = hdr_input[:, :3] if hdr_input.shape[1] > 3 else hdr_input
            pred_tonamepped = apply_tonemap(pred, tonemap=tonemap),

            logger.info(f"Input (RGB) Min {input_rgb.min():.4f} - Max {input_rgb.max():.4f} - Mean {input_rgb.mean():.4f} - Var {input_rgb.var():.4f}")
            logger.info(f"Target Min {hdr_target.min():.4f} - Max {hdr_target.max():.4f} - Mean {hdr_target.mean():.4f} - Var {hdr_target.var():.4f}")
            logger.info(f"Pred Min {pred_tonamepped[0].min():.4f} - Max {pred_tonamepped[0].max():.4f} - Mean {pred_tonamepped[0].mean():.4f} - Var {pred_tonamepped[0].var():.4f}")
            logger.info("-------------------------------------------------------------------")

        # LOSS (in tonemapped space if tonemap != none)
        loss = criterion(apply_tonemap(pred, tonemap=tonemap), apply_tonemap(hdr_target, tonemap=tonemap))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # DEBUG (plot the first batch)
        if debug and batch_idx == 0 and epoch is not None and (epoch % plot_every_n) == 0:
            random_index = random.randint(0, pred.shape[0] - 1)  # pred.shape[0] == batch size
            plot_debug_images(
                {key: val[random_index:random_index+1] for key, val in batch.items()},  # single-item batch
                preds=pred_tonamepped[0][random_index:random_index+1],
                epoch=epoch,
                batch_idx=batch_idx,
                correct=True
            )

    return total_loss / len(dataloader)


# VALIDATION STEP
def validate_epoch(model, dataloader, criterion, device, tonemap, mode, n_bins, epoch, plot_every_n, debug=False):
    model.eval()
    total_loss = 0
    total_psnr = 0
    count = 0

    aggregator = PatchAggregator(kernel_size=7, sigma_color=0.1)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            hdr_input = batch['input'].to(device)           # B, 3, H, W
            hdr_target = batch['target'].to(device)         # B, 3, H, W
            clean = batch['clean'].to(device)               # B, 3, H, W

            if mode == "hist":
                x_hist = hdr_input[:, :3*n_bins]                    # (B, 48, H, W)
                x_spatial = hdr_input[:, 3*n_bins:, :, :]           # (B, 6, H, W)                  

                # PREDICTION
                pre_agg_pred = model(x_spatial, x_hist)         # B, 3, H, W (HDR space)
                
                # AGGREGATOR
                pred = aggregator(output=pre_agg_pred, features=[(x_hist, 0.3)])                      # hist only
                # pred_img  = aggregator(output=pre_agg_pred, features=[(x_spatial, 0.2)])                 # img only
                # pred = aggregator(output=pre_agg_pred, features=[(x_spatial, 0.2), (x_hist, 0.3)])  # hybrid
            else:
                pre_agg_pred = model(hdr_input)                             # B, 3, H, W (HDR space)
                pred  = aggregator(output=pre_agg_pred, features=[(hdr_input, 0.2)]) 

            loss = criterion(apply_tonemap(pred, tonemap=tonemap), apply_tonemap(hdr_target, tonemap=tonemap))
            total_loss += loss.item()

            # PSNR
            for i in range(pred.shape[0]):
                pred_i_hdr = pred[i]
                clean_i_hdr = clean[i]

                total_psnr += compute_psnr(apply_tonemap(pred_i_hdr, tonemap=tonemap), apply_tonemap(clean_i_hdr, tonemap=tonemap))
                count += 1

            # DEBUG (plot the first batch)
            if debug and batch_idx == 0 and epoch is not None and (epoch % plot_every_n) == 0:
                start = 3 * n_bins
                end = start + 3
                input_rgb = hdr_input[:, :3] if hdr_input.shape[1] <= 10 else hdr_input[:, start:end]  # shape: [10, 3, 128, 128]
                # plot_debug_aggregation(pre_agg_pred, pred, input_rgb, clean, epoch)
                
                # randomly pick one index from the batch
                # random_idx = random.randint(0, pred.shape[0] - 1)
                # plot_aggregation_analysis(pre_agg_pred, pred, pred_img, clean, epoch, idx=random_idx)

    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / count
    return avg_loss, avg_psnr


# TRAINING LOOP
def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # DATA LOADERS
    train_loader, val_loader = get_data_loaders(config, run_mode="train")

    # CONFIG
    dataset_cfg = config['dataset']
    model_cfg = config['model']

    logger.info(f"\nDataset Config: {config['dataset']['mode'].upper()} mode | Crop Size: {config['dataset']['crop_size']} | Augmentation: {config['dataset']['data_augmentation']}")
    logger.info(f"Model Config: Depth={config['model']['depth']} | Start Filters={config['model']['start_filters']} | Output: {config['model']['out_mode']}")
    logger.info(f"Training for {config['num_epochs']} epochs | Batch Size: {config['batch_size']} | Val Split: {config['val_split']} | Learning Rate: {config['model']['learning_rate']}\n")

    # MODEL
    if model_cfg['model_name'] == 'gap':
        model = GapUNet(
            in_channels=model_cfg["in_channels"],
            n_bins_input=model_cfg["n_bins_input"],
            n_bins_output=dataset_cfg['hist_bins'],
            out_mode=model_cfg["out_mode"],
            merge_mode=model_cfg["merge_mode"],
            depth=model_cfg["depth"],
            start_filters=model_cfg["start_filters"],
            mode=dataset_cfg["mode"]
        ).to(device)

    elif model_cfg['model_name'] == 'n2n':
        if dataset_cfg["mode"] == "hist":
            model = N2Net(
                in_channels=model_cfg["in_channels"],       # total channels: histogram + spatial
                hist_bins=dataset_cfg["hist_bins"],         # how many bins per channel
                out_mode=model_cfg['out_mode'],
                mode="hist"
            ).to(device)
        else:  # "img" mode
            model = N2Net(
                in_channels=model_cfg["in_channels"],       # e.g., 3 or 9 channels (mean + AOV + variance)
                out_mode=model_cfg['out_mode'],
                mode="img"
            ).to(device)

    # OPTIMIZER
    optimizer = optim.Adam(model.parameters(), lr=float(model_cfg["learning_rate"]))

    # LR SCHEDULER
    # reduces learning rate if val loss has not decreased in the last 10 epochs
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # LOSS FUNCTIONS 
    logger.info(f"Input Tonemap: {dataset_cfg['input_tonemap'].upper()}")
    logger.info(f"Loss Tonemap: {dataset_cfg['tonemap'].upper()}")
    logger.info(f"Using Loss: {config['loss'].upper()}")
    if config['loss']=='mse':
        criterion = nn.MSELoss()            # when input is same as output
    elif config['loss']=='rel_mse':
        criterion = RelativeMSELoss()       # when input tone mapping but output isn't
    elif config['loss']=='root_mse':
        criterion = RootMSELoss()       # when input tone mapping but output isn't
    elif config['loss']=='lhdr':
        criterion = LHDRLoss()              # when input tone mapping but output isn't
    elif config['loss']=='l1':
        criterion = nn.L1Loss()             # like MSE but abs value
    elif config['loss']=='smape':
        criterion = SMAPELoss()             # supervised learning (Pixar paper)
    
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

    train_losses = []
    val_losses = []
    psnr_values = []

    for epoch in range(config["num_epochs"]):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, tonemap=dataset_cfg['tonemap'], mode=dataset_cfg["mode"], n_bins=dataset_cfg["hist_bins"], epoch=epoch, debug=dataset_cfg['debug'], plot_every_n=config['plot_every'])
        val_loss, val_psnr = validate_epoch(model, val_loader, criterion, device, tonemap=dataset_cfg['tonemap'], mode=dataset_cfg["mode"], n_bins=dataset_cfg["hist_bins"], epoch=epoch, debug=dataset_cfg['debug'], plot_every_n=config['plot_every'])

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        psnr_values.append(val_psnr)

        scheduler.step(val_loss)

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


# EVALUATE MODELS
def evaluate_model(config):
    """
    Evaluates model using using histograms vs. images
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on device: {device}")

    # Number of random samples to evaluate
    n_samples = config["eval"].get("n_eval_samples", 5)

    dataset_cfg = config["dataset"]
    model_cfg = config['model']

    # TODO: implement smart selection of the datsets
    hist_dataset = HistogramDataset(**{**dataset_cfg, "mode": "hist"}, run_mode="test")
    img_dataset = ImageDataset(**{**dataset_cfg, "mode": "img"}, run_mode="test")
    # img_dataset = ImageDataset(**{**dataset_cfg, "mode": "stat"}, run_mode="test")

    # Randomly select n indices
    total_samples = len(img_dataset)
    selected_indices = random.sample(range(total_samples), n_samples)
    logger.info(f"Randomly selected indices: {selected_indices}")

    # Load models
    hist_model = load_model(model_cfg, config["eval"]["hist_checkpoint"], mode="hist", device=device)
    img_model = load_model(model_cfg, config["eval"]["img_checkpoint"], mode="img", device=device)
    # img_model = load_model(model_cfg, config["eval"]["img_checkpoint"], mode="stat", device=device)

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
        hist_pred, hist_psnr = evaluate_sample(hist_model, hist_input, clean)
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


def evaluate_model_aov(config):
    """
    Evaluates model using img denoising with and without AOVs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on device: {device}")

    # Number of random samples to evaluate
    n_samples = config["eval"].get("n_eval_samples", 5)

    dataset_cfg = config["dataset"]
    model_cfg = config['model']

    aov_dataset = ImageDataset(**{**dataset_cfg, "aov": True}, run_mode="test")
    img_dataset = ImageDataset(**{**dataset_cfg, "aov": False}, run_mode="test")

    # Randomly select n indices
    total_samples = len(img_dataset)
    selected_indices = random.sample(range(total_samples), n_samples)
    logger.info(f"Randomly selected indices: {selected_indices}")

    # Load models
    model_cfg["in_channels"] = 9
    aov_model = load_model(model_cfg, config["eval"]["hist_checkpoint"], mode="img", device=device)
    model_cfg["in_channels"] = 3
    img_model = load_model(model_cfg, config["eval"]["img_checkpoint"], mode="img", device=device)
    # model_cfg["in_channels"] = 10
    # stat_model = load_model(model_cfg, config["eval"]["img_checkpoint"], mode="stat", device=device)

    for idx in selected_indices:
        logger.info(f"\nEvaluating index: {idx}")

        # Get samples
        aov_sample = aov_dataset.__getitem__(idx)
        crop_coords = aov_sample["crop_coords"]
        img_sample = img_dataset.__getitem__(idx) # crop_coords=crop_coords)

        # Prepare inputs
        aov_input = aov_sample["input"].unsqueeze(0).to(device)
        img_input = img_sample["input"].unsqueeze(0).to(device)
        target = aov_sample["target"].to(device)
        noisy = aov_sample["noisy"].to(device)
        clean = aov_sample.get("clean", None)
        if clean is not None:
            clean = clean.to(device)
        scene = aov_sample["scene"]

        # Evaluate models
        aov_pred, aov_psnr = evaluate_sample(aov_model, aov_input, clean)
        img_pred, img_psnr = evaluate_sample(img_model, img_input, clean)
        init_psnr = compute_psnr(noisy, clean)

        logger.info(f"Scene: {scene}")
        logger.info(f"Noisy Input PSNR:  {init_psnr:.2f} dB")
        logger.info(f"AOV PSNR:  {aov_psnr:.2f} dB")
        logger.info(f"No AOV PSNR: {img_psnr:.2f} dB")

        # PLOT
        plot_images(
            noisy, init_psnr, 
            aov_pred, aov_psnr, 
            img_pred, img_psnr,
            target, clean,
            save_path=f'plots/denoised_{idx}.png',
            correct=True        # whether to plot tonemapped + gamma corrected images
        )


def benchmark_num_workers(config, batch_size=32, max_workers=8):
    """
    Test the speed of different num_works on GPU/CPU to find the optimal number
    """
    dataset_cfg = config['dataset'].copy()
    dataset_cfg['root_dir'] = Path(__file__).resolve().parents[1] / dataset_cfg['root_dir']

    gloab_mean, glob_std = compute_global_mean_std(dataset_cfg['root_dir'])
    logger.info(f"DATASET mean {[round(v.item(), 4) for v in gloab_mean.view(-1)]} - std {[round(v.item(), 4) for v in glob_std.view(-1)]}")

    if dataset_cfg['mode']=='img' or dataset_cfg['mode']=='stat':
        full_dataset = ImageDataset(**dataset_cfg, run_mode="test")
    elif dataset_cfg['mode']=='hist':
        full_dataset = HistogramDataset(**dataset_cfg, global_mean=gloab_mean, global_std=glob_std, run_mode="test")

    logger.info(f"Total dataset size: {len(full_dataset)}")

    for workers in range(0, max_workers + 1, 2):
        loader = DataLoader(full_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)
        start = time.time()
        for i, batch in enumerate(loader):
            if i >= 50:  # test first 50 batches
                break
        duration = time.time() - start
        print(f'num_workers={workers} took {duration:.2f} seconds for 50 batches')