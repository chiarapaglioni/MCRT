import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

from model.UNet import UNet
from dataset.HistDataset import HistogramDataset
from utils.utils import save_loss_plot, decode_pred_logits, compute_psnr

import logging
logger = logging.getLogger(__name__)


def get_generative_dataloaders(config):
    dataset_cfg = config['dataset'].copy()
    dataset_cfg['root_dir'] = Path(__file__).resolve().parents[1] / dataset_cfg['root_dir']

    dataset = HistogramDataset(**dataset_cfg)

    val_ratio = config.get('val_split', 0.1)
    val_len = int(len(dataset) * val_ratio)
    train_len = len(dataset) - val_len

    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    return train_loader, val_loader


def train_generative_epoch(model, dataloader, optimizer, criterion, device, n_bins):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        input_hist = batch['input_hist'].to(device)       # (B, C, bins, H, W)
        target_bins = batch['target_hist'].to(device).long()  # (B, 3, H, W)

        optimizer.zero_grad()
        pred_logits = model(input_hist)                   # (B, 3, n_bins, H, W)

        # Convert target from histogram to class labels
        target_bins = torch.argmax(target_bins, dim=2)    # (B, 3, H, W)

        # Rearrange and flatten predictions
        pred_logits = pred_logits.permute(0, 1, 3, 4, 2).contiguous()  # (B, 3, H, W, n_bins)
        pred_logits = pred_logits.view(-1, n_bins)                     # (B*3*H*W, n_bins)
        target_bins = target_bins.view(-1)                             # (B*3*H*W,)

        loss = criterion(pred_logits, target_bins)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate_generative_epoch(model, dataloader, criterion, device, n_bins):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_hist = batch['input_hist'].to(device)
            target_hist = batch['target_hist'].to(device)

            # Model prediction
            pred_logits = model(input_hist)  # (B, 3, n_bins, H, W)

            # Prepare target bin indices for CrossEntropyLoss
            # We assume target_hist is a full histogram -> pick the most probable bin
            target_bins = torch.argmax(target_hist, dim=2).long()  # (B, 3, H, W)

            # Reshape for loss: CrossEntropy expects (N, C) and (N,)
            pred_logits_reshaped = pred_logits.permute(0, 1, 3, 4, 2).reshape(-1, n_bins)  # [B * 3 * H * W, n_bins]
            target_bins_reshaped = target_bins.view(-1)                                    # [B * 3 * H * W]

            # Compute loss
            loss = criterion(pred_logits_reshaped, target_bins_reshaped)
            total_loss += loss.item()

            # Decode to RGB for PSNR
            probs = torch.softmax(pred_logits, dim=2)  # (B, 3, n_bins, H, W)
            pred_rgb = decode_pred_logits(probs).clamp(0, 1)  # (B, 3, H, W)

            # Use soft version of target as distribution
            target_rgb = decode_pred_logits(target_hist).clamp(0, 1)

            # PSNR per image
            for i in range(pred_rgb.size(0)):
                psnr_i = psnr(pred_rgb[i].cpu().numpy(), target_rgb[i].cpu().numpy(), data_range=1.0)
                total_psnr += psnr_i
                count += 1

    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / count if count > 0 else 0.0
    return avg_loss, avg_psnr



def train_histogram_generator(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    train_loader, val_loader = get_generative_dataloaders(config)
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    n_bins = dataset_cfg["hist_bins"]

    model = UNet(
        in_channels=model_cfg["in_channels"],
        n_bins=n_bins,
        out_mode=model_cfg["out_mode"],
        merge_mode=model_cfg["merge_mode"],
        depth=model_cfg["depth"],
        start_filters=model_cfg["start_filters"],
        mode=dataset_cfg["mode"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(model_cfg["learning_rate"]))
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    model_name = f"{datetime.now().strftime('%Y%m%d')}_hist2hist_{model_cfg['out_mode']}.pth"
    save_path = Path(config["save_dir"]) / model_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training...")

    for epoch in range(config["num_epochs"]):
        start_time = time.time()

        train_loss = train_generative_epoch(model, train_loader, optimizer, criterion, device, n_bins)
        val_loss, val_psnr = validate_generative_epoch(model, val_loader, criterion, device, n_bins)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"[Epoch {epoch+1}/{config['num_epochs']}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f} dB | Time: {time.time() - start_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")

    save_loss_plot(train_losses, val_losses, save_dir="plots", filename=f"{model_name}_hist2hist_loss_plot.png")


def iterative_evaluate(config, num_samples=4, num_steps=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Generating on {device}")

    dataset_cfg = config["dataset"]
    n_bins = dataset_cfg["hist_bins"]

    dataset = HistogramDataset(**dataset_cfg)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=config['num_workers'], shuffle=False)

    model_cfg = config["model"]
    model = UNet(
        in_channels=model_cfg["in_channels"],
        n_bins=n_bins,
        out_mode=model_cfg["out_mode"],
        merge_mode=model_cfg["merge_mode"],
        depth=model_cfg["depth"],
        start_filters=model_cfg["start_filters"],
        mode=dataset_cfg["mode"]
    ).to(device)

    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.eval()

    os.makedirs("plots", exist_ok=True)

    for idx, batch in enumerate(dataloader):
        if idx >= num_samples:
            break

        input_hist = batch["input_hist"].to(device)  # shape: (1, C, bins, H, W)
        clean_img = batch["clean"].to(device)        # shape: (1, 3, H, W)

        B, C, _, H, W = input_hist.shape
        accumulated_radiance = torch.zeros((B, C, H, W), device=device)

        _, axs = plt.subplots(1, num_steps + 2, figsize=(4 * (num_steps + 2), 4))

        for step in range(num_steps):
            logits = model(input_hist)
            probs = torch.softmax(logits, dim=2)
            pred_radiance = decode_pred_logits(probs).clamp(0, 1)

            accumulated_radiance += (pred_radiance / num_steps)

            psnr = compute_psnr(accumulated_radiance[0], clean_img[0])
            axs[step].imshow(accumulated_radiance[0].permute(1, 2, 0).detach().cpu().numpy())
            axs[step].set_title(f"Step {step+1}\nPSNR: {psnr:.2f} dB")
            axs[step].axis("off")

        # Final Output
        final_psnr = compute_psnr(accumulated_radiance[0], clean_img[0])
        axs[num_steps].imshow(accumulated_radiance[0].permute(1, 2, 0).detach().cpu().numpy())
        axs[num_steps].set_title(f"Final Output\nPSNR: {final_psnr:.2f} dB")
        axs[num_steps].axis("off")

        # Ground Truth
        axs[num_steps + 1].imshow(clean_img[0].permute(1, 2, 0).detach().cpu().numpy())
        axs[num_steps + 1].set_title("Ground Truth")
        axs[num_steps + 1].axis("off")

        plt.tight_layout()
        plot_path = os.path.join("plots", f"sample_{idx}_evaluation.png")
        plt.savefig(plot_path)
        plt.show()
        print(f"Saved plot to {plot_path}")
