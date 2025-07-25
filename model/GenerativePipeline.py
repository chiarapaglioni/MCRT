# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# Utils
import os
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
# Custom
from model.UNet import UNet
from dataset.HistDataset import HistogramDataset
from utils.utils import save_loss_plot, decode_image_from_probs, compute_psnr, save_psnr_plot

import logging
logger = logging.getLogger(__name__)


# DATA LOADERS
def get_generative_dataloaders(config):
    dataset_cfg = config['dataset'].copy()
    dataset_cfg['root_dir'] = Path(__file__).resolve().parents[1] / dataset_cfg['root_dir']

    dataset = HistogramDataset(**dataset_cfg)

    val_ratio = config.get('val_split', 0.1)
    val_len = int(len(dataset) * val_ratio)
    train_len = len(dataset) - val_len

    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])

    logger.info(f"Training set size:  {train_len}")
    logger.info(f"Validation set size: {val_len}")

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    return train_loader, val_loader


# TRAINING STEP - RESIDUAL
def train_residual_epoch(model, loss_fn, dataloader, optimizer, device, n_bins, epoch):
    model.train()
    running_loss = 0.0
    i = 0

    for batch in dataloader:
        input_hist = batch['input_hist'].to(device)                     # (B, C, bins, H, W)
        target_hist = batch['target_hist'].to(device) - batch['input_hist'].to(device) # residual                  # (B, C, bins, H, W), already normalized
        bin_edges = batch['bin_edges'].to(device)                       # (B, bins+1)

        optimizer.zero_grad()
        pred_logits = model(input_hist)                                 # (B, C, n_bins, H, W)

        # Rearrange predictions and targets for loss
        pred_logits = pred_logits.permute(0, 1, 3, 4, 2).contiguous()   # (B, C, H, W, n_bins)
        target_hist = target_hist.permute(0, 1, 3, 4, 2).contiguous()   # (B, C, H, W, n_bins)

        # Flatten to (N, n_bins) where N = B*C*H*W
        pred_logits_flat = pred_logits.view(-1, n_bins)                 # (B*C*H*W, n_bins)
        target_hist_flat = target_hist.view(-1, n_bins)                 # (B*C*H*W, n_bins)

        loss = loss_fn(pred_logits_flat, target_hist_flat)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if epoch>=100:
            plot_input_vs_prediction(
                input_hist=input_hist,
                residual=pred_logits.permute(0, 1, 4, 2, 3).detach().cpu(),  # (B, C, bins, H, W)
                bin_edges=bin_edges.detach().cpu(),
                device=device,
                step_info=f"Epoch, Batch {i}",
                clean_img=batch["clean"] if "clean" in batch else None
            )
        i += 1

    return running_loss / len(dataloader)


# VALIDATION STEP - RESIDUAL
def validate_residual_epoch(model, loss_fn, dataloader, device, n_bins):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_hist = batch['input_hist'].to(device)                     # (B, C, bins, H, W)
            target_hist = batch['target_hist'].to(device) - input_hist      # residual histogram (already normalized)
            clean_img = batch['clean'].to(device)                           # (B, C, H, W)
            bin_edges = batch['bin_edges'].to(device)

            # Forward pass
            pred_logits = model(input_hist)                                 # (B, C, bins, H, W)

            # Prepare shapes for loss
            pred_logits_flat = pred_logits.permute(0, 1, 3, 4, 2).contiguous().view(-1, n_bins)  # (N, bins)
            target_hist_flat = target_hist.permute(0, 1, 3, 4, 2).contiguous().view(-1, n_bins)  # (N, bins)

            # Apply softmax to logits for MSE
            pred_probs_flat = torch.softmax(pred_logits_flat, dim=1)        # (N, bins)

            # Compute MSE loss between predicted residual probs and target residual probs
            loss = loss_fn(pred_probs_flat, target_hist_flat)
            total_loss += loss.item()

            # Reconstruct full histogram from predicted residuals
            pred_probs = torch.softmax(pred_logits, dim=2)                  # (B, C, bins, H, W)

            pred_hist_reconstructed = input_hist + pred_probs               # add predicted residual
            pred_hist_reconstructed = torch.clamp(pred_hist_reconstructed, min=0.0)
            pred_hist_reconstructed = pred_hist_reconstructed / (pred_hist_reconstructed.sum(dim=2, keepdim=True) + 1e-8)

            # Reconstruct full histogram from target residuals
            target_hist_reconstructed = input_hist + target_hist
            target_hist_reconstructed = torch.clamp(target_hist_reconstructed, min=0.0)
            target_hist_reconstructed = target_hist_reconstructed / (target_hist_reconstructed.sum(dim=2, keepdim=True) + 1e-8)

            # Decode histograms to RGB
            input_rgb_img = decode_image_from_probs(input_hist, bin_edges)
            target_rgb_img = decode_image_from_probs(target_hist_reconstructed, bin_edges)
            pred_rgb_img = decode_image_from_probs(pred_hist_reconstructed, bin_edges)

            # DEBUG logging
            if count % 5 == 0:
                for name, rgb in [("Input", input_rgb_img[0]),
                                  ("Target", target_rgb_img[0]),
                                  ("Predicted", pred_rgb_img[0])]:
                    means = rgb.mean(dim=(1, 2))
                    stds = rgb.std(dim=(1, 2))
                    mins = rgb.amin(dim=(1, 2))
                    maxs = rgb.amax(dim=(1, 2))
                    logger.info(f"{name} RGB Stats [Validation]")
                    for c in range(3):
                        logger.info(f"  Channel {c}: mean={means[c]:.4f}, std={stds[c]:.4f}, min={mins[c]:.4f}, max={maxs[c]:.4f}")

            # Compute PSNR
            for i in range(pred_rgb_img.size(0)):
                total_psnr += compute_psnr(pred_rgb_img[i], clean_img[i])
                count += 1

    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / count if count > 0 else 0.0
    return avg_loss, avg_psnr


# TRAIN STEP - DISTRIBUTION
def train_generative_epoch(model, loss_fn, dataloader, optimizer, device, n_bins, epoch):
    model.train()
    running_loss = 0.0
    i = 0

    for batch in dataloader:
        input_hist = batch['input_hist'].to(device)                     # (B, C, bins, H, W), already normalized
        target_hist = batch['target_hist'].to(device)                   # (B, C, bins, H, W), already normalized
        bin_edges = batch['bin_edges'].to(device)                       # (B, bins+1)

        optimizer.zero_grad()
        pred_logits = model(input_hist)                                 # (B, C, n_bins, H, W)

        # Rearrange predictions and targets for loss
        pred_logits = pred_logits.permute(0, 1, 3, 4, 2).contiguous()   # (B, C, H, W, n_bins)
        target_hist = target_hist.permute(0, 1, 3, 4, 2).contiguous()   # (B, C, H, W, n_bins)

        # Flatten to (N, n_bins) where N = B*C*H*W
        pred_logits_flat = pred_logits.view(-1, n_bins)                 # (B*C*H*W, n_bins)
        target_hist_flat = target_hist.view(-1, n_bins)                 # (B*C*H*W, n_bins)

        loss = loss_fn(pred_logits_flat, target_hist_flat)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if epoch>=100:
            plot_input_vs_prediction(
                input_hist=input_hist,
                residual=pred_logits.permute(0, 1, 4, 2, 3).detach().cpu(),  # (B, C, bins, H, W)
                bin_edges=bin_edges.detach().cpu(),
                device=device,
                step_info=f"Epoch, Batch {i}",
                clean_img=batch["clean"] if "clean" in batch else None
            )
        i += 1

    return running_loss / len(dataloader)


# VALIDATION STEP - DISTRIBUTION
def validate_generative_epoch(model, loss_fn, dataloader, device, n_bins):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_hist = batch['input_hist'].to(device)                     # (B, C, bins, H, W), already normalized
            target_hist = batch['target_hist'].to(device)                   # (B, C, bins, H, W), already normalized
            clean_img = batch['clean'].to(device)                           # (B, C, H, W)
            bin_edges = batch['bin_edges'].to(device)                       # (B, bins+1)

            # Forward pass
            pred_logits = model(input_hist)                                 # (B, C, bins, H, W)

            # Prepare shapes for loss
            pred_logits_flat = pred_logits.permute(0, 1, 3, 4, 2).contiguous().view(-1, n_bins)  # (N, bins)
            target_hist_flat = target_hist.permute(0, 1, 3, 4, 2).contiguous().view(-1, n_bins)  # (N, bins)

            # Compute cross entrpy loss between logits out from network and target distribution
            loss = loss_fn(pred_logits_flat, target_hist_flat)
            total_loss += loss.item()

            # Decode histograms to RGB
            pred_probs = torch.softmax(pred_logits, dim=2)
            pred_rgb_img = decode_image_from_probs(pred_probs, bin_edges)   # (B, C, H, W)

            # DEBUG logging
            if count % 5 == 0:
                # print dist stats
                x, y = 50, 50
                logger.info(f"Input Dist: {input_hist[0, 0, :, x, y]}")
                logger.info(f"Target Dist: {target_hist[0, 0, :, x, y]}")
                logger.info(f"Pred Dist: {pred_probs[0, 0, :, x, y]}")

                # print image stats (pred)
                means = pred_rgb_img.mean(dim=[0, 2, 3])  # Mean over batch, height, width per channel
                stds = pred_rgb_img.std(dim=[0, 2, 3])    # Std over batch, height, width per channel
                mins = pred_rgb_img.amin(dim=[0, 2, 3])   # Min per channel
                maxs = pred_rgb_img.amax(dim=[0, 2, 3])   # Max per channel

                for c in range(3):
                    logger.info(f"  Channel {c}: mean={means[c].item():.4f}, std={stds[c].item():.4f}, min={mins[c].item():.4f}, max={maxs[c].item():.4f}")

            # Compute PSNR
            for i in range(pred_rgb_img.size(0)):
                total_psnr += compute_psnr(pred_rgb_img[i], clean_img[i])
                count += 1

    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / count if count > 0 else 0.0
    return avg_loss, avg_psnr


# CROSS ENTROPY LOSS FUNCTION
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_logits, target_probs):
        """
        Args:
            pred_logits: (N, n_bins) unnormalized logits predicted by your model
            target_probs: (N, n_bins) target histograms (normalized to sum=1)
        
        Returns:
            Scalar loss value
        """
        log_probs = F.log_softmax(pred_logits, dim=1)           # (N, n_bins)
        
        # Cross entropy: -sum(target * log(predicted)) per sample
        ce_loss = -(target_probs * log_probs).sum(dim=1)        # (N,)
        
        if self.reduction == 'mean':
            return ce_loss.mean()
        elif self.reduction == 'sum':
            return ce_loss.sum()
        else:
            return ce_loss


# TRAINING LOOP - DISTRIBUTION
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
    loss_fn = CustomCrossEntropyLoss()

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    psnr_values = []

    date_str = datetime.now().strftime("%Y-%m-%d")
    model_name = f"{datetime.now().strftime('%Y%m%d')}_hist2hist_{model_cfg['out_mode']}_{config['num_epochs']}ep.pth"
    save_path = Path(config["save_dir"]) / model_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training...")

    for epoch in range(config["num_epochs"]):
        start_time = time.time()

        train_loss = train_generative_epoch(model, loss_fn, train_loader, optimizer, device, n_bins, epoch)
        val_loss, val_psnr = validate_generative_epoch(model, loss_fn, val_loader, device, n_bins)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        psnr_values.append(val_psnr)

        logger.info(f"[Epoch {epoch+1}/{config['num_epochs']}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f} dB | Time: {time.time() - start_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")

    save_loss_plot(train_losses, val_losses, save_dir="plots", filename=f"{date_str}_hist2hist_dist_loss_plot.png")
    save_psnr_plot(psnr_values, save_dir="plots", filename=f"{date_str}_{dataset_cfg['mode']}_dist_psnr_plot.png")


# TRAINING LOOP - RESIDUAL
def train_histogram_residual(config):
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
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    psnr_values = []

    date_str = datetime.now().strftime("%Y-%m-%d")
    model_name = f"{datetime.now().strftime('%Y%m%d')}_hist2hist_{model_cfg['out_mode']}_{config['num_epochs']}ep.pth"
    save_path = Path(config["save_dir"]) / model_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training...")

    for epoch in range(config["num_epochs"]):
        start_time = time.time()

        train_loss = train_residual_epoch(model, loss_fn, train_loader, optimizer, device, n_bins, epoch)
        val_loss, val_psnr = validate_residual_epoch(model, loss_fn, val_loader, device, n_bins)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        psnr_values.append(val_psnr)

        logger.info(f"[Epoch {epoch+1}/{config['num_epochs']}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f} dB | Time: {time.time() - start_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")

    save_loss_plot(train_losses, val_losses, save_dir="plots", filename=f"{date_str}_hist2hist_residual_oss_plot.png")
    save_psnr_plot(psnr_values, save_dir="plots", filename=f"{date_str}_{dataset_cfg['mode']}_residual_psnr_plot.png")


# GENERATIVE ACCUMULATION - INFERENCE
def run_generative_accumulation_pipeline(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Load dataset
    dataset_cfg = config["dataset"]
    dataset = HistogramDataset(**dataset_cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'])

    # Load model
    model_cfg = config["model"]
    model = UNet(
        in_channels=model_cfg["in_channels"],
        n_bins=dataset_cfg["hist_bins"],
        out_mode=model_cfg["out_mode"],
        merge_mode=model_cfg["merge_mode"],
        depth=model_cfg["depth"],
        start_filters=model_cfg["start_filters"],
        mode=dataset_cfg["mode"]
    ).to(device)
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.eval()

    output_dir = Path(config.get("output_dir", "plots"))
    output_dir.mkdir(exist_ok=True, parents=True)

    num_samples = config.get("num_samples", 4)
    num_steps = config.get("num_steps", 50)

    for idx, batch in enumerate(dataloader):
        if idx >= num_samples:
            break

        bin_edges = batch["bin_edges"].to(device)
        clean_img = batch["clean"].to(device)
        input_hist = batch["input_hist"].to(device)  # normalized histogram

        # Restore raw count scale
        total_init_counts = 32  # or whatever your dataset normalization used
        current_hist = input_hist * total_init_counts

        # Track photon count for normalization
        photon_count = torch.full_like(current_hist[:, :, :1], total_init_counts)  # shape (B, C, 1, H, W)

        imgs_to_show = []

        with torch.no_grad():
            for _ in range(num_steps):
                # Normalize before feeding into model
                normalized_hist = current_hist / photon_count.clamp(min=1e-6)

                logits = model(normalized_hist)
                probs = torch.softmax(logits, dim=2)

                # Sample one bin per pixel
                B, C, n_bins, H, W = probs.shape
                probs_flat = probs.permute(0, 1, 3, 4, 2).reshape(-1, n_bins)
                sampled_bins = torch.multinomial(probs_flat, num_samples=1).squeeze(1)                      # kind of random can go to wrong bins
                # sampled_bins = torch.argmax(probs, dim=2)  # shape: (B, C, H, W)                          # goes to most probabable bin (wrong, too deterministic)
                sampled_one_hot = torch.nn.functional.one_hot(sampled_bins, num_classes=n_bins).float()

                # Reshape back to histogram format
                sampled_hist = sampled_one_hot.view(B, C, H, W, n_bins).permute(0, 1, 4, 2, 3).contiguous()

                # Accumulate photon count
                current_hist += sampled_hist
                photon_count += 1.0  # adding one photon per pixel

                # Decode to RGB using normalized histogram
                normalized_hist = current_hist / photon_count.clamp(min=1e-6)
                pred_rgb = decode_image_from_probs(normalized_hist, bin_edges)
                imgs_to_show.append(pred_rgb[0].cpu())

        total_plots = num_steps + 1
        cols = min(6, total_plots)
        rows = (total_plots + cols - 1) // cols

        plt.figure(figsize=(4 * cols, 4 * rows))
        for i, img in enumerate(imgs_to_show):
            psnr_val = compute_psnr(img, clean_img[0])
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img.permute(1, 2, 0))
            plt.title(f"Step {i+1} - PSNR {psnr_val:.2f}")
            plt.axis("off")

        # Ground Truth
        plt.subplot(rows, cols, total_plots)
        plt.imshow(clean_img[0].permute(1, 2, 0).cpu())
        plt.title("Ground Truth Image")
        plt.axis("off")

        plt.tight_layout()
        save_path = output_dir / f"sample_{idx}_progressive_generation.png"
        plt.savefig(save_path)
        plt.show()
        print(f"Saved progressive generation plot to {save_path}")


def iterative_evaluate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Generating on {device}")

    dataset_cfg = config["dataset"]
    n_bins = dataset_cfg["hist_bins"]
    num_samples = config.get("num_samples", 4)
    num_steps = config.get("num_steps", 10)

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
        bin_edges = batch["bin_edges"].to(device)    # shape: (1, 3, H, W)

        B, C, _, H, W = input_hist.shape
        accumulated_radiance = torch.zeros((B, C, H, W), device=device)

        _, axs = plt.subplots(1, num_steps + 3, figsize=(4 * (num_steps + 2), 4))

        axs[0].imshow(decode_image_from_probs(input_hist, bin_edges)[0].permute(1, 2, 0).cpu())
        axs[0].set_title("Original Input Hist")
        axs[0].axis("off")

        for step in range(1, num_steps+1):
            logits = model(input_hist)
            probs = torch.softmax(logits, dim=2)
            pred_radiance = decode_image_from_probs(probs, bin_edges)

            accumulated_radiance += (pred_radiance / num_steps)

            psnr = compute_psnr(accumulated_radiance[0], clean_img[0])
            axs[step].imshow(accumulated_radiance[0].permute(1, 2, 0).cpu())
            axs[step].set_title(f"Step {step+1}\nPSNR: {psnr:.2f} dB")
            axs[step].axis("off")

        # Final Output
        final_psnr = compute_psnr(accumulated_radiance[0], clean_img[0])
        axs[num_steps + 1].imshow(accumulated_radiance[0].permute(1, 2, 0).cpu())
        axs[num_steps + 1].set_title(f"Final Output\nPSNR: {final_psnr:.2f} dB")
        axs[num_steps + 1].axis("off")

        # Ground Truth
        axs[num_steps + 2].imshow(clean_img[0].permute(1, 2, 0).cpu())
        axs[num_steps + 2].set_title("Ground Truth")
        axs[num_steps + 2].axis("off")

        plt.tight_layout()
        plot_path = os.path.join("plots", f"sample_{idx}_evaluation.png")
        plt.savefig(plot_path)
        plt.show()
        print(f"Saved plot to {plot_path}")


def visualize_predictions(model, dataloader, device, num_samples=5):
    model.eval()

    with torch.no_grad():
        count = 0
        for batch in dataloader:
            input_hist = batch['input_hist'].to(device)                  # (B, C, bins, H, W)
            target_hist = batch['target_hist'].to(device) - input_hist   # residual histogram (already normalized)
            bin_edges = batch['bin_edges'].to(device)                    # (B, bins+1)
            clean = batch['clean']                                        # clean image for PSNR comparison

            pred_logits = model(input_hist)                               # (B, C, bins, H, W)

            # Permute for consistency: (B, C, H, W, bins)
            pred_logits = pred_logits.permute(0, 1, 3, 4, 2).contiguous()
            target_hist = target_hist.permute(0, 1, 3, 4, 2).contiguous()

            # Convert predicted residual logits -> probabilities with softmax
            residual_probs = torch.softmax(pred_logits, dim=-1)          # (B, C, H, W, bins)

            # Permute residual_probs back to (B, C, bins, H, W) for addition
            residual_probs = residual_probs.permute(0, 1, 4, 2, 3)

            # Add predicted residual probabilities to input histogram and clamp
            pred_hist_reconstructed = input_hist + residual_probs
            pred_hist_reconstructed = torch.clamp(pred_hist_reconstructed, min=0.0)

            # Re-normalize over bins (dim=2)
            pred_hist_reconstructed = pred_hist_reconstructed / (pred_hist_reconstructed.sum(dim=2, keepdim=True) + 1e-8)

            # Do the same for the target residual histogram
            target_hist_reconstructed = input_hist + target_hist.permute(0, 1, 4, 2, 3)
            target_hist_reconstructed = torch.clamp(target_hist_reconstructed, min=0.0)
            target_hist_reconstructed = target_hist_reconstructed / (target_hist_reconstructed.sum(dim=2, keepdim=True) + 1e-8)

            # Decode histograms to RGB images
            input_rgb_img = decode_image_from_probs(input_hist, bin_edges)                 # (B,C,H,W)
            target_rgb_img = decode_image_from_probs(target_hist_reconstructed, bin_edges) # (B,C,H,W)
            pred_rgb_img = decode_image_from_probs(pred_hist_reconstructed, bin_edges)     # (B,C,H,W)

            # Compute PSNR with clean images
            psnr_input_clean = compute_psnr(input_rgb_img, clean.to(device))
            psnr_pred_clean = compute_psnr(pred_rgb_img, clean.to(device))

            B, _, _, _, _ = input_hist.shape

            for i in range(B):
                # IMAGE PREDICTIONS
                _, axes = plt.subplots(1, 4, figsize=(12, 4))
                axes[0].imshow(pred_rgb_img[i].cpu().permute(1, 2, 0))
                axes[0].set_title(f'Predicted RGB {psnr_pred_clean}')
                axes[1].imshow(target_rgb_img[i].cpu().permute(1, 2, 0))
                axes[1].set_title('Target RGB')
                axes[2].imshow(clean[i].cpu().permute(1, 2, 0))
                axes[2].set_title('Clean (High SPP)')
                axes[3].imshow(input_rgb_img[i].cpu().permute(1, 2, 0))
                axes[3].set_title(f'Input RGB {psnr_input_clean}')

                for ax in axes:
                    ax.axis('off')
                plt.tight_layout()
                plt.show()

                count += 1
                if count >= num_samples:
                    return


def test_histogram_generator(config):
    '''
    Tests a single prediction of the generative pipeline given a noisy histogram as input. This way we see whether the model has learning anything from training.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Testing on {device}")

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    n_bins = dataset_cfg["hist_bins"]

    # Load model
    model = UNet(
        in_channels=model_cfg["in_channels"],
        n_bins=n_bins,
        out_mode=model_cfg["out_mode"],
        merge_mode=model_cfg["merge_mode"],
        depth=model_cfg["depth"],
        start_filters=model_cfg["start_filters"],
        mode=dataset_cfg["mode"]
    ).to(device)

    model_path = config.get("model_path")
    assert model_path is not None and os.path.exists(model_path), f"Model path '{model_path}' is invalid"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Loaded model from {model_path}")

    # Load validation set
    _, val_loader = get_generative_dataloaders(config)

    # Run visualization
    num_samples = config.get("num_samples", 4)
    visualize_predictions(model, val_loader, device, num_samples=num_samples)


def plot_input_vs_prediction(input_hist, residual, bin_edges, device, step_info="", clean_img=None):
    """
    Plots decoded input vs prediction (and optionally the clean GT).

    Args:
        input_hist: tensor (B, C, bins, H, W), original histograms
        residual: tensor (B, C, bins, H, W), predicted residuals (logits)
        bin_edges: tensor (B, bins+1), bin edges for decoding
        device: torch device (not strictly needed here if tensors already on CPU)
        step_info: str, info string for plot title
        clean_img: tensor (B, C, H, W), ground truth RGB images (optional)
    """
    # Convert predicted residual logits -> probabilities with softmax
    residual_probs = torch.softmax(residual, dim=2)          # (B, C, bins, H, W)

    # Add predicted residual probabilities to input histogram and clamp
    pred_hist_reconstructed = input_hist + residual_probs
    pred_hist_reconstructed = torch.clamp(pred_hist_reconstructed, min=0.0)

    # Re-normalize over bins (dim=2)
    pred_hist_reconstructed = pred_hist_reconstructed / (pred_hist_reconstructed.sum(dim=2, keepdim=True) + 1e-8)

    # Decode histograms to RGB images
    input_rgb_img = decode_image_from_probs(input_hist, bin_edges)                 # (B,C,H,W)
    pred_rgb_img = decode_image_from_probs(pred_hist_reconstructed, bin_edges)     # (B,C,H,W)

    psnr_input = compute_psnr(input_rgb_img, clean_img)
    psnr_pred = compute_psnr(pred_rgb_img, clean_img)

    num_imgs = 5

    for i in range(num_imgs):
        n_cols = 3 if clean_img is not None else 2
        _, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

        axes[0].imshow(input_rgb_img[i].permute(1, 2, 0))
        axes[0].set_title(f"Input Histogram - PSNR {psnr_input}")
        axes[0].axis("off")

        axes[1].imshow(pred_rgb_img[i].permute(1, 2, 0))
        axes[1].set_title(f"Prediction {step_info} - PSNR {psnr_pred}")
        axes[1].axis("off")

        if clean_img is not None:
            axes[2].imshow(clean_img[i].cpu().permute(1, 2, 0))
            axes[2].set_title("Ground Truth")
            axes[2].axis("off")

        plt.tight_layout()
        plt.show(block=True)
