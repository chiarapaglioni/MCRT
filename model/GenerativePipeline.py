# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Utils
import os
import math
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
# Custom
from model.UNet import GapUNet
from dataset.HistImgDataset import CropHistogramBinomDataset, HistogramBinomDataset
from utils.utils import load_model, save_loss_plot, decode_image_from_probs, compute_psnr, save_psnr_plot, tonemap_gamma_correct

import logging
logger = logging.getLogger(__name__)




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
        
# WEIGHTED ENTROPY LOSS FUNCTION
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_logits, target_probs, bin_weights=None):
        """
        pred_logits: (N, B) - raw logits
        target_probs: (N, B) - target histogram probabilities
        bin_weights: (N, B) or (B,) - optional weights per bin
        """
        log_probs = F.log_softmax(pred_logits, dim=1)  # (N, B)

        if bin_weights is not None:
            # Apply weights
            weighted_ce = -(bin_weights * target_probs * log_probs).sum(dim=1)
        else:
            weighted_ce = -(target_probs * log_probs).sum(dim=1)

        if self.reduction == 'mean':
            return weighted_ce.mean()
        elif self.reduction == 'sum':
            return weighted_ce.sum()
        else:
            return weighted_ce


# DATA LOADERS
def get_generative_dataloaders(config, device):
    dataset_cfg = config['dataset'].copy()
    dataset_cfg['root_dir'] = Path(__file__).resolve().parents[1] / dataset_cfg['root_dir']
    dataset_cfg['device'] = device  

    if config['crop']:
        full_dataset = CropHistogramBinomDataset(**dataset_cfg)     # computes histogram per crop
    else:
        full_dataset = HistogramBinomDataset(**dataset_cfg)         # computes histogram per image

    total_len = len(full_dataset)
    val_ratio = config.get('val_split', 0.1)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    train_set, val_set = random_split(full_dataset, [train_len, val_len])

    logger.info(f"Total dataset size: {total_len}")
    logger.info(f"Training set size:  {len(train_set)}")
    logger.info(f"Validation set size: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    sample = next(iter(train_loader))
    logger.info(f"Input shape:  {sample['input_hist'].shape}")
    logger.info(f"Target shape: {sample['target_hist'].shape}")
    logger.info(f"Bin Edges:  {sample['bin_edges'].shape}")
    if 'clean' in sample and sample['clean'] is not None:
        logger.info(f"Clean shape:  {sample['clean'].shape}")

    return train_loader, val_loader


def count_bin0_only_histograms(input_hist, title):
    # input_hist: shape (B, C, bins, H, W)
    bins = input_hist.shape[2]
    device = input_hist.device

    # One-hot vector [1, 0, ..., 0]
    target = torch.zeros((bins,), device=device)
    target[0] = 1.0

    # Reshape histograms to (N, bins)
    input_flat = input_hist.permute(0, 1, 3, 4, 2).reshape(-1, bins)

    # Boolean mask where each histogram exactly matches target
    matches = (input_flat == target).all(dim=1)

    # Count and total
    count = matches.sum().item()
    total = input_flat.shape[0]

    percentage = (count / total) * 100
    logger.info(f"{title} hist: {count}/{total} histograms are bin-0-only ({percentage:.2f}%)")


# TRAIN STEP - DISTRIBUTION
def train_generative_epoch(model, loss_fn, dataloader, optimizer, device, n_bins, epoch, debug=True, plot_every_n=1):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        input_hist = batch['input_hist'].to(device)                     # (B, C, bins, H, W), already normalized
        target_hist = batch['target_hist'].to(device)                   # (B, C, bins, H, W), already normalized
        bin_edges = batch['bin_edges'].to(device)                       # (B, bins+1)
        # bin_weights = batch['bin_weights'][0].to(device)              # (B, bins+1)

        # logger.info(f"Bin Weights: {bin_weights}")
        if batch_idx % 20 == 0:
            count_bin0_only_histograms(input_hist, title="Input")
            count_bin0_only_histograms(target_hist, title="Target")

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

        total_loss += loss.item()

        # DEBUG
        if debug and batch_idx == 0 and epoch is not None and (epoch % plot_every_n == 0):
            plot_input_vs_prediction(
                input_hist=input_hist.detach().cpu(),
                predicted_logits=pred_logits.permute(0, 1, 4, 2, 3).detach().cpu(),  # (B, C, bins, H, W)
                bin_edges=bin_edges.detach().cpu(),
                clean_img=batch["clean"].detach().cpu() if "clean" in batch else None,
                step_info=f"epoch_{epoch}_batch_{batch_idx}",
                save_dir="debug_plots_gen"
            )

    return total_loss / len(dataloader)


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
            # bin_weights = batch['bin_weights'][1].to(device)                # (B, bins+1)

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


# TRAINING LOOP - DISTRIBUTION
def train_histogram_generator(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    # DATA LOADERS
    train_loader, val_loader = get_generative_dataloaders(config, device)
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    n_bins = dataset_cfg["hist_bins"]

    # MODEL
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

    # LOSS
    loss_fn = CustomCrossEntropyLoss()

    # OPTIMIZER & LR SCHEDULER
    optimizer = optim.Adam(model.parameters(), lr=float(model_cfg["learning_rate"]))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    psnr_values = []

    # Model Name
    date_str = datetime.now().strftime("%Y-%m-%d")
    model_name = f"{datetime.now().strftime('%Y%m%d')}_hist2hist_{model_cfg['out_mode']}_{config['num_epochs']}ep.pth"
    save_path = Path(config["save_dir"]) / model_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting DISTRIBUTION training...")

    for epoch in range(config["num_epochs"]):
        start_time = time.time()

        train_loss = train_generative_epoch(model, loss_fn, train_loader, optimizer, device, n_bins, epoch, debug=dataset_cfg['debug'], plot_every_n=config['plot_every'])
        val_loss, val_psnr = validate_generative_epoch(model, loss_fn, val_loader, device, n_bins)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        psnr_values.append(val_psnr)

        scheduler.step(val_loss)

        logger.info(f"[Epoch {epoch+1}/{config['num_epochs']}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f} dB | Time: {time.time() - start_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")

    save_loss_plot(train_losses, val_losses, save_dir="plots", filename=f"{date_str}_hist2hist_dist_loss_plot.png")
    save_psnr_plot(psnr_values, save_dir="plots", filename=f"{date_str}_{dataset_cfg['mode']}_dist_psnr_plot.png")



# GENERATIVE ACCUMULATION - INFERENCE
def run_generative_accumulation_pipeline(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Config
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    plot_every = config.get("plot_every", 1)  # plot every N steps, default to every step

    # Dataset
    if config['crop']:
        dataset = CropHistogramBinomDataset(**dataset_cfg)
    else:
        dataset = HistogramBinomDataset(**dataset_cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'])

    # Load model
    model = load_model(model_cfg, dataset_cfg, config['model_path'], device=device)

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

        B, C, n_bins, H, W = input_hist.shape

        # Initialize histogram and photon count based on start_from option
        if config.get("start_from", "noisy") == "empty":
            current_hist = torch.zeros_like(input_hist)
            photon_count = torch.zeros_like(current_hist[:, :, :1])  # (B,C,1,H,W)
        else:
            total_init_counts = dataset_cfg['low_spp'] - dataset_cfg['target_sample']
            current_hist = input_hist * total_init_counts
            photon_count = torch.full_like(current_hist[:, :, :1], total_init_counts)

        imgs_to_show = []
        steps_to_plot = []

        with torch.no_grad():
            for step in range(num_steps):
                normalized_hist = current_hist / photon_count.clamp(min=1e-6)

                logits = model(normalized_hist)
                probs = torch.softmax(logits, dim=2)

                B, C, n_bins, H, W = probs.shape
                probs_flat = probs.permute(0, 1, 3, 4, 2).reshape(-1, n_bins)
                sampled_bins = torch.multinomial(probs_flat, num_samples=1).squeeze(1)
                sampled_one_hot = torch.nn.functional.one_hot(sampled_bins, num_classes=n_bins).float()

                sampled_hist = sampled_one_hot.view(B, C, H, W, n_bins).permute(0, 1, 4, 2, 3).contiguous()

                current_hist += sampled_hist
                photon_count += 1.0

                # Plot only every N iterations
                if (step % plot_every) == 0:
                    normalized_hist = current_hist / photon_count.clamp(min=1e-6)
                    pred_rgb = decode_image_from_probs(normalized_hist, bin_edges)
                    imgs_to_show.append(pred_rgb[0].cpu())
                    steps_to_plot.append(step+1)

        # Include final step if not included yet
        if (num_steps - 1) % plot_every != 0:
            normalized_hist = current_hist / photon_count.clamp(min=1e-6)
            pred_rgb = decode_image_from_probs(normalized_hist, bin_edges)
            imgs_to_show.append(pred_rgb[0].cpu())
            steps_to_plot.append(num_steps)

        total_plots = len(imgs_to_show) + 1  # plus ground truth
        cols = min(6, total_plots)
        rows = (total_plots + cols - 1) // cols

        plt.figure(figsize=(4 * cols, 4 * rows))

        for i, (img, step_num) in enumerate(zip(imgs_to_show, steps_to_plot)):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img.permute(1, 2, 0))

            # Only compute and show PSNR if starting from noisy histogram
            if config.get("start_from", "noisy") == "noisy":
                psnr_val = compute_psnr(img, clean_img[0])
                plt.title(f"Step {step_num} - PSNR {psnr_val:.2f}")
            else:
                plt.title(f"Step {step_num}")

            plt.axis("off")

        # Ground Truth
        if config.get("start_from", "noisy") == "noisy":
            plt.subplot(rows, cols, total_plots)
            plt.imshow(clean_img[0].permute(1, 2, 0).cpu())
            plt.title("Ground Truth Image")
            plt.axis("off")

        plt.tight_layout()
        save_path = output_dir / f"sample_{idx}_progressive_generation.png"
        plt.savefig(save_path)
        plt.show(block=True)
        print(f"Saved progressive generation plot to {save_path}")



# ADDS RESIDUAL ITERATIVELLY TO RECOVER CLEAN IMAGE
def iterative_evaluate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Generating on {device}")

    dataset_cfg = config["dataset"]
    num_samples = config.get("num_samples", 4)
    num_steps = config.get("num_steps", 10)

    # Dataset
    if config['crop']:
        dataset = CropHistogramBinomDataset(**dataset_cfg)
    else:
        dataset = HistogramBinomDataset(**dataset_cfg)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=config['num_workers'], shuffle=False)

    model_cfg = config["model"]
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
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.eval()

    os.makedirs("plots", exist_ok=True)
    with torch.no_grad():
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

                # TODO: uncomment previous logic for residual prediction
                # accumulated_radiance += (pred_radiance / num_steps)
                accumulated_radiance = pred_radiance

                psnr = compute_psnr(accumulated_radiance[0], clean_img[0])
                axs[step].imshow(accumulated_radiance[0].permute(1, 2, 0).detach().cpu().numpy())
                axs[step].set_title(f"Step {step+1}\nPSNR: {psnr:.2f} dB")
                axs[step].axis("off")

            # Final Output
            final_psnr = compute_psnr(accumulated_radiance[0], clean_img[0])
            axs[num_steps + 1].imshow(accumulated_radiance[0].permute(1, 2, 0).detach().cpu().numpy())
            axs[num_steps + 1].set_title(f"Final Output\nPSNR: {final_psnr:.2f} dB")
            axs[num_steps + 1].axis("off")

            # Ground Truth
            axs[num_steps + 2].imshow(clean_img[0].permute(1, 2, 0).detach().cpu().numpy())
            axs[num_steps + 2].set_title("Ground Truth")
            axs[num_steps + 2].axis("off")

            plt.tight_layout()
            plot_path = os.path.join("plots", f"sample_{idx}_evaluation.png")
            plt.savefig(plot_path)
            plt.show()
            print(f"Saved plot to {plot_path}")


def visualize_residual_predictions(model, dataloader, device, num_samples=5):
    model.eval()

    with torch.no_grad():
        count = 0
        for batch in dataloader:
            input_hist = batch['input_hist'].to(device)                     # (B, C, bins, H, W)
            target_hist = batch['target_hist'].to(device) - input_hist      # residual histogram (already normalized)
            bin_edges = batch['bin_edges'].to(device)                       # (B, bins+1)
            clean = batch['clean']                                          # clean image for PSNR comparison

            pred_logits = model(input_hist)                                 # (B, C, bins, H, W)

            # Permute for consistency: (B, C, H, W, bins)
            pred_logits = pred_logits.permute(0, 1, 3, 4, 2).contiguous()
            target_hist = target_hist.permute(0, 1, 3, 4, 2).contiguous()

            # Convert predicted logits to probabilities with softmax
            residual_probs = torch.softmax(pred_logits, dim=-1)             # (B, C, H, W, bins)
            residual_probs = residual_probs.permute(0, 1, 4, 2, 3)          # (B, C, bins, H, W)

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
                

def visualize_predictions(model, dataloader, device, num_samples=5):
    model.eval()

    with torch.no_grad():
        count = 0
        for batch in dataloader:
            input_hist = batch['input_hist'].to(device)                         # (B, C, bins, H, W)
            target_hist = batch['target_hist'].to(device)                       # residual histogram (already normalized)
            bin_edges = batch['bin_edges'].to(device)                           # (B, bins+1)
            clean = batch['clean']                                              # clean image for PSNR comparison

            pred_logits = model(input_hist)                                     # (B, C, bins, H, W) logits
            pred_probs = torch.softmax(pred_logits, dim=2)                      # (B, C, bins, H, W) probs

            # Decode histograms to RGB images
            input_rgb_img = decode_image_from_probs(input_hist, bin_edges)      # (B,C,H,W)
            target_rgb_img = decode_image_from_probs(target_hist, bin_edges)    # (B,C,H,W)
            pred_rgb_img = decode_image_from_probs(pred_probs, bin_edges)       # (B,C,H,W)

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

    model_path = config.get("model_path")
    assert model_path is not None and os.path.exists(model_path), f"Model path '{model_path}' is invalid"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Loaded model from {model_path}")

    # Load validation set
    _, val_loader = get_generative_dataloaders(config, device)

    # Run visualization
    num_samples = config.get("num_samples", 4)
    if config['gen_mode']=='res':
        visualize_residual_predictions(model, val_loader, device, num_samples=num_samples)
    else: 
        visualize_predictions(model, val_loader, device, num_samples=num_samples)


def plot_input_vs_prediction(input_hist, predicted_logits, bin_edges, clean_img=None, step_info="", save_dir="debug_plots_gen"):
    """
    Plot and save decoded RGB images from input histograms and predicted logits.

    Args:
        input_hist (tensor): (B, C, bins, H, W)
        predicted_logits (tensor): (B, C, bins, H, W)
        bin_edges (tensor): (B, bins+1)
        clean_img (tensor): (B, C, H, W)
        step_info (str): string used in filenames and titles
        save_dir (str): directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert logits to probabilities
    predicted_probs = torch.softmax(predicted_logits, dim=2)  # (B, C, bins, H, W)

    # Decode histograms into RGB images
    input_hist = input_hist[:, :, :16, :, :]  # Remove confidence bin
    input_rgb_img = decode_image_from_probs(input_hist, bin_edges)                 # (B,C,H,W)
    pred_rgb_img = decode_image_from_probs(predicted_probs, bin_edges)             # (B,C,H,W)

    logger.info(f"Input: Min {input_rgb_img.min()} - Max {input_rgb_img.max()} - Mean {input_rgb_img.mean()} - Std {input_rgb_img.std()}")
    logger.info(f"Pred: Min {pred_rgb_img.min()} - Max {pred_rgb_img.max()} - Mean {pred_rgb_img.mean()} - Std {pred_rgb_img.std()}")

    # PSNR if GT available
    psnr_input = compute_psnr(input_rgb_img[0], clean_img[0]) if clean_img is not None else None
    psnr_pred = compute_psnr(pred_rgb_img[0], clean_img[0]) if clean_img is not None else None

    # Only plot the first image in the batch
    i = 0
    n_cols = 3 if clean_img is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(tonemap_gamma_correct(input_rgb_img[i].permute(1, 2, 0).numpy()))
    axes[0].set_title(f"Input (PSNR={psnr_input:.2f}dB)" if psnr_input else "Input")
    axes[0].axis("off")

    axes[1].imshow(tonemap_gamma_correct(pred_rgb_img[i].permute(1, 2, 0).numpy()))
    axes[1].set_title(f"Prediction (PSNR={psnr_pred:.2f}dB)" if psnr_pred else "Prediction")
    axes[1].axis("off")

    if clean_img is not None:
        axes[2].imshow(tonemap_gamma_correct(clean_img[i].permute(1, 2, 0).numpy()))
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

    plt.suptitle(f"Step: {step_info}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Avoid title cropping

    # Save plot
    save_path = os.path.join(save_dir, f"plot_{step_info}.png")
    plt.savefig(save_path)
    plt.close()
