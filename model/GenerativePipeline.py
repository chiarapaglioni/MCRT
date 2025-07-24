# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# Utils
import os
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
# Custom
from model.UNet import UNet
from dataset.HistDataset import HistogramDataset
from utils.utils import save_loss_plot, decode_pred_logits, compute_psnr, save_psnr_plot

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


def train_generative_epoch(model, dataloader, optimizer, device, n_bins, epoch):
    model.train()
    running_loss = 0.0
    i = 0

    for batch in dataloader:
        input_hist = batch['input_hist'].to(device)                     # (B, C, bins, H, W)
        target_hist = batch['target_hist'].to(device)                   # (B, C, bins, H, W), already normalized
        bin_edges = batch['bin_edges'].to(device)                       # (B, bins+1)

        optimizer.zero_grad()
        pred_logits = model(input_hist)                                 # (B, C, n_bins, H, W)

        # Rearrange predictions and targets for loss
        pred_logits = pred_logits.permute(0, 1, 3, 4, 2).contiguous()  # (B, C, H, W, n_bins)
        target_hist = target_hist.permute(0, 1, 3, 4, 2).contiguous()  # (B, C, H, W, n_bins)

        # Flatten to (N, n_bins) where N = B*C*H*W
        pred_logits_flat = pred_logits.view(-1, n_bins)
        target_hist_flat = target_hist.view(-1, n_bins)

        # Compute log probabilities
        log_probs = F.log_softmax(pred_logits_flat, dim=1)

        # KL divergence expects input=log_probs, target=probabilities
        loss = F.kl_div(log_probs, target_hist_flat, reduction='batchmean')

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if epoch==20:
            plot_input_vs_prediction(
                input_hist=batch["input_hist"],
                pred_logits=model(batch["input_hist"].to(device)),
                bin_edges=batch["bin_edges"],
                device=device,
                step_info=f"Epoch, Batch {i}",
                clean_img=batch["clean"] if "clean" in batch else None
            )
        i += 1

    return running_loss / len(dataloader)


def validate_generative_epoch(model, dataloader, device, n_bins):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_hist = batch['input_hist'].to(device)
            target_hist = batch['target_hist'].to(device)
            bin_edges = batch['bin_edges'].to(device)

            pred_logits = model(input_hist)

            pred_logits = pred_logits.permute(0, 1, 3, 4, 2).contiguous()
            target_hist = target_hist.permute(0, 1, 3, 4, 2).contiguous()

            pred_logits_flat = pred_logits.view(-1, n_bins)
            target_hist_flat = target_hist.view(-1, n_bins)

            log_probs = F.log_softmax(pred_logits_flat, dim=1)
            loss = F.kl_div(log_probs, target_hist_flat, reduction='batchmean')
            total_loss += loss.item()

            # Decode to RGB for PSNR calculation
            probs = torch.softmax(pred_logits, dim=2).permute(0, 1, 4, 2, 3)  # (B, C, n_bins, H, W)
            pred_rgb = decode_pred_logits(probs, bin_edges)
            target_rgb = decode_pred_logits(target_hist.permute(0, 1, 4, 2, 3), bin_edges)

            # DEBUG
            if count == 0:  # log once per validation
                input_rgb = decode_pred_logits(input_hist, bin_edges)
                for name, rgb in [("Input", input_rgb), ("Target", target_rgb), ("Predicted", pred_rgb)]:
                    means = rgb.mean(dim=(0, 2, 3))
                    stds = rgb.std(dim=(0, 2, 3))
                    mins = rgb.amin(dim=(0, 2, 3))
                    maxs = rgb.amax(dim=(0, 2, 3))

                    logger.info(f"{name} RGB Stats [Validation]")
                    for c in range(3):
                        logger.info(f"  Channel {c}: mean={means[c]:.4f}, std={stds[c]:.4f}, min={mins[c]:.4f}, max={maxs[c]:.4f}")

            # TOTAL PSNR
            for i in range(pred_rgb.size(0)):
                total_psnr += compute_psnr(pred_rgb[i], target_rgb[i])
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
    # TODO: test replacement by KL loss
    criterion = nn.CrossEntropyLoss()

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

        train_loss = train_generative_epoch(model, train_loader, optimizer, device, n_bins, epoch)
        val_loss, val_psnr = validate_generative_epoch(model, val_loader, device, n_bins)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        psnr_values.append(val_psnr)

        logger.info(f"[Epoch {epoch+1}/{config['num_epochs']}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f} dB | Time: {time.time() - start_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")

    save_loss_plot(train_losses, val_losses, save_dir="plots", filename="hist2hist_loss_plot.png")
    save_psnr_plot(psnr_values, save_dir="plots", filename=f"{date_str}_{dataset_cfg['mode']}_psnr_plot.png")



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
            for step in range(num_steps):
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
                pred_rgb = decode_pred_logits(normalized_hist, bin_edges)
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

        axs[0].imshow(decode_pred_logits(input_hist, bin_edges)[0].permute(1, 2, 0).cpu())
        axs[0].set_title("Original Input Hist")
        axs[0].axis("off")

        for step in range(1, num_steps+1):
            logits = model(input_hist)
            probs = torch.softmax(logits, dim=2)
            pred_radiance = decode_pred_logits(probs, bin_edges)

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
            input_hist = batch['input_hist'].to(device)
            target_hist = batch['target_hist'].to(device)
            bin_edges = batch['bin_edges'].to(device)
            clean = batch['clean']  # for comparison (optional)

            pred_logits = model(input_hist)
            probs = torch.softmax(pred_logits, dim=2)

            input_rgb = decode_pred_logits(input_hist, bin_edges)
            pred_rgb = decode_pred_logits(probs, bin_edges)
            target_rgb = decode_pred_logits(target_hist, bin_edges)

            psnr_input_clean = compute_psnr(input_rgb, clean)
            psnr_pred_clean = compute_psnr(pred_rgb, clean)

            B, C, _, H, W = input_hist.shape

            for i in range(B):
                # IMAGE PREDICTIONS
                fig, axes = plt.subplots(1, 4, figsize=(12, 4))
                axes[0].imshow(pred_rgb[i].cpu().permute(1, 2, 0))
                axes[0].set_title(f'Predicted RGB {psnr_pred_clean}')
                axes[1].imshow(target_rgb[i].cpu().permute(1, 2, 0))
                axes[1].set_title('Target RGB')
                axes[2].imshow(clean[i].cpu().permute(1, 2, 0))
                axes[2].set_title('Clean (High SPP)')
                axes[3].imshow(input_rgb[i].cpu().permute(1, 2, 0))
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



def visualize_predictions(model, dataloader, device, num_samples=5):
    model.eval()

    with torch.no_grad():
        count = 0
        for batch in dataloader:
            input_hist = batch['input_hist'].to(device)
            target_hist = batch['target_hist'].to(device)
            bin_edges = batch['bin_edges'].to(device)
            clean = batch['clean']  # for comparison (optional)

            pred_logits = model(input_hist)
            probs = torch.softmax(pred_logits, dim=2)

            input_rgb = decode_pred_logits(input_hist, bin_edges)
            pred_rgb = decode_pred_logits(probs, bin_edges)
            target_rgb = decode_pred_logits(target_hist, bin_edges)

            psnr_input_clean = compute_psnr(input_rgb, clean)
            psnr_pred_clean = compute_psnr(pred_rgb, clean)

            B, C, _, H, W = input_hist.shape

            for i in range(B):
                # IMAGE PREDICTIONS
                fig, axes = plt.subplots(1, 4, figsize=(12, 4))
                axes[0].imshow(pred_rgb[i].cpu().permute(1, 2, 0))
                axes[0].set_title(f'Predicted RGB {psnr_pred_clean}')
                axes[1].imshow(target_rgb[i].cpu().permute(1, 2, 0))
                axes[1].set_title('Target RGB')
                axes[2].imshow(clean[i].cpu().permute(1, 2, 0))
                axes[2].set_title('Clean (High SPP)')
                axes[3].imshow(input_rgb[i].cpu().permute(1, 2, 0))
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


def plot_input_vs_prediction(input_hist, pred_logits, bin_edges, device, step_info="", clean_img=None):
    """
    Plots decoded input vs prediction (and optionally the clean GT).
    """
    input_rgb = decode_pred_logits(input_hist.detach().cpu(), bin_edges.detach().cpu())
    probs = torch.softmax(pred_logits.detach().cpu(), dim=2)
    pred_rgb = decode_pred_logits(probs, bin_edges.cpu())

    B = input_rgb.size(0)

    for i in range(B):
        fig, axes = plt.subplots(1, 3 if clean_img is not None else 2, figsize=(15, 5))

        axes[0].imshow(input_rgb[i].permute(1, 2, 0))
        axes[0].set_title("Input Histogram (decoded)")
        axes[0].axis("off")

        axes[1].imshow(pred_rgb[i].permute(1, 2, 0))
        axes[1].set_title(f"Prediction {step_info}")
        axes[1].axis("off")

        if clean_img is not None:
            axes[2].imshow(clean_img[i].cpu().permute(1, 2, 0))
            axes[2].set_title("Ground Truth")
            axes[2].axis("off")

        plt.tight_layout()
        plt.show()
