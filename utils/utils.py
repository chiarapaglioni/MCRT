import tifffile
import matplotlib.pyplot as plt

def save_tiff(data, file_name):
    """
    Saves data of shape (N, H, W, C, B) to TIFF file using BigTIFF if needed.

    Parameters:
    - data (np array): data to save
    - file_name (str): file name / scene name
    """
    tifffile.imwrite(file_name, data, compression='lzw', bigtiff=True)
    print(f"Saved {file_name} with shape {data.shape} <3")


def plot_images(noisy, hist_pred, noise_pred, target, clean=None):
    """
    Plot denoised images generated from the noise2noise and hist2nosie next to the clean one.

    Parameters: 
    - noisy (torch tensor): noisy input, i.e. average or histogram of N-1 samples rendered with 1 spp
    - hist_pred (torch tensor): denoised prediction from hist2noise
    - noise_pred (torch tensor): denoised prediction from noise2noise
    - target (torch tensor): noisy target (1 sample)
    - clean (torch tensor): clean GT rendered with high spp
    """
    def to_img(t):
        if t.dim() == 4:  # [1, 3, H, W]
            t = t.squeeze(0)
        return t.detach().cpu().numpy().transpose(1, 2, 0)
    
    fig, axes = plt.subplots(1, 5 if clean is not None else 4, figsize=(20, 4))
    axes[0].imshow(to_img(noisy));       axes[0].set_title("Noisy Input")
    axes[1].imshow(to_img(target));      axes[1].set_title("Target Sample")
    axes[2].imshow(to_img(hist_pred));   axes[2].set_title("Hist2Noise Output")
    axes[3].imshow(to_img(noise_pred));  axes[3].set_title("Noise2Noise Output")
    if clean is not None:
        axes[4].imshow(to_img(clean));   axes[4].set_title("Clean (GT)")
    for ax in axes: ax.axis('off')
    plt.tight_layout(); plt.show()