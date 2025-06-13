import jdata
import numpy as np
import matplotlib.pyplot as plt
import os

def load_mcx_jnifti(path):
    """Load MCX output (.json JNIfTI format) and extract raw data."""
    data = jdata.load(path)
    array = np.array(data["NIFTIData"])
    return array

def reshape_to_TDHW(array):
    """
    Convert from [Z, Y, X, T, C=1] to [T, D, H, W].
    """
    if array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)  # → [Z, Y, X, T]
    Z, Y, X, T = array.shape
    return np.transpose(array, (3, 0, 1, 2))  # → [T, D, H, W]

def plot_decay(voxel_decay, label=None):
    plt.plot(voxel_decay, label=label)
    plt.xlabel("Time Bin")
    plt.ylabel("Fluence (a.u.)")
    plt.title("Photon Arrival Histogram")
    if label:
        plt.legend()
    plt.grid(True)

def main():
    # === Replace these with your actual file paths ===
    channel_files = [
        "ch1_520nm.json",
        "ch2_580nm.json",
        "ch3_650nm.json"
    ]

    all_channels = []

    for path in channel_files:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        print(f"Loading {path}")
        data_raw = load_mcx_jnifti(path)
        data_tdhw = reshape_to_TDHW(data_raw)
        all_channels.append(data_tdhw)

    # Stack across channel dimension
    data = np.stack(all_channels, axis=0)  # [C, T, D, H, W]
    print(f"\n Final data shape: {data.shape} (C, T, D, H, W)")

    # === Visualization ===
    # Pick voxel (center by default)
    C, T, D, H, W = data.shape
    z, y, x = D // 2, H // 2, W // 2

    plt.figure(figsize=(8, 5))
    for c in range(C):
        decay = data[c, :, z, y, x]
        plot_decay(decay, label=f"Channel {c+1}")
    plt.tight_layout()
    plt.show(block=True)

if __name__ == "__main__":
    main()
