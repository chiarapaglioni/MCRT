# Data Processing + Utils
import zipfile
import tempfile
import numpy as np
from pathlib import Path

# Mitsuba Renderer
import mitsuba as mi

# Image Processing and Visialisation
import matplotlib.pyplot as plt

# PSNR
from skimage.metrics import peak_signal_noise_ratio

# Custom
from utils.utils import save_tiff
from renderer.SceneRenderer import SceneRenderer



def extract_scene_zip(zip_path):
    """
    Extracts a zip file to a temporary directory

    Parameters:
    - zip_path (str): path to the zip file

    Returns:
    - temp_dir (str): path to the extracted directory
    """
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

def load_scene_xml(scene_dir):
    """
    Loads a scene from .xml file with Mitsuba

    Parameters:
    - scene_dir (str): folder containing files of a scene

    Returns:
    - scene (Mitsuba Scene)
    """
    scene_dir = Path(scene_dir)
    xml_files = [
        f for f in scene_dir.rglob("*.xml")                           # recursive search
        if not f.name.startswith("._") and "__MACOSX" not in str(f)   # filter out metadata files on Mac
    ]

    if not xml_files:
        raise FileNotFoundError(f"No valid XML scene found in {scene_dir}")

    return mi.load_file(str(xml_files[0]))

def plot_images(clean_image, noisy_image, noisy_avg_image):
    """
    NOISY vs. CLEAN Visualisation

    Parameters:
    - clean_image (np array): clean image of shape (H, W, C)
    - noisy_image (np array): noisy image of shape (H, W, C) = average from histograms
    """
    print("Clean Image: ", clean_image.shape)
    print("Average from Renderer: ", noisy_image.shape)
    print("Average from Histogram: ", noisy_avg_image.shape)

    plt.figure(figsize=(12,5))

    # CLEAN IMAGE
    plt.subplot(1,3,1)
    plt.title("Clean Scene")
    plt.imshow(np.clip(clean_image, 0, 1))
    plt.axis('off')

    # AGERAGE FROM RENDERER
    plt.subplot(1,3,2)
    plt.title("Renderer Avg Patch (Low spp)")
    plt.imshow(np.clip(noisy_image, 0, 1))
    plt.axis('off')

    # AGERAGE FROM HISTOGRAM
    plt.subplot(1,3,3)
    plt.title("Histogram Avg Patch (Low spp)")
    plt.imshow(np.clip(noisy_avg_image, 0, 1))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def calculate_psnr_rgb(low_img_tensor, high_img_tensor):
    """Load RGB TIFFs and compute PSNR between low- and high-spp images."""
    # Automatically determine max intensity from high quality reference
    low_img = mi.TensorXf(low_img_tensor).numpy()
    high_img = mi.TensorXf(high_img_tensor).numpy()

    data_range = np.max(high_img)

    psnr = peak_signal_noise_ratio(high_img, low_img, data_range=data_range)
    print(f"PSNR: {psnr} !!!")

def render_scene(scene_path, output_dir, low_spp, high_spp):
    scene = mi.load_file(str(scene_path))
    sensor = scene.sensors()[0]
    integrator = scene.integrator()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get parts for filename
    scene_folder_name = scene_path.parent.name
    xml_name = scene_path.stem  # filename without extension
    print(f"Rendering {scene_path.name} .....")

    # 1 SPP MULTI-RENDER
    renderer = SceneRenderer(scene_path, debug=False)
    low_images = renderer.render_n_images(n=low_spp, spp=1, seed_start=0)
    stack_low = np.stack(low_images, axis=0)
    low_tiff_path = output_dir / f"{scene_folder_name}_{xml_name}_spp1x{low_spp}.tiff"
    save_tiff(stack_low, low_tiff_path)

    # Compute average of the low SPP images for comparison
    img_low_avg = np.mean(stack_low, axis=0)

    # LOW SPP
    img_low = mi.render(scene, sensor=sensor, integrator=integrator, spp=low_spp)
    low_out_path = output_dir / f"{scene_folder_name}_{xml_name}_spp{low_spp}.tiff"
    save_tiff(img_low, low_out_path)

    # HIGH SPP
    img_high = mi.render(scene, sensor=sensor, integrator=integrator, spp=high_spp)
    high_out_path = output_dir / f"{scene_folder_name}_{xml_name}_spp{high_spp}.tiff"
    save_tiff(img_high, high_out_path)
    
    print("Single vs. High")
    calculate_psnr_rgb(img_low, img_high)
    print("Hist vs. High")
    calculate_psnr_rgb(img_low_avg, img_high)
    plot_images(img_high, img_low, img_low_avg)

def render_1ray_scene(xml_file, output_dir, n_images=10, spp=1, base_seed=0, debug=False):
    """
    Renders a scene from an XML file and saves a multi-page TIFF.
    """
    print(f"Rendering scene: {xml_file}")
    
    mi.set_variant('scalar_rgb')
    scene = mi.load_file(str(xml_file))
    
    renderer = SceneRenderer(scene, debug=debug)
    images = renderer.render_n_images(n=n_images, spp=spp, seed_start=base_seed)

    # Construct output filename
    tiff_name = xml_file.stem + "_rendered.tiff"
    output_path = output_dir / tiff_name

    renderer.save_images_to_tiff(images, output_path)

def generate_data(config):
    """
    Process all scene folders inside root_folder.
    Expects scene XML files inside each folder.
    """
    # mitsuba variant
    mi.set_variant(config["mi_variant"])
    print("Using.. ", config["mi_variant"])

    # paths
    root_folder = Path(config['input_path'])
    output_root = Path(config['output_path'])
    output_root.mkdir(parents=True, exist_ok=True)
    
    # get all scene folders
    scene_folders = sorted([d for d in root_folder.iterdir() if d.is_dir()])
    print(f"Found {len(scene_folders)} scene folders in {root_folder}")

    for scene_dir in scene_folders:
        xml_files = sorted(scene_dir.glob("*.xml"))
        if not xml_files:
            print(f"No XML files found in {scene_dir}, skipping.")
            continue
        
        # create output directory
        relative_path = scene_dir.relative_to(root_folder)
        output_dir = output_root / relative_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # render all xml files of a scene
        for xml_file in xml_files:
            render_scene(xml_file, output_dir, low_spp=config['low_spp'], high_spp=config['high_spp'])