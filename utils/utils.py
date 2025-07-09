import tifffile

def save_tiff(data, file_name):
    """
    Saves data of shape (N, H, W, C, B) to TIFF file using BigTIFF if needed.

    Parameters:
    - data (np array): data to save
    - file_name (str): file name / scene name
    """
    tifffile.imwrite(file_name, data, compression='lzw', bigtiff=True)
    print(f"Saved {file_name} with shape {data.shape} <3")