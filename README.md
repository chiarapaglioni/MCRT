# MCRT Denoising

## Project Structure
```
MCRT/
├── data/
├── scripts/
├── model/
├── mcrt.yml
```

where: 
- **data**: raw data (_XML_) of the scenes/images to be rendered
- **scripts**: jupyter notebooks to render scenes in low/high res (_TIFF_) and generate histograms (_NPZ_)
- **model**: binomial split dataset and denoising model

## Setup
Requirements:
- **Python**: 3.9
- **Anaconda** or **Miniconda**

```bash
# Create the environment from the YAML file
conda env create -f mcrt.yml

# Activate the environment
conda activate mcrt
```