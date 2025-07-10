# MCRT Denoising

## Project Structure
```
MCRT/
├── data/
├── scripts/
├── model/
├── output/
├── config/
├── launch.py
├── mcrt.yml
```

where: 
- **data**: raw data (_XML_) of the scenes/images to be rendered
- **scripts**: jupyter notebooks to render scenes in low/high res (_TIFF_) and generate histograms (_NPZ_)
- **model**: binomial split dataset and denoising model
- **output**: folder containing the output (_TIFF_) files
- **config**: folder containing the configurations for the various tasks
- **launch.py**: main file to run the project

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

## How to Run
From project root: 
```bash
python launch.py data_gen
python launch.py data_loader
python launch.py train
python launch.py eval
```