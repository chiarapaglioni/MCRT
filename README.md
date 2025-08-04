# MCRT Denoising

## Project Structure
```
MCRT/
├── data/
├── dataset/
├── scripts/
├── model/
├── renderer/
├── output/
├── utils/
├── config/
├── launch.py
├── mcrt.yml
```

where: 
- **data**: raw data (_XML_) of the scenes/images to be rendered
- **dataset**: implementation of custom dataloader class to load and process image and histogram data
- **scripts**: jupyter notebooks to render scenes in low/high res (_TIFF_) and generate histograms (_NPZ_)
- **model**: binomial split dataset and denoising model
- **renderer**: functions used to render noisy and clean images using _Mitsuba3_ (supports both CPU and GPU)
- **output**: folder containing the output (_TIFF_) files
- **utils**: contains utils functions for tone mapping, logging and plotting
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
python launch.py train --config config/train_n2n.yml                # for Noise2Noise (IMG mode)
python launch.py train --config config/train_n2n_stat.yml           # for Noise2Noise (STAT mode)
python launch.py train --config config/train_h2n.yml                # for Hist2Noise
python launch.py eval
python launch.py train_gen --config config/train_h2h.yml            # for Hist2Hist (HIST mode)
python launch.py train_gen --config config/train_h2h_stat.yml       # for Hist2Hist (STAT mode)
python launch.py eval_gen
python launch.py test_workers --config config/test_cpu.yml          # to be fixed
```