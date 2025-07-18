import yaml
import argparse
from pathlib import Path
from utils.utils import setup_logger

from renderer.RenderingPipeline import generate_data
from dataset.HistogramLoader import test_data_loader
from model.DenoisingPipeline import train_model, evaluate_model
from model.GenerativePipeline import train_histogram_generator, iterative_evaluate

logger = setup_logger()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # PARSE CONFIG
    parser = argparse.ArgumentParser(description="MCRT Pipeline Launcher")
    parser.add_argument("task", type=str, choices=["data_gen", "data_loader", "train", "eval", "train_gen", "eval_gen"],
                        help="Task to run.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    args = parser.parse_args()

    logger.info("MCRT DENOISING :)")

    task = args.task
    config_path = Path(args.config) if args.config else Path(f"config/{task}.yml")
    config = load_config(config_path)

    # RENDERING DATA GENERATION (OK) <3
    if task == "data_gen":
        generate_data(config)

    # DATA LOADER (OK) <3
    elif task == "data_loader":
        test_data_loader(config)

    # TRAIN (OK) <3
    elif task == "train":
        train_model(config)
    
    # EVAL (OK) <3
    elif task == "eval":
        evaluate_model(config)

    # GEN-TRAIN (OK) <3
    elif task == "train_gen":
        train_histogram_generator(config)
    
    # TODO: EVAL (generative)
    elif task == "eval_gen":
        iterative_evaluate(config)

if __name__ == "__main__":
    main()
