import argparse
import yaml
from pathlib import Path

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # PARSE CONFIG
    parser = argparse.ArgumentParser(description="MCRT Pipeline Launcher")
    parser.add_argument("task", type=str, choices=["data_gen", "data_loader", "train", "eval"],
                        help="Task to run.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    args = parser.parse_args()

    task = args.task
    config_path = Path(args.config) if args.config else Path(f"config/{task}.yml")
    config = load_config(config_path)

    # RENDERING DATA GENERATION (OK) <3
    if task == "data_gen":
        from renderer.RenderingPipeline import generate_data
        generate_data(config)

    # DATA LOADER (OK) <3
    elif task == "data_loader":
        from dataset.HistogramLoader import test_data_loader
        test_data_loader(config)

    # TODO: TRAIN
    elif task == "train":
        from model.DenoisingPipeline import train_model
        train_model(config)
    
    # TODO: EVAL
    elif task == "eval":
        from model.DenoisingPipeline import evaluate_model
        evaluate_model(config)

if __name__ == "__main__":
    main()
