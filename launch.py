import yaml
import argparse
from pathlib import Path
from utils.utils import setup_logger, plot_experiments

from renderer.RenderingPipeline import generate_data
from dataset.HistogramLoader import test_data_loader
from model.DenoisingPipeline import train_model, evaluate_model, benchmark_num_workers, plot_all_model_predictions, plot_hist_model_predictions
from model.GenerativePipeline import train_histogram_generator, iterative_evaluate, run_generative_accumulation_pipeline, test_histogram_generator

logger = setup_logger()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # PARSE CONFIG
    parser = argparse.ArgumentParser(description="MCRT Pipeline Launcher")
    parser.add_argument("task", type=str, choices=["data_gen", "data_loader", "train", "eval", "train_gen", "eval_gen", "test_workers", "experiments", "experiments_plot", "experiments_hist"], help="Task to run.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    args = parser.parse_args()

    logger.info("MCRT DENOISING :)")

    task = args.task
    config_path = Path(args.config) if args.config else Path(f"config/{task}.yml")
    config = load_config(config_path)

    # RENDERING DATA GENERATION (OK) 
    if task == "data_gen":
        generate_data(config)

    # DATA LOADER (OK) 
    elif task == "data_loader":
        test_data_loader(config)

    # TRAIN (OK) 
    elif task == "train":
        train_model(config)
    
    # EVAL (OK) 
    elif task == "eval":
        evaluate_model(config)

    # GEN-TRAIN (OK) 
    elif task == "train_gen":
        train_histogram_generator(config)                       # Train x dist
        # train_histogram_residual(config)                      # Train x residual
    
    # GEN-EVAL (OK) 
    elif task == "eval_gen":
        # iterative_evaluate(config)
        # run_generative_accumulation_pipeline(config)          # GAP
        test_histogram_generator(config)                        # noisy vs. pred

    # GPU TEST (OK) 
    elif task == "test_workers":
        benchmark_num_workers(config)

    # EXPERIMENTS (OK)
    elif task == "experiments":
        plot_experiments(config)

    elif task == "experiments_plot":
        plot_all_model_predictions(config)

    elif task == "experiments_hist":
        plot_hist_model_predictions(config)

if __name__ == "__main__":
    main()
