import yaml
import argparse
from pathlib import Path
from utils.utils import setup_logger, plot_experiments

from renderer.RenderingPipeline import generate_data
from dataset.HistogramLoader import test_data_loader
from model.DenoisingPipeline import train_model, evaluate_model, evaluate_model_aov, benchmark_num_workers
from model.GenerativePipeline import train_histogram_generator, iterative_evaluate, run_generative_accumulation_pipeline, test_histogram_generator
from model.AdaptiveSampling import run_adaptive_sampling

logger = setup_logger()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # PARSE CONFIG
    parser = argparse.ArgumentParser(description="MCRT Pipeline Launcher")
    parser.add_argument("task", type=str, choices=["data_gen", "data_loader", "train", "eval", "train_gen", "eval_gen", "test_workers", "adaptive_sampling", "experiments"], help="Task to run.")
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
        # evaluate_model(config)
        evaluate_model_aov(config)

    # GEN-TRAIN (OK) <3
    elif task == "train_gen":
        train_histogram_generator(config)                       # Train x dist
        # train_histogram_residual(config)                      # Train x residual
    
    # GEN-EVAL (OK) <3
    elif task == "eval_gen":
        # iterative_evaluate(config)
        run_generative_accumulation_pipeline(config)            # GAP
        # test_histogram_generator(config)                      # noisy vs. pred

    # GPU TEST (OK) <3
    elif task == "test_workers":
        benchmark_num_workers(config)

    # ADAPTIVE SAMPLING
    elif task == "adaptive_sampling":
        run_adaptive_sampling(config)

    elif task == "experiments":
        plot_experiments(config)

if __name__ == "__main__":
    main()
