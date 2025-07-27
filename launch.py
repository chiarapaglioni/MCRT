import yaml
import argparse
from pathlib import Path
from utils.utils import setup_logger

from renderer.RenderingPipeline import generate_data
from dataset.HistogramLoader import test_data_loader
from model.DenoisingPipeline import train_model, evaluate_model, train_n2n, test_n2n
from model.GenerativePipeline import train_histogram_generator, iterative_evaluate, run_generative_accumulation_pipeline, test_histogram_generator, train_histogram_residual

logger = setup_logger()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # PARSE CONFIG
    parser = argparse.ArgumentParser(description="MCRT Pipeline Launcher")
    parser.add_argument("task", type=str, choices=["data_gen", "data_loader", "train", "eval", "train_gen", "eval_gen"], help="Task to run.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")

    parser.add_argument('-t', '--train-dir', help='training set path', default='./../data/train')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='./../data/valid')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../ckpts')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--report-interval', help='batch report interval', default=500, type=int)
    parser.add_argument('-ts', '--train-size', help='size of train dataset', type=int)
    parser.add_argument('-vs', '--valid-size', help='size of valid dataset', type=int)

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2', 'hdr'], default='hdr', type=str)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type', choices=['gaussian', 'poisson', 'text', 'mc'], default='mc', type=str)
    parser.add_argument('-p', '--noise-param', help='noise parameter (e.g. std for gaussian)', default=50, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=128, type=int)
    parser.add_argument('--clean-targets', help='use clean targets for training', action='store_true')
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
        # train_n2n(config)
    
    # EVAL (OK) <3
    elif task == "eval":
        evaluate_model(config)
        # test_n2n(config)

    # GEN-TRAIN (OK) <3
    elif task == "train_gen":
        # train_histogram_generator(config)                       # Train x dist
        train_histogram_residual(config)                        # Train x residual
    
    # GEN-EVAL (OK) <3
    elif task == "eval_gen":
        iterative_evaluate(config)
        # run_generative_accumulation_pipeline(config)          # GAP
        # test_histogram_generator(config)                      # noisy vs. pred

if __name__ == "__main__":
    main()
