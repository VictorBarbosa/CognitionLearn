#!/usr/bin/env python3

"""
Main entry point for the standalone supervised learning tool for ML-Agents.
"""

import argparse
import sys
import os

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from standalone_supervised.trainer import SupervisedTrainer
from standalone_supervised.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Supervised Learning for ML-Agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to the CSV file with training data (overrides config)"
    )
    
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ppo", "sac", "tdsac"],
        help="Algorithm to use (overrides config)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory for trained models (overrides config)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training (overrides config)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for training (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.csv:
        config["data"]["csv_path"] = args.csv
    if args.algorithm:
        config["model"]["algorithm"] = args.algorithm
    if args.output:
        config["output"]["dir"] = args.output
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    
    # Validate configuration
    if "data" not in config or "csv_path" not in config["data"]:
        print("Error: CSV path must be specified in config or command line")
        return 1
    
    if "model" not in config or "algorithm" not in config["model"]:
        print("Error: Algorithm must be specified in config or command line")
        return 1
    
    if "output" not in config or "dir" not in config["output"]:
        print("Error: Output directory must be specified in config or command line")
        return 1
    
    # Create trainer and start training
    trainer = SupervisedTrainer(config)
    trainer.train()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())