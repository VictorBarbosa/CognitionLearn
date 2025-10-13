#!/usr/bin/env python3

"""
Standalone CLI for ML-Agents supervised training.
"""

import argparse
import sys
import os
import yaml
import subprocess
from typing import Dict, Any, Optional

import torch
from mlagents.trainers.torch_entities.model_serialization import ModelSerializer
from mlagents.trainers.supervised_trainer import SupervisedTrainer
from mlagents.trainers.settings import (
    TrainerSettings,
    CheckpointSettings,
    NetworkSettings,
    RunOptions,
)
from mlagents.trainers.supervised_settings import SupervisedLearningSettings
from mlagents.trainers.ppo.optimizer_torch import PPOSettings
from mlagents.trainers.sac.optimizer_torch import SACSettings
from mlagents.trainers.tdsac.optimizer_torch import TDSACSettings

def get_hyperparameters(alg_name: str):
    if alg_name == "ppo":
        return PPOSettings()
    if alg_name == "sac":
        return SACSettings()
    if alg_name == "tdsac":
        return TDSACSettings()
    return PPOSettings() # Default to PPO

def parse_network_settings(net_config: Dict[str, Any]) -> NetworkSettings:
    """
    Parses the network_settings dictionary from the YAML and creates a NetworkSettings object.
    """
    config = net_config.copy()
    
    use_recurrent = bool(config.pop('use_recurrent', False))
    memory_size = int(config.pop('memory_size', 0))
    
    memory_settings = None
    if use_recurrent and memory_size > 0:
        memory_settings = NetworkSettings.MemorySettings(memory_size=memory_size)
        
    config['hidden_units'] = int(config.get('hidden_units', 128))
    config['num_layers'] = int(config.get('num_layers', 2))
    config['normalize'] = bool(config.get('normalize', False))
    config['deterministic'] = True # Force deterministic for supervised learning

    return NetworkSettings(memory=memory_settings, **config)

def main():
    parser = argparse.ArgumentParser(
        description="Standalone supervised training for ML-Agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to the YAML configuration file. Required if not using --generate-yaml."
    )

    parser.add_argument(
        "--generate-yaml",
        "-gy",
        action="store_true",
        help="Starts an interactive wizard to create a YAML configuration file."
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
        help="Algorithm to use (overrides algorithm list in config, trains only for this one)"
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
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)"
    )
    
    parser.add_argument(
        "--use-sequential-model",
        action="store_true",
        help="Use the sequential model instead of the original model"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information during training"
    )
    
    args = parser.parse_args()

    if args.generate_yaml:
        script_path = os.path.join(os.path.dirname(__file__), "create_config.py")
        print("Starting interactive configuration wizard...")
        try:
            subprocess.run([sys.executable, script_path], check=True)
        except FileNotFoundError:
            print(f"Error: The wizard script ('{script_path}') was not found.")
        except subprocess.CalledProcessError as e:
            print(f"Error during wizard execution: {e}")
        return 0

    if not args.config:
        parser.error("The --config argument is required when not using --generate-yaml.")
        return 1
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        return 1
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    data_conf = config.get('data', {})
    model_conf = config.get('model', {})
    training_conf = config.get('training', {})
    output_conf = config.get('output', {})
    additional_conf = config.get('additional_settings', {})

    algorithms_from_config = model_conf.get('algorithm', 'ppo')
    if isinstance(algorithms_from_config, str):
        algorithms = [algorithms_from_config]
    else:
        algorithms = algorithms_from_config

    if args.algorithm:
        print(f"Warning: The --algorithm CLI argument will override the algorithm list from the config file.")
        algorithms = [args.algorithm]
    
    training_algorithm = algorithms[0]
    output_dir = args.output or output_conf.get('dir', './results')
    csv_path = args.csv or data_conf.get('csv_path')
    epochs = args.epochs or training_conf.get('epochs', 10)
    batch_size = args.batch_size or training_conf.get('batch_size', 128)
    learning_rate = args.learning_rate or training_conf.get('learning_rate', 3e-4)
    verbose = args.verbose or additional_conf.get('verbose', False)
    device = additional_conf.get('device', 'cpu')

    from mlagents.torch_utils import set_torch_config
    from mlagents.trainers.settings import TorchSettings
    set_torch_config(TorchSettings(device=device))

    if not csv_path:
        print("Error: Path to CSV file (csv_path) not specified in config or via CLI.")
        return 1

    behavior_name = "SupervisedBehavior"
    obs_cols = data_conf.get('observation_columns', [])
    action_cols = data_conf.get('action_columns', [])

    supervised_settings = SupervisedLearningSettings(
        csv_path=csv_path,
        observation_columns=obs_cols,
        action_columns=action_cols,
        observation_shape=[len(obs_cols)],
        action_size=len(action_cols),
        num_epoch=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        checkpoint_interval=int(output_conf.get('checkpoint_interval', 1000)),
        validation_split=float(training_conf.get('validation_split', 0.2)),
        shuffle=bool(training_conf.get('shuffle', True)),
        augment_noise=float(training_conf.get('augment_noise', 0.01)),
        early_stopping=bool(training_conf.get('early_stopping', True)),
        patience=int(training_conf.get('patience', 5)),
        min_delta=float(training_conf.get('min_delta', 0.001)),
        dropout_rate=float(training_conf.get('dropout_rate', 0.1)),
        weight_decay=float(training_conf.get('weight_decay', 1e-4)),
        lr_patience=int(training_conf.get('lr_patience', 5)),
    )

    behavior_config = TrainerSettings(
        trainer_type=training_algorithm,
        hyperparameters=get_hyperparameters(training_algorithm),
        network_settings=parse_network_settings(model_conf.get('network_settings', {})),
        supervised=supervised_settings,
        max_steps=500000
    )

    run_options = RunOptions(
        behaviors={behavior_name: behavior_config},
        checkpoint_settings=CheckpointSettings(run_id=output_dir)
    )

    from mlagents.plugins.stats_writer import register_stats_writer_plugins
    from mlagents.trainers.stats import StatsReporter

    StatsReporter.writers.clear()
    stats_writers = register_stats_writer_plugins(run_options)
    for sw in stats_writers:
        StatsReporter.add_writer(sw)

    try:
        trainer = SupervisedTrainer(
            behavior_name=behavior_name,
            behavior_config=behavior_config,
            run_options=run_options,
            all_target_algorithms=algorithms, # Pass the full list
            use_sequential_model=args.use_sequential_model,
            verbose=verbose
        )
        trainer.train()
        print("\nSupervised training completed successfully!\n")
        return 0
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
