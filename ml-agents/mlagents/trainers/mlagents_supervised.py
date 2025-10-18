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
from mlagents.trainers.td3.optimizer_torch import TD3Settings
from mlagents.trainers.tqc.optimizer_torch import TQCSettings
from mlagents.trainers.dcac.optimizer_torch import DCACSettings
from mlagents.trainers.crossq.optimizer_torch import CrossQSettings
from mlagents.trainers.drqv2.optimizer_torch import DrQv2Settings
from mlagents.trainers.ppo_et.settings import PPOETSettings
from mlagents.trainers.ppo_ce.settings import PPOCESettings
from mlagents.trainers.sac_ae.settings import SACAESettings
from mlagents_envs.base_env import ActionSpec, ObservationSpec, DimensionProperty

def convert_csv_to_demo(args):
    """
    Converts CSV file to .demo format.
    
    :param args: Parsed command line arguments
    """
    if not args.demo:
        print("Error: --demo argument requires a CSV file path.")
        return 1
    
    # Check if CSV file exists
    if not os.path.exists(args.demo):
        print(f"Error: CSV file not found: {args.demo}")
        return 1
    
    # Get other required arguments for conversion
    if not args.config:
        print("Error: --config argument is required for CSV to .demo conversion.")
        return 1
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    data_conf = config.get('data', {})
    model_conf = config.get('model', {})
    
    csv_path = args.demo
    observation_columns = data_conf.get('observation_columns', [])
    action_columns = data_conf.get('action_columns', [])
    
    if not observation_columns or not action_columns:
        print("Error: Both observation_columns and action_columns must be specified in the config file.")
        return 1
    
    # Determine output path for .demo file
    demo_path = args.output or os.path.splitext(csv_path)[0] + ".demo"
    
    # Determine action specification from model config
    algorithm = model_conf.get('algorithm', 'ppo')
    action_spec = get_action_spec_from_algorithm(algorithm, config)
    
    try:
        # Import the conversion module
        from mlagents.trainers.csv_to_demo_converter import convert_csv_to_demo
        
        # Perform the conversion
        convert_csv_to_demo(
            csv_path=csv_path,
            demo_path=demo_path,
            observation_columns=observation_columns,
            action_columns=action_columns,
            action_spec=action_spec,
            brain_name="SupervisedBrain",
            demonstration_name=os.path.basename(csv_path)
        )
        
        print(f"Successfully converted {csv_path} to {demo_path}")
        return 0
    except Exception as e:
        print(f"Error during CSV to .demo conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


def get_action_spec_from_algorithm(algorithm: str, config: Dict[str, Any]) -> ActionSpec:
    """
    Creates an ActionSpec based on the algorithm and configuration.
    
    :param algorithm: Algorithm name
    :param config: Configuration dictionary
    :return: ActionSpec object
    """
    data_conf = config.get('data', {})
    model_conf = config.get('model', {})
    action_columns = data_conf.get('action_columns', [])
    
    # Default to continuous actions
    continuous_size = len(action_columns)
    discrete_branches = []
    
    # Special cases for algorithms that might use discrete actions
    # This would be determined by the actual environment specification
    # For now, we'll use heuristics based on algorithm names and configuration
    
    # Algorithms typically using discrete actions
    discrete_algorithms = ['ppo_ce']  # Curiosity-based algorithms sometimes use discrete actions
    
    if algorithm in discrete_algorithms:
        # Try to determine discrete branches from configuration or data
        # For simplicity, we'll assume a single discrete branch with the same number of actions
        continuous_size = 0
        discrete_branches = [max(2, len(action_columns))]  # Assume at least 2 actions per branch
    
    # Override with explicit configuration if provided
    action_spec_conf = model_conf.get('action_spec', {})
    if action_spec_conf:
        continuous_size = action_spec_conf.get('continuous_size', continuous_size)
        discrete_branches = action_spec_conf.get('discrete_branches', discrete_branches)
    
    return ActionSpec(
        continuous_size=continuous_size,
        discrete_branches=tuple(discrete_branches)
    )


def get_action_spec_from_algorithm(algorithm: str, config: Dict[str, Any]) -> ActionSpec:
    """
    Creates an ActionSpec based on the algorithm and configuration.
    
    :param algorithm: Algorithm name
    :param config: Configuration dictionary
    :return: ActionSpec object
    """
    data_conf = config.get('data', {})
    action_columns = data_conf.get('action_columns', [])
    
    # Default to continuous actions
    continuous_size = len(action_columns)
    discrete_size = 0
    discrete_branches = []
    
    # Special cases for algorithms that might use discrete actions
    # Algorithms typically using discrete actions
    discrete_algorithms = ['ppo_ce']  # Curiosity-based algorithms sometimes use discrete actions
    
    if algorithm in discrete_algorithms:
        # For discrete actions, we'll assume a single discrete branch
        continuous_size = 0
        discrete_size = len(action_columns)
        discrete_branches = [max(2, len(action_columns))]  # Assume at least 2 actions per branch
    
    # Override with explicit configuration if provided
    model_conf = config.get('model', {})
    action_spec_conf = model_conf.get('action_spec', {})
    if action_spec_conf:
        continuous_size = action_spec_conf.get('continuous_size', continuous_size)
        discrete_size = action_spec_conf.get('discrete_size', discrete_size)
        discrete_branches = action_spec_conf.get('discrete_branches', discrete_branches)
    
    return ActionSpec(
        continuous_size=continuous_size,
        discrete_branches=tuple(discrete_branches)
    )


def convert_csv_to_demo(args):
    """
    Converts CSV file to .demo format.
    
    :param args: Parsed command line arguments
    """
    if not args.demo:
        print("Error: --demo argument requires a CSV file path.")
        return 1
    
    # Check if CSV file exists
    if not os.path.exists(args.demo):
        print(f"Error: CSV file not found: {args.demo}")
        return 1
    
    # Get other required arguments for conversion
    if not args.config:
        print("Error: --config argument is required for CSV to .demo conversion.")
        return 1
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    data_conf = config.get('data', {})
    model_conf = config.get('model', {})
    
    csv_path = args.demo
    observation_columns = data_conf.get('observation_columns', [])
    action_columns = data_conf.get('action_columns', [])
    
    if not observation_columns or not action_columns:
        print("Error: Both observation_columns and action_columns must be specified in the config file.")
        return 1
    
    # Determine output path for .demo file
    demo_path = args.output or os.path.splitext(csv_path)[0] + ".demo"
    
    # Determine action specification from model config
    algorithm = model_conf.get('algorithm', 'ppo')
    action_spec = get_action_spec_from_algorithm(algorithm, config)
    
    try:
        # Import the conversion module
        from mlagents.trainers.csv_to_demo_converter import convert_csv_to_demo as do_convert
        
        # Perform the conversion
        do_convert(
            csv_path=csv_path,
            demo_path=demo_path,
            observation_columns=observation_columns,
            action_columns=action_columns,
            action_spec=action_spec,
            brain_name="SupervisedBrain",
            demonstration_name=os.path.basename(csv_path)
        )
        
        print(f"Successfully converted {csv_path} to {demo_path}")
        return 0
    except Exception as e:
        print(f"Error during CSV to .demo conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


def get_hyperparameters(alg_name: str):
    if alg_name == "ppo":
        return PPOSettings()
    if alg_name == "sac":
        return SACSettings()
    if alg_name == "tdsac":
        return TDSACSettings()
    if alg_name == "td3":
        return TD3Settings()
    if alg_name == "tqc":
        return TQCSettings()
    if alg_name == "dcac":
        return DCACSettings()
    if alg_name == "crossq":
        return CrossQSettings()
    if alg_name == "drqv2":
        return DrQv2Settings()
    if alg_name == "ppo_et":
        return PPOETSettings()
    if alg_name == "ppo_ce":
        return PPOCESettings()
    if alg_name == "sac_ae":
        return SACAESettings()
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
        description="Standalone supervised training for ML-Agents. Can also convert CSV files to .demo format.",
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
        "--demo",
        "-d",
        type=str,
        help="Path to the CSV file to convert to .demo format. Generates a .demo file for use with Unity."
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory or file path for trained models or generated .demo files (overrides config)"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to the CSV file with training data (overrides config)"
    )
    
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ppo", "sac", "tdsac", "td3", "tqc", "dcac", "crossq", "drqv2", "ppo_et", "ppo_ce", "sac_ae"],
        help="Algorithm to use (overrides algorithm list in config, trains only for this one)"
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

    if args.demo:
        # Handle CSV to .demo conversion
        print("Converting CSV to .demo format for Unity ML-Agents...")
        return convert_csv_to_demo(args)

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
