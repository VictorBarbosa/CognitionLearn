"""
Configuration loader for the standalone supervised learning tool.
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    :param config_path: Path to the YAML configuration file
    :return: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set default values if not present
    if "data" not in config:
        config["data"] = {}
    
    if "model" not in config:
        config["model"] = {}
    
    if "training" not in config:
        config["training"] = {
            "epochs": 10,
            "batch_size": 128,
            "learning_rate": 3e-4,
            "validation_split": 0.2,
            "shuffle": True,
            "augment_noise": 0.01
        }
    
    if "output" not in config:
        config["output"] = {
            "dir": "./results",
            "checkpoint_interval": 1000
        }
    
    return config