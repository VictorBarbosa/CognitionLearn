"""
Utilities for supervised training in ML-Agents.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch import nn
from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.torch_utils import torch as torch_module, default_device


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file for supervised training.
    :param csv_path: Path to the CSV file
    :return: DataFrame with the loaded data
    """
    try:
        data = pd.read_csv(csv_path)
        print(f"[SUPERVISED] Data loaded: {len(data)} samples")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    except Exception as e:
        raise ValueError(f"Error loading CSV: {str(e)}")


def prepare_observations(data: pd.DataFrame, observation_columns: List[str]) -> np.ndarray:
    """
    Prepares observations from the CSV data.
    :param data: DataFrame with the data
    :param observation_columns: Observation columns
    :return: Numpy array with observations
    """
    try:
        observations = data[observation_columns].values.astype(np.float32)
        print(f"[SUPERVISED] Observations prepared: {observations.shape}")
        return observations
    except KeyError as e:
        raise KeyError(f"Observation columns not found: {e}")
    except Exception as e:
        raise ValueError(f"Error preparing observations: {str(e)}")


def prepare_actions(data: pd.DataFrame, action_columns: List[str]) -> np.ndarray:
    """
    Prepares actions from the CSV data.
    :param data: DataFrame with the data
    :param action_columns: Action columns
    :return: Numpy array with actions
    """
    try:
        actions = data[action_columns].values.astype(np.float32)
        print(f"[SUPERVISED] Actions prepared: {actions.shape}")
        return actions
    except KeyError as e:
        raise KeyError(f"Action columns not found: {e}")
    except Exception as e:
        raise ValueError(f"Error preparing actions: {str(e)}")


def split_data(
    observations: np.ndarray, 
    actions: np.ndarray, 
    validation_split: float = 0.2,
    shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the data into training and validation sets.
    :param observations: Array of observations
    :param actions: Array of actions
    :param validation_split: Fraction of data for validation
    :param shuffle: Whether the data should be shuffled
    :return: Tuple with (train_obs, train_actions, val_obs, val_actions)
    """
    try:
        n_samples = len(observations)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(indices)
        
        n_val = int(n_samples * validation_split)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_obs = observations[train_indices]
        train_actions = actions[train_indices]
        val_obs = observations[val_indices]
        val_actions = actions[val_indices]
        
        print(f"[SUPERVISED] Data split - Train: {len(train_obs)}, Validation: {len(val_obs)}")
        return train_obs, train_actions, val_obs, val_actions
    except Exception as e:
        raise ValueError(f"Error splitting data: {str(e)}")


def augment_data(
    observations: np.ndarray, 
    actions: np.ndarray, 
    noise_level: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adds noise to the data for augmentation.
    :param observations: Array of observations
    :param actions: Array of actions
    :param noise_level: Noise level
    :return: Tuple with (augmented_obs, augmented_actions)
    """
    try:
        if noise_level > 0:
            obs_noise = np.random.normal(0, noise_level, observations.shape).astype(observations.dtype)
            action_noise = np.random.normal(0, noise_level, actions.shape).astype(actions.dtype)
            
            augmented_obs = observations + obs_noise
            augmented_actions = actions + action_noise
            
            print(f"[SUPERVISED] Data augmented with noise level {noise_level}")
            return augmented_obs, augmented_actions
        else:
            return observations, actions
    except Exception as e:
        raise ValueError(f"Error augmenting data: {str(e)}")


def create_data_loader(
    observations: np.ndarray,
    actions: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = True
) -> torch_module.utils.data.DataLoader:
    """
    Creates a PyTorch DataLoader with the data.
    :param observations: Array of observations
    :param actions: Array of actions
    :param batch_size: Batch size
    :param shuffle: Whether the data should be shuffled
    :return: PyTorch DataLoader
    """
    try:
        observations_tensor = torch_module.tensor(observations)
        actions_tensor = torch_module.tensor(actions)
        
        dataset = torch_module.utils.data.TensorDataset(observations_tensor, actions_tensor)
        # Create a generator compatible with the device defined for the model
        generator = torch_module.Generator(device=default_device().type)
        data_loader = torch_module.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator
        )
        
        print(f"[SUPERVISED] DataLoader created with batch_size {batch_size}")
        return data_loader
    except Exception as e:
        raise ValueError(f"Error creating DataLoader: {str(e)}")


def prepare_observations_for_model(observations: torch_module.Tensor) -> List[torch_module.Tensor]:
    """
    Prepares observations for input into the ML-Agents model.
    :param observations: Tensor of observations
    :return: List of observation tensors
    """
    try:
        # Flattening and concatenating observations
        if len(observations.shape) == 2:
            # Simple case: batch_size x features
            return [observations]
        elif len(observations.shape) == 3:
            # Case with multiple observations: batch_size x num_obs x features
            return [observations[:, i, :] for i in range(observations.shape[1])]
        else:
            # Generic case: flattening and reconstructing
            return [observations.view(observations.size(0), -1)]
    except Exception as e:
        raise ValueError(f"Error preparing observations for model: {str(e)}")


def move_model_to_device(model: nn.Module, device: torch_module.device) -> nn.Module:
    """
    Moves the model to the specified device.
    :param model: PyTorch model
    :param device: Device
    :return: Model moved to the device
    """
    try:
        model.to(device)
        print(f"[SUPERVISED] Model moved to device: {device}")
        return model
    except Exception as e:
        raise ValueError(f"Error moving model to device: {str(e)}")


def restore_model_to_original_device(model: nn.Module, original_device: torch_module.device) -> nn.Module:
    """
    Restores the model to the original device.
    :param model: PyTorch model
    :param original_device: Original device
    :return: Model restored to the original device
    """
    try:
        model.to(original_device)
        print(f"[SUPERVISED] Model restored to original device: {original_device}")
        return model
    except Exception as e:
        raise ValueError(f"Error restoring model to original device: {str(e)}")
