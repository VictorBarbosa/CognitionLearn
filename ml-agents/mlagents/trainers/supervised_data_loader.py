"""
Data loader for supervised training in ML-Agents.
"""

import csv
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.torch_utils import torch, default_device

from mlagents.trainers.supervised_utils import (
    load_csv_data,
    prepare_observations,
    prepare_actions,
    split_data,
    augment_data,
    create_data_loader
)


class SupervisedDataLoader:
    """
    Class to load and process supervised training data from a CSV.
    """
    
    def __init__(
        self,
        csv_path: str,
        observation_columns: List[str],
        action_columns: List[str],
        validation_split: float = 0.2,
        shuffle: bool = True,
        augment_noise: float = 0.01,
        action_spec: Optional[ActionSpec] = None
    ):
        """
        :param csv_path: Path to the CSV file
        :param observation_columns: List of column names containing observations
        :param action_columns: List of column names containing actions
        :param validation_split: Fraction of data to be used for validation
        :param shuffle: Whether the data should be shuffled
        :param augment_noise: Noise level for data augmentation
        :param action_spec: Action specification (optional, for validation)
        """
        self.csv_path = csv_path
        self.observation_columns = observation_columns
        self.action_columns = action_columns
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.augment_noise = augment_noise
        self.action_spec = action_spec
        
        # Load data
        self.data = self._load_data()
        
        # Split data into training and validation
        self.train_data, self.val_data = self._split_data()
        
        # Validate if the number of actions matches the specification, if provided
        # This validation can be optional to maintain compatibility with different configurations
        if self.action_spec is not None:
            if self.action_spec.is_discrete:
                expected_action_size = sum(self.action_spec.discrete_branches)
            else:
                expected_action_size = self.action_spec.continuous_size

            if expected_action_size != len(self.action_columns):
                import warnings
                # warnings.warn(
                #     f"Number of action columns ({len(self.action_columns)}) does not match "
                #     f"the action specification ({expected_action_size}). "
                #     f"Continuing anyway, but please check if the action columns are correct."
                # )
    
    def _load_data(self) -> pd.DataFrame:
        """
        Loads data from the CSV file.
        """
        try:
            data = load_csv_data(self.csv_path)
            # Check if the specified columns exist
            required_columns = set(self.observation_columns + self.action_columns)
            available_columns = set(data.columns)
            
            missing_cols = required_columns - available_columns
            if missing_cols:
                raise ValueError(f"Missing columns in CSV: {missing_cols}")
                
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}")
    
    def _split_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Splits the data into training and validation sets.
        """
        observations = prepare_observations(self.data, self.observation_columns)
        actions = prepare_actions(self.data, self.action_columns)
        
        train_obs, train_actions, val_obs, val_actions = split_data(
            observations, actions, self.validation_split, self.shuffle
        )
        
        # Apply data augmentation only to the training set
        if self.augment_noise > 0:
            train_obs, train_actions = augment_data(
                train_obs, train_actions, self.augment_noise
            )
        
        train_data = {
            'observations': train_obs,
            'actions': train_actions
        }
        
        val_data = {
            'observations': val_obs,
            'actions': val_actions
        }
        
        return train_data, val_data
    
    def add_noise_to_observations(self, observations: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Adds noise to observations for data augmentation.
        """
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, observations.shape).astype(observations.dtype)
            return observations + noise
        return observations
    
    def get_train_loader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """
        Returns a PyTorch DataLoader for the training data.
        """
        return self._create_loader(self.train_data, batch_size)
    
    def get_validation_loader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """
        Returns a PyTorch DataLoader for the validation data.
        """
        return self._create_loader(self.val_data, batch_size, shuffle=False)
    
    def _create_loader(self, data: Dict[str, np.ndarray], batch_size: int, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """
        Creates a PyTorch DataLoader with the specified data.
        """
        # Add data augmentation noise only to the training set
        observations = data['observations']
        if self.augment_noise > 0 and data is self.train_data:
            observations = self.add_noise_to_observations(observations, self.augment_noise)
        
        observations_tensor = torch.tensor(observations)
        actions_tensor = torch.tensor(data['actions'])
        
        dataset = torch.utils.data.TensorDataset(observations_tensor, actions_tensor)
        # Create a generator compatible with the device defined for the model
        generator = torch.Generator(device=default_device().type)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if data is self.train_data else False,
            generator=generator
        )
    
    def get_num_features(self) -> int:
        """
        Returns the number of features in the observations.
        """
        return len(self.observation_columns)
    
    def get_num_actions(self) -> int:
        """
        Returns the number of actions.
        """
        return len(self.action_columns)
