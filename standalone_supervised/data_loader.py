"""
Data loader for supervised learning from CSV files.
"""

import csv
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from mlagents_envs.base_env import ActionSpec, ObservationSpec, ObservationType
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.torch_utils import torch, default_device
from torch.utils.data import Dataset, DataLoader


class SupervisedDataset(Dataset):
    """
    Dataset class for supervised learning data.
    """
    
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        augment_noise: float = 0.0
    ):
        """
        :param observations: Array de observações
        :param actions: Array de ações
        :param augment_noise: Nível de ruído para aumentação de dados
        """
        self.observations = observations
        self.actions = actions
        self.augment_noise = augment_noise
    
    def __len__(self) -> int:
        return len(self.observations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = self.observations[idx]
        action = self.actions[idx]
        
        # Aplicar ruído de aumentação de dados se necessário
        if self.augment_noise > 0:
            obs = obs + np.random.normal(0, self.augment_noise, obs.shape).astype(obs.dtype)
        
        return torch.tensor(obs, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)


class SupervisedDataLoader:
    """
    Class to load and process supervised learning data from a CSV file.
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
        :param observation_columns: List of observation column names
        :param action_columns: List of action column names
        :param validation_split: Fraction of data to use for validation
        :param shuffle: Whether to shuffle the data
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
        
        # Split data into train and validation
        self.train_data, self.val_data = self._split_data()
        
        # Create observation and action specs
        self.observation_specs = self._create_observation_specs()
        self.action_spec = self._create_action_spec()
        
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
                #     f"action specification ({expected_action_size}). "
                #     f"Continuing anyway, but check if action columns are correct."
                # )
    
    def _load_data(self) -> pd.DataFrame:
        """
        Loads data from the CSV file.
        """
        try:
            data = pd.read_csv(self.csv_path)
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
        Splits data into training and validation sets.
        """
        if self.shuffle:
            self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n_samples = len(self.data)
        n_val = int(n_samples * self.validation_split)
        
        val_indices = self.data.index[:n_val].tolist()
        train_indices = self.data.index[n_val:].tolist()
        
        # Separate observations and actions
        train_data = {
            'observations': self.data.loc[train_indices, self.observation_columns].values.astype(np.float32),
            'actions': self.data.loc[train_indices, self.action_columns].values.astype(np.float32)
        }
        
        val_data = {
            'observations': self.data.loc[val_indices, self.observation_columns].values.astype(np.float32),
            'actions': self.data.loc[val_indices, self.action_columns].values.astype(np.float32)
        }
        
        return train_data, val_data
    
    def _create_observation_specs(self) -> List[ObservationSpec]:
        """
        Creates observation specifications based on the data.
        """
        # For simplicity, we assume all observations are vector observations
        # In a more complex scenario, we might need to handle different observation types
        obs_size = len(self.observation_columns)
        return [ObservationSpec(ObservationType.DEFAULT, (obs_size,))]
    
    def _create_action_spec(self) -> ActionSpec:
        """
        Creates action specifications based on the data.
        """
        # Assume continuous actions for now
        # In a more complex scenario, we might need to determine if actions are discrete
        action_size = len(self.action_columns)
        return ActionSpec.create_continuous_action_spec(action_size)
    
    def add_noise_to_observations(self, observations: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Adds noise to observations for data augmentation.
        """
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, observations.shape).astype(observations.dtype)
            return observations + noise
        return observations
    
    def get_train_loader(self, batch_size: int) -> DataLoader:
        """
        Returns a PyTorch DataLoader for training data.
        """
        return self._create_loader(self.train_data, batch_size)
    
    def get_validation_loader(self, batch_size: int) -> DataLoader:
        """
        Returns a PyTorch DataLoader for validation data.
        """
        return self._create_loader(self.val_data, batch_size, shuffle=False)
    
    def _create_loader(self, data: Dict[str, np.ndarray], batch_size: int, shuffle: bool = True) -> DataLoader:
        """
        Creates a PyTorch DataLoader with the specified data.
        """
        # Add data augmentation noise only to training data
        observations = data['observations']
        if self.augment_noise > 0 and data is self.train_data:
            observations = self.add_noise_to_observations(observations, self.augment_noise)
        
        dataset = SupervisedDataset(observations, data['actions'], self.augment_noise)
        # Create a generator compatible with the device set for the model
        generator = torch.Generator(device=default_device())
        return DataLoader(
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