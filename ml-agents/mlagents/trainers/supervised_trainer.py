"""
Implementation of the standalone supervised trainer for ML-Agents.
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader

from mlagents_envs.base_env import ActionSpec, BehaviorSpec, ObservationSpec, ObservationType, DimensionProperty
from mlagents.trainers.settings import (
    SupervisedLearningSettings, 
    NetworkSettings, 
    RunOptions,
    TrainerSettings
)
from mlagents.trainers.supervised_data_loader import SupervisedDataLoader
from mlagents.trainers.trajectory import ObsUtil
from mlagents.torch_utils import torch as torch_module, default_device, set_torch_config
from mlagents.trainers.torch_entities.model_serialization import ModelSerializer
from mlagents.trainers.torch_entities.sequential_model import SequentialActor
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.stats import StatsReporter


class SupervisedTrainer:
    """
    Trainer for supervised training of policy neural networks.
    """
    
    def __init__(
        self,
        behavior_name: str,
        behavior_config: TrainerSettings,
        run_options: RunOptions,
        all_target_algorithms: List[str],
        use_sequential_model: bool = False,
        verbose: bool = False
    ):
        """
        :param behavior_name: Name of the behavior to be trained
        :param behavior_config: Behavior configuration for the base training run
        :param run_options: Run options
        :param all_target_algorithms: List of all algorithm types to export models for
        :param use_sequential_model: Whether to use the sequential model instead of the original model
        :param verbose: Whether to show detailed information during training
        """
        self.behavior_name = behavior_name
        self.behavior_config = behavior_config
        self.run_options = run_options
        self.all_target_algorithms = all_target_algorithms
        self.use_sequential_model = use_sequential_model
        self.verbose = verbose
        self.stats_reporter = StatsReporter(behavior_name)
        
        self.supervised_settings = behavior_config.supervised
        self.network_settings = behavior_config.network_settings
        
        self.device = default_device()
        print(f"""################################################################\n{self.all_target_algorithms}\n################################################################""")
        print(f"[SUPERVISED] Using device: {self.device}")
        # Create a dummy policy for the base training architecture
        self.policy = self._create_dummy_policy(self.behavior_config.trainer_type)
        
        if use_sequential_model:
            self.actor = self._create_sequential_actor()
        else:
            self.actor = self.policy.actor
            
        self.actor.to(self.device)
        
        self.optimizer = torch_module.optim.Adam(
            self.actor.parameters(),
            lr=self.supervised_settings.learning_rate,
            weight_decay=getattr(self.supervised_settings, 'weight_decay', 1e-4)
        )
        
        self.criterion = nn.MSELoss()
        
        self.scheduler = torch_module.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=getattr(self.supervised_settings, 'lr_patience', 5)
        )
        
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_model_state = None
        self.patience_counter = 0
        self.min_delta = self.supervised_settings.min_delta
        self.patience = self.supervised_settings.patience
        self.use_early_stopping = self.supervised_settings.early_stopping
        
        self.data_loader = self._load_data()
        
        print(f"[SUPERVISED] Trainer configured for behavior: {behavior_name}")
        print(f"[SUPERVISED] Using {'sequential' if use_sequential_model else 'original'} model for training.")

    def _create_dummy_policy(self, algorithm: str) -> TorchPolicy:
        """
        Creates a dummy policy for a given algorithm type.
        """
        from mlagents.trainers.torch_entities.networks import SimpleActor

        if algorithm == "sac" or algorithm == "tdsac":
            actor_kwargs = {"conditional_sigma": True, "tanh_squash": True}
        else:  # ppo, poca
            actor_kwargs = {"conditional_sigma": False, "tanh_squash": False}

        obs_shape = tuple(self.supervised_settings.observation_shape)
        observation_specs = [
            ObservationSpec(
                name="vector_observation",
                shape=obs_shape,
                dimension_property=(DimensionProperty.NONE,) * len(obs_shape),
                observation_type=ObservationType.DEFAULT
            )
        ]
        
        if self.supervised_settings.action_type == "continuous":
            action_spec = ActionSpec.create_continuous(self.supervised_settings.action_size)
        else:
            action_spec = ActionSpec.create_discrete((self.supervised_settings.action_size,))
        
        behavior_spec = BehaviorSpec(observation_specs, action_spec)
        
        return TorchPolicy(
            seed=42,
            behavior_spec=behavior_spec,
            network_settings=self.network_settings,
            actor_cls=self._get_actor_class(),
            actor_kwargs=actor_kwargs
        )

    def _get_actor_class(self):
        """
        Returns the appropriate actor class.
        """
        if self.use_sequential_model:
            return SequentialActor
        else:
            from mlagents.trainers.torch_entities.networks import SimpleActor
            return SimpleActor

    def _create_sequential_actor(self):
        """
        Creates a sequential model for supervised training.
        """
        return SequentialActor(
            observation_specs=self.policy.behavior_spec.observation_specs,
            network_settings=self.network_settings,
            action_spec=self.policy.behavior_spec.action_spec,
            name_behavior=self.behavior_name,
            dropout_rate=getattr(self.supervised_settings, 'dropout_rate', 0.1)
        )

    def _load_data(self) -> SupervisedDataLoader:
        """
        Loads supervised training data from CSV.
        """
        return SupervisedDataLoader(
            csv_path=self.supervised_settings.csv_path,
            observation_columns=self.supervised_settings.observation_columns,
            action_columns=self.supervised_settings.action_columns,
            validation_split=self.supervised_settings.validation_split,
            shuffle=self.supervised_settings.shuffle,
            augment_noise=self.supervised_settings.augment_noise,
            action_spec=self.policy.behavior_spec.action_spec
        )

    def train_epoch(self, data_loader: DataLoader) -> float:
        """
        Executes one training epoch.
        """
        self.actor.train()
        total_loss = 0.0
        num_batches = 0
        for batch_idx, (observations, actions) in enumerate(data_loader):
            observations = observations.to(self.device)
            actions = actions.to(self.device)
            processed_obs = self._prepare_observations(observations)
            try:
                agent_action, _, _ = self.actor.get_action_and_stats(processed_obs, masks=None)
                predicted_actions = agent_action.continuous_tensor if agent_action.continuous_tensor is not None else None
            except Exception as e:
                if self.verbose:
                    print(f"[SUPERVISED] Error getting predicted actions: {e}")
                continue
            
            if predicted_actions is not None and predicted_actions.shape == actions.shape:
                loss = self.criterion(predicted_actions, actions)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            else:
                if self.verbose:
                    print(f"[SUPERVISED] Incompatible action format in batch {batch_idx}")
        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self, data_loader: DataLoader) -> float:
        """
        Validates the model on the validation set.
        """
        self.actor.eval()
        total_loss = 0.0
        num_batches = 0
        with torch_module.no_grad():
            for observations, actions in data_loader:
                observations = observations.to(self.device)
                actions = actions.to(self.device)
                processed_obs = self._prepare_observations(observations)
                try:
                    agent_action, _, _ = self.actor.get_action_and_stats(processed_obs, masks=None)
                    predicted_actions = agent_action.continuous_tensor if agent_action.continuous_tensor is not None else None
                except Exception as e:
                    if self.verbose:
                        print(f"[SUPERVISED] Error getting predicted actions in validation: {e}")
                    continue
                
                if predicted_actions is not None and predicted_actions.shape == actions.shape:
                    loss = self.criterion(predicted_actions, actions)
                    total_loss += loss.item()
                    num_batches += 1
                else:
                    if self.verbose:
                        print("[SUPERVISED] Incompatible action format in validation")
        return total_loss / num_batches if num_batches > 0 else 0.0

    def _prepare_observations(self, observations: torch_module.Tensor) -> List[torch_module.Tensor]:
        """
        Prepares observations for input into the model.
        """
        if len(observations.shape) == 2:
            return [observations]
        elif len(observations.shape) == 3:
            return [observations[:, i, :] for i in range(observations.shape[1])]
        else:
            return [observations.view(observations.size(0), -1)]

    def should_stop_early(self, val_loss: float, current_epoch: int) -> bool:
        """
        Checks if training should stop early.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = current_epoch
            self.patience_counter = 0
            self.best_model_state = {key: value.clone() for key, value in self.actor.state_dict().items()}
        else:
            self.patience_counter += 1
        return self.patience_counter >= self.patience

    def restore_best_model(self):
        """
        Restores the best model found during training.
        """
        if self.best_model_state is not None:
            self.actor.load_state_dict(self.best_model_state)
            print(f"[SUPERVISED] Best model from epoch {self.best_epoch + 1} with loss {self.best_loss:.6f} restored")
        else:
            print("[SUPERVISED] No best model found to restore")

    def train(self):
        """
        Executes the complete supervised training.
        """
        print("[SUPERVISED] Starting supervised training...")
        train_loader = self.data_loader.get_train_loader(self.supervised_settings.batch_size)
        val_loader = self.data_loader.get_validation_loader(self.supervised_settings.batch_size)
        
        for epoch in range(self.supervised_settings.num_epoch):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.stats_reporter.add_stat("Supervised/Train Loss", train_loss)
            self.stats_reporter.add_stat("Supervised/Validation Loss", val_loss)
            self.stats_reporter.write_stats(epoch + 1)
            
            self.scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) % self.supervised_settings.checkpoint_interval == 0:
                print(f"[SUPERVISED] Epoch {epoch+1}/{self.supervised_settings.num_epoch} - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if self.use_early_stopping and self.should_stop_early(val_loss, epoch):
                print(f"[SUPERVISED] Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % self.supervised_settings.checkpoint_interval == 0:
                self._save_checkpoint(epoch + 1)
        
        if self.use_early_stopping:
            self.restore_best_model()
        
        self._save_final_model()
        print("[SUPERVISED] Supervised training completed.")

    def _save_model_for_algorithm(self, algorithm: str, state_dict: Dict, basename: str):
        """
        Helper function to save a model for a specific algorithm architecture.
        """
        try:
            base_dir = self.run_options.checkpoint_settings.run_id
            output_dir = os.path.join(base_dir, algorithm)
            os.makedirs(output_dir, exist_ok=True)

            temp_policy = self._create_dummy_policy(algorithm)
            temp_policy.actor.load_state_dict(state_dict, strict=False)
            
            model_path = os.path.join(output_dir, basename)
            
            torch_module.save({'Policy': temp_policy.actor.state_dict()}, f"{model_path}.pt")
            print(f"  - Saved {algorithm.upper()} PT: {model_path}.pt")

            exporter = ModelSerializer(temp_policy)
            exporter.export_policy_model(model_path)
            print(f"  - Exported {algorithm.upper()} ONNX: {model_path}.onnx")
        except Exception as e:
            print(f"  - Error saving model for {algorithm.upper()}: {e}")

    def _save_checkpoint(self, epoch: int):
        """
        Saves a checkpoint for all target algorithms.
        """
        print(f"--- Checkpointing models for Epoch {epoch} ---")
        trained_state_dict = self.actor.state_dict()
        for algorithm in self.all_target_algorithms:
            self._save_model_for_algorithm(algorithm, trained_state_dict, f"supervised_checkpoint_{epoch}")

    def _save_final_model(self):
        """
        Saves the final best model for all target architectures.
        """
        if self.best_model_state is None:
            print("[SUPERVISED] No best model state found to save. Saving current model instead.")
            best_state_dict = self.actor.state_dict()
        else:
            best_state_dict = self.best_model_state

        print("\n--- Saving final best models for all architectures ---")
        for algorithm in self.all_target_algorithms:
            self._save_model_for_algorithm(algorithm, best_state_dict, "supervised_best")