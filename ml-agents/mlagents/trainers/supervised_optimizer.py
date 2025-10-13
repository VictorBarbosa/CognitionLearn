
"""
Module to implement the supervised learning optimizer.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import torch
from torch import nn

from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import SupervisedLearningSettings, NetworkSettings
from mlagents.trainers.supervised_data_loader import SupervisedDataLoader
from mlagents.trainers.trajectory import ObsUtil
from mlagents.torch_utils import torch as torch_module, default_device, set_torch_config
from mlagents.trainers.torch_entities.model_serialization import ModelSerializer
from mlagents.trainers.torch_entities.sequential_model import SequentialActor


class SupervisedTorchOptimizer:
    """
    Optimizer for supervised training of policy neural networks.
    """
    
    def __init__(
        self,
        policy: TorchPolicy,
        supervised_settings: SupervisedLearningSettings,
        stats_reporter=None,
        use_sequential_model: bool = False
    ):
        """
        :param policy: Policy to be trained with supervised learning
        :param supervised_settings: Supervised learning settings
        :param stats_reporter: Stats reporter for TensorBoard
        :param use_sequential_model: Whether to use the sequential model instead of the original model
        """
        self.policy = policy
        self.settings = supervised_settings
        self.stats_reporter = stats_reporter
        self.use_sequential_model = use_sequential_model
        
        if use_sequential_model:
            # Create a sequential model compatible with the existing behavior_spec
            self.sequential_actor = SequentialActor(
                observation_specs=self.policy.behavior_spec.observation_specs,
                network_settings=self.policy.network_settings,
                action_spec=self.policy.behavior_spec.action_spec,
                name_behavior=self.policy.behavior_spec.name,
                dropout_rate=getattr(self.settings, 'dropout_rate', 0.1)  # Adding dropout
            )
            
            # Replace the optimizer to use the new sequential model with weight decay for regularization
            self.optimizer = torch_module.optim.Adam(
                self.sequential_actor.parameters(),
                lr=self.settings.learning_rate,
                weight_decay=getattr(self.settings, 'weight_decay', 1e-4)  # Adding weight decay
            )
            
            # Update the policy to use the sequential model (this will affect export)
            self.original_actor = self.policy.actor
            self.policy.actor = self.sequential_actor
        else:
            # Configure the optimizer and loss function for the original model with weight decay
            self.optimizer = torch_module.optim.Adam(
                self.policy.actor.parameters(),
                lr=self.settings.learning_rate,
                weight_decay=getattr(self.settings, 'weight_decay', 1e-4)  # Adding weight decay
            )
        
        # Use MSE for action regression
        self.criterion = nn.MSELoss()
        
        # Settings for early stopping
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_model_state = None
        self.patience_counter = 0
        self.min_delta = self.settings.min_delta
        self.patience = self.settings.patience
        self.use_early_stopping = self.settings.early_stopping if hasattr(self.settings, 'early_stopping') else True  # Checking if early stopping is enabled
        
        # Adding learning rate scheduler
        self.scheduler = torch_module.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=getattr(self.settings, 'lr_patience', 5)  # Patience for learning rate reduction
        )
        
        # Exporter for ONNX
        self.exporter = ModelSerializer(self.policy)
        
        self.device = default_device()
        if use_sequential_model:
            self.sequential_actor.to(self.device)
        else:
            self.policy.actor.to(self.device)
    
    def train_epoch(
        self,
        data_loader: torch_module.utils.data.DataLoader
    ) -> float:
        """
        Executes one training epoch.
        :param data_loader: DataLoader with the training data
        :return: Average loss for the epoch
        """
        self.policy.actor.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (observations, actions) in enumerate(data_loader):
            observations = observations.to(self.device)
            actions = actions.to(self.device)
            
            # Prepare observations for the expected format by the ML-Agents policy
            processed_obs = self._prepare_observations(observations)
            
            # Get the predicted actions from the current policy
            # Using get_action_and_stats which returns (action, run_out, memories)
            try:
                predicted_action_tuple, run_out, _ = self.policy.actor.get_action_and_stats(
                    processed_obs, masks=None  # masks=None for deterministic actions
                )
                
                # Extract actions from the returned tuple
                if hasattr(predicted_action_tuple, 'continuous_tensor') and predicted_action_tuple.continuous_tensor is not None:
                    predicted_actions = predicted_action_tuple.continuous_tensor
                elif hasattr(predicted_action_tuple, 'discrete_tensor') and predicted_action_tuple.discrete_tensor is not None:
                    predicted_actions = predicted_action_tuple.discrete_tensor
                else:
                    # If we don't find continuous or discrete actions, try another approach
                    predicted_actions = None
            except:
                predicted_actions = None
            
            # If we still don't find actions, try get_stats
            if predicted_actions is None:
                try:
                    run_out = self.policy.actor.get_stats(processed_obs)
                    # Extract actions from the run_out which is an output dictionary
                    if 'action' in run_out:
                        action_tuple = run_out['action']
                        if hasattr(action_tuple, 'continuous_tensor') and action_tuple.continuous_tensor is not None:
                            predicted_actions = action_tuple.continuous_tensor
                        elif hasattr(action_tuple, 'discrete_tensor') and action_tuple.discrete_tensor is not None:
                            predicted_actions = action_tuple.discrete_tensor
                except:
                    predicted_actions = None
            
            # Ensure predicted and real actions have the same format
            if predicted_actions is not None:
                # Convert actions to the correct format if necessary
                if predicted_actions.shape != actions.shape:
                    # If predicted actions have more dimensions than real ones, reduce
                    if len(predicted_actions.shape) > len(actions.shape):
                        # We might have extra actions or extra dimensions that need to be adjusted
                        if predicted_actions.shape[0] == actions.shape[0]:
                            # Same number of samples, adjust the other dimensions
                            if len(actions.shape) == 2 and actions.shape[1] == 1 and len(predicted_actions.shape) >= 2:
                                # Real actions are columns, we flatten the predicted ones to have a compatible shape
                                predicted_actions = predicted_actions.view(actions.shape)
                
                # Calculate the loss between the predicted actions and the actions from the CSV
                loss = self.criterion(predicted_actions, actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            else:
                print(f"WARNING: Could not get predicted actions for batch {batch_idx}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(
        self,
        data_loader: torch_module.utils.data.DataLoader
    ) -> float:
        """
        Validates the model on the validation set.
        :param data_loader: DataLoader with the validation data
        :return: Average validation loss
        """
        self.policy.actor.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch_module.no_grad():
            for observations, actions in data_loader:
                observations = observations.to(self.device)
                actions = actions.to(self.device)
                
                # Prepare observations and get predictions
                processed_obs = self._prepare_observations(observations)
                
                # Get predicted actions from the current policy
                # Using get_action_and_stats which returns (action, run_out, memories)
                try:
                    predicted_action_tuple, run_out, _ = self.policy.actor.get_action_and_stats(
                        processed_obs, masks=None  # masks=None for deterministic actions
                    )
                    
                    # Extract actions from the returned tuple
                    if hasattr(predicted_action_tuple, 'continuous_tensor') and predicted_action_tuple.continuous_tensor is not None:
                        predicted_actions = predicted_action_tuple.continuous_tensor
                    elif hasattr(predicted_action_tuple, 'discrete_tensor') and predicted_action_tuple.discrete_tensor is not None:
                        predicted_actions = predicted_action_tuple.discrete_tensor
                    else:
                        predicted_actions = None
                except:
                    predicted_actions = None
                
                # If we still don't find actions, try get_stats
                if predicted_actions is None:
                    try:
                        run_out = self.policy.actor.get_stats(processed_obs)
                        # Extract actions from the run_out which is an output dictionary
                        if 'action' in run_out:
                            action_tuple = run_out['action']
                            if hasattr(action_tuple, 'continuous_tensor') and action_tuple.continuous_tensor is not None:
                                predicted_actions = action_tuple.continuous_tensor
                            elif hasattr(action_tuple, 'discrete_tensor') and action_tuple.discrete_tensor is not None:
                                predicted_actions = action_tuple.discrete_tensor
                    except:
                        predicted_actions = None
                
                if predicted_actions is not None:
                    # Adjust action format if necessary
                    if predicted_actions.shape != actions.shape:
                        if len(predicted_actions.shape) > len(actions.shape):
                            predicted_actions = predicted_actions.view(actions.shape)
                    
                    loss = self.criterion(predicted_actions, actions)
                    total_loss += loss.item()
                    num_batches += 1
                else:
                    print("WARNING: Could not get predicted actions for validation")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def _prepare_observations(self, observations: torch_module.Tensor) -> List[torch_module.Tensor]:
        """
        Prepares observations for input into the ML-Agents policy.
        Converts from a simple tensor to the list of tensors that the policy expects.
        """
        # Get the observation specifications from the policy
        obs_specs = self.policy.behavior_spec.observation_specs
        
        # Split the flat tensor into the different expected observations
        prepared_obs = []
        current_idx = 0
        
        for obs_spec in obs_specs:
            # Calculate the size of the expected observation
            obs_size = int(np.prod(obs_spec.shape))
            # Extract the data for this specific observation
            obs_tensor = observations[:, current_idx:current_idx + obs_size].contiguous()
            # Resize to the expected shape: (batch_size, *obs_shape)
            obs_tensor = obs_tensor.view(observations.size(0), *obs_spec.shape)
            prepared_obs.append(obs_tensor)
            current_idx += obs_size
        
        return prepared_obs
    
    def should_stop_early(self, val_loss: float, current_epoch: int, metrics: Dict[str, Any]) -> bool:
        """
        Checks if training should stop early due to lack of improvement.
        :param val_loss: Current validation loss
        :param current_epoch: Current epoch (0-indexed)
        :param metrics: Dictionary of metrics to be updated
        :return: True if training should stop early
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = current_epoch
            self.patience_counter = 0
            # Update best_epoch when we find a better loss
            metrics['best_epoch'] = current_epoch + 1  # 1-indexed for reporting consistency
            # Save the state of the best model
            self.best_model_state = {key: value.clone() for key, value in self.policy.actor.state_dict().items()}
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.patience
    
    def train(
        self,
        csv_path: str,
        observation_columns: List[str],
        action_columns: List[str],
        batch_size: Optional[int] = None,
        num_epochs: Optional[int] = None,
        validation_split: Optional[float] = None,
        shuffle: Optional[bool] = None,
        augment_noise: Optional[float] = None,
        checkpoint_interval: Optional[int] = None,
        artifact_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the complete supervised training.
        :param csv_path: Path to the training CSV file
        :param observation_columns: Observation columns in the CSV
        :param action_columns: Action columns in the CSV
        :param batch_size: Batch size (uses config if not provided)
        :param num_epochs: Number of training epochs (uses config if not provided)
        :param validation_split: Fraction of data for validation (uses config if not provided)
        :param shuffle: Whether to shuffle the data (uses config if not provided)
        :param augment_noise: Noise level for data augmentation (uses config if not provided)
        :param checkpoint_interval: Epoch interval to save ONNX (uses config if not provided)
        :param artifact_path: Path to save artifacts (ONNX models)
        :return: Training metrics
        """
        # Use default settings if not provided
        batch_size = batch_size or self.settings.batch_size
        num_epochs = num_epochs or self.settings.num_epoch
        validation_split = validation_split or self.settings.validation_split
        shuffle = shuffle or self.settings.shuffle
        augment_noise = augment_noise or self.settings.augment_noise
        checkpoint_interval = checkpoint_interval or self.settings.checkpoint_interval
        # Load data
        data_loader = SupervisedDataLoader(
            csv_path=csv_path,
            observation_columns=observation_columns,
            action_columns=action_columns,
            validation_split=validation_split,
            shuffle=shuffle,
            augment_noise=augment_noise,
            action_spec=self.policy.behavior_spec.action_spec  # Adding action_spec for validation
        )
        train_loader = data_loader.get_train_loader(batch_size)
        val_loader = data_loader.get_validation_loader(batch_size)
        
        metrics = {
            'train_losses': [],
            'val_losses': [],
            'best_epoch': 0
        }
        
        print("Starting supervised training...")
        # ... (ASCII art removed for brevity)

        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            metrics['train_losses'].append(train_loss)
            metrics['val_losses'].append(val_loss)
            
            # Update scheduler based on validation loss
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Export ONNX at defined intervals
            if artifact_path and (epoch + 1) % checkpoint_interval == 0:
                try:
                    checkpoint_path = f"{artifact_path}/supervised_{epoch+1}"
                    self.exporter.export_policy_model(checkpoint_path)
                    print(f"ONNX model exported: {checkpoint_path}.onnx")
                except Exception as e:
                    print(f"Error exporting ONNX model at epoch {epoch+1}: {e}")
            
            # Check for early stopping
            if self.use_early_stopping and self.should_stop_early(val_loss, epoch, metrics):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # After training, restore the best model if early stopping is enabled
        if self.use_early_stopping and self.best_model_state is not None:
            self.restore_best_model()
        
        # Export final model (if early stopping is enabled, this will be the best model found)
        if artifact_path:
            try:
                final_path = f"{artifact_path}/supervised_final"
                self.exporter.export_policy_model(final_path)
                print(f"Final ONNX model exported: {final_path}.onnx")
            except Exception as e:
                print(f"Error exporting final ONNX model: {e}")
            
            # Export the best model found during training
            if self.best_model_state is not None:
                try:
                    best_path = f"{artifact_path}/supervised_best"
                    self.exporter.export_policy_model(best_path)
                    print(f"Best ONNX model exported: {best_path}.onnx (epoch {self.best_epoch + 1})")
                except Exception as e:
                    print(f"Error exporting best ONNX model: {e}")
        
        print("Supervised training completed.")

        # Save trained weights for later verification
        self._save_trained_weights(artifact_path)
        
        # Additional evaluation to check model quality
        if artifact_path:
            self._evaluate_model_quality(val_loader, artifact_path)
        
        # Restore the original model if using a sequential model
        # This is crucial to ensure RL training is not affected
        if self.use_sequential_model and hasattr(self, 'original_actor'):
            self.policy.actor = self.original_actor
            print("[AUDIT] Original model restored for RL training")

        return metrics

    def _save_trained_weights(self, artifact_path: Optional[str]) -> None:
        """
        Saves the trained weights for later verification.
        :param artifact_path: Path to save the weights
        """
        if artifact_path:
            try:
                import os
                weights_path = f"{artifact_path}/supervised_weights.pth"
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                
                # Save weights in a format compatible with ML-Agents
                state_dict = self.policy.actor.state_dict()
                torch_module.save(state_dict, weights_path)
                print(f"[AUDIT] Supervised weights saved: {weights_path}")
                # Calculate and log hash of weights for verification
                weights_hash = self._calculate_weights_hash()
                print(f"[AUDIT] Hash of supervised weights: {weights_hash}")
            except Exception as e:
                print(f"Error saving supervised weights: {e}")
    
    def _calculate_weights_hash(self) -> str:
        """
        Calculates a hash of the current weights for verification.
        :return: MD5 hash of the current weights
        """
        try:
            import hashlib
            import pickle
            
            # Serialize the current weights
            state_dict = self.policy.actor.state_dict()
            # Sort keys to ensure consistency
            sorted_items = sorted(state_dict.items())
            weights_bytes = pickle.dumps(sorted_items)
            # Calculate hash
            return hashlib.md5(weights_bytes).hexdigest()
        except Exception as e:
            print(f"Error calculating weights hash: {e}")
            return "unknown"
    
    def restore_best_model(self):
        """
        Restores the model state with the best performance found during training.
        """
        if self.best_model_state is not None:
            if self.use_sequential_model and hasattr(self, 'sequential_actor'):
                # Load the best state into the sequential model
                self.sequential_actor.load_state_dict(self.best_model_state)
                # Update the current policy to use the model with the best state
                self.policy.actor = self.sequential_actor
            else:
                self.policy.actor.load_state_dict(self.best_model_state)
            print(f"Best model from epoch {self.best_epoch + 1} with loss {self.best_loss:.6f} restored")
        else:
            print("No best model found to restore")
    
    def get_trained_weights_hash(self) -> str:
        """
        Returns the hash of the trained weights for verification.
        :return: MD5 hash of the trained weights
        """
        return self._calculate_weights_hash()
    
    def compare_weights_with_saved(self, weights_file_path: str) -> bool:
        """
        Compares the current weights with weights saved in a file.
        :param weights_file_path: Path to the weights file
        :return: True if the weights are the same, False otherwise
        """
        try:
            import os
            if not os.path.exists(weights_file_path):
                print(f"Weights file not found: {weights_file_path}")
                return False
                
            current_state = self.policy.actor.state_dict()
            saved_state = torch_module.load(weights_file_path)
            
            # Compare each weight tensor
            for key in current_state.keys():
                if key not in saved_state:
                    print(f"Key {key} not found in saved weights")
                    return False
                    
                if not torch_module.equal(current_state[key], saved_state[key]):
                    print(f"Weights differ at key {key}")
                    return False
                    
            print("Weights verified: IDENTICAL")
            return True
            
        except Exception as e:
            print(f"Error comparing weights: {e}")
            return False
    
    def export_policy_model(self, output_path: str) -> None:
        """
        Exports the policy model in a format compatible with ML-Agents.
        :param output_path: Path to save the model (without extension)
        """
        try:
            # Ensure the model is in evaluation mode
            self.policy.actor.eval()
            
            # Export in .pt format compatible with ML-Agents
            pt_path = f"{output_path}.pt"
            state_dict = self.policy.actor.state_dict()
            torch_module.save(state_dict, pt_path)
            print(f"PyTorch model exported: {pt_path}")
            
            # Also export in ONNX format using the ML-Agents exporter for full compatibility
            if getattr(self.settings, 'export_onnx', True):
                try:
                    # Use the ML-Agents exporter to ensure full compatibility
                    self.exporter.export_policy_model(output_path)
                    print(f"ONNX model exported: {output_path}.onnx")
                except Exception as onnx_error:
                    print(f"Warning: Error exporting ONNX model: {onnx_error}")
                    # Try alternative export
                    onnx_path = f"{output_path}.onnx"
                    self._export_to_onnx_alternative(onnx_path)
                    print(f"ONNX model exported (alternative): {onnx_path}")
                
        except Exception as e:
            print(f"Error exporting model: {e}")
    
    def _export_to_onnx_alternative(self, output_path: str) -> None:
        """
        Exports the model to ONNX format using an alternative method.
        :param output_path: Path to save the ONNX model
        """
        try:
            # Ensure the model is in evaluation mode
            self.policy.actor.eval()
            
            # Create dummy inputs for export following the ML-Agents standard
            observation_specs = self.policy.behavior_spec.observation_specs
            batch_dim = [1]
            seq_len_dim = [1]
            
            # Create dummy observations based on specifications
            dummy_obs = []
            for obs_spec in observation_specs:
                obs_shape = self._get_onnx_shape(obs_spec.shape)
                dummy_obs.append(
                    torch_module.zeros(batch_dim + list(obs_shape)).to(self.device)
                )
            
            # Create dummy action masks
            total_discrete_branches = sum(self.policy.behavior_spec.action_spec.discrete_branches)
            dummy_masks = torch_module.ones(
                batch_dim + [total_discrete_branches]
            ).to(self.device) if self.policy.behavior_spec.action_spec.is_discrete and total_discrete_branches > 0 else None
            
            # Create dummy memories if necessary
            dummy_memories = torch_module.zeros(
                batch_dim + seq_len_dim + [self.policy.export_memory_size]
            ).to(self.device) if self.policy.export_memory_size > 0 else None
            
            # Export to ONNX
            torch_module.onnx.export(
                self.policy.actor,
                (dummy_obs, dummy_masks, dummy_memories),
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['vector_observation', 'action_masks', 'recurrent_in'],
                output_names=['version_number', 'memory_size'],
                dynamic_axes={
                    'vector_observation': {0: 'batch_size'},
                    'action_masks': {0: 'batch_size'},
                    'recurrent_in': {0: 'batch_size'}
                }
            )
            
        except Exception as e:
            print(f"Error exporting alternative ONNX model: {e}")
            raise
    
    def _get_onnx_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Converts the shape of an observation to be compatible with the NCHW format of ONNX.
        :param shape: Original shape of the observation
        :return: ONNX-compatible shape
        """
        if len(shape) == 3:
            # For 3D visual observations, keep the format (channel, height, width)
            return shape[0], shape[1], shape[2]
        return shape
    
    def _evaluate_model_quality(self, val_loader, artifact_path: str) -> None:
        """
        Additional evaluation to check model quality.
        """
        print("Evaluating model quality...")
        
        # Calculate additional metrics
        self.policy.actor.eval()
        all_predictions = []
        all_targets = []
        
        with torch_module.no_grad():
            for observations, actions in val_loader:
                observations = observations.to(self.device)
                actions = actions.to(self.device)
                
                # Prepare observations and get predictions
                processed_obs = self._prepare_observations(observations)
                
                # Get predicted actions from the current policy
                try:
                    predicted_action_tuple, run_out, _ = self.policy.actor.get_action_and_stats(
                        processed_obs, masks=None
                    )
                    
                    # Extract actions from the returned tuple
                    if hasattr(predicted_action_tuple, 'continuous_tensor') and predicted_action_tuple.continuous_tensor is not None:
                        predicted_actions = predicted_action_tuple.continuous_tensor
                    elif hasattr(predicted_action_tuple, 'discrete_tensor') and predicted_action_tuple.discrete_tensor is not None:
                        predicted_actions = predicted_action_tuple.discrete_tensor
                    else:
                        continue
                except:
                    continue
                
                # Adjust action format if necessary
                if predicted_actions.shape != actions.shape:
                    if len(predicted_actions.shape) > len(actions.shape):
                        predicted_actions = predicted_actions.view(actions.shape)
                
                all_predictions.append(predicted_actions.cpu())
                all_targets.append(actions.cpu())
        
        if all_predictions and all_targets:
            all_predictions = torch_module.cat(all_predictions, dim=0)
            all_targets = torch_module.cat(all_targets, dim=0)
            
            # Calculate additional metrics
            mse = torch_module.mean((all_predictions - all_targets) ** 2).item()
            mae = torch_module.mean(torch_module.abs(all_predictions - all_targets)).item()
            
            # Calculate R² (coefficient of determination)
            ss_res = torch_module.sum((all_targets - all_predictions) ** 2)
            ss_tot = torch_module.sum((all_targets - torch_module.mean(all_targets)) ** 2)
            r2 = (1 - ss_res / ss_tot).item() if ss_tot != 0 else 0
            
            print(f"Evaluation Metrics - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
            
            # Save metrics to file
            import os
            metrics_path = f"{artifact_path}/supervised_metrics.txt"
            with open(metrics_path, 'w') as f:
                f.write(f"MSE: {mse:.6f}\n")
                f.write(f"MAE: {mae:.6f}\n")
                f.write(f"R²: {r2:.6f}\n")
                f.write(f"Best Epoch: {self.best_epoch + 1 if self.best_epoch is not None else 'N/A'}\n")
                f.write(f"Best Val Loss: {self.best_loss:.6f}\n")
            
            print(f"Metrics saved to: {metrics_path}")
