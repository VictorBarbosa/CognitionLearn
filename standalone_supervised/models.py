"""
Model definitions for supervised learning compatible with ML-Agents algorithms.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from mlagents_envs.base_env import BehaviorSpec, ActionSpec, ObservationSpec, ObservationType
from mlagents.trainers.torch_entities.encoders import VectorInput
from mlagents.trainers.torch_entities.decoders import ValueHeads
from mlagents.trainers.torch_entities.action_model import ActionModel
from mlagents.trainers.settings import NetworkSettings


class SequentialActor(nn.Module):
    """
    Sequential actor model compatible with ML-Agents export system.
    """
    
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        name_behavior: str = "sequential_actor",
        dropout_rate: float = 0.1
    ):
        """
        :param observation_specs: Observation specifications
        :param network_settings: Network settings
        :param action_spec: Action specification
        :param name_behavior: Behavior name
        :param dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.observation_specs = observation_specs
        self.action_spec = action_spec
        self.network_settings = network_settings
        self.name_behavior = name_behavior
        self.dropout_rate = dropout_rate
        
        # Calculate total observation size
        total_obs_size = 0
        for obs_spec in observation_specs:
            total_obs_size += int(torch.prod(torch.tensor(obs_spec.shape)))
        
        # Create vector encoder for observations
        self.vector_encoder = VectorInput(
            total_obs_size,
            normalize=network_settings.normalize
        )
        
        # Create sequential hidden layers with dropout for regularization
        hidden_layers = []
        current_size = total_obs_size
        
        # Add hidden layers with dropout
        for _ in range(network_settings.num_layers):
            hidden_layers.append(nn.Linear(current_size, network_settings.hidden_units))
            hidden_layers.append(nn.ReLU())
            # Add dropout after each ReLU for regularization
            if dropout_rate > 0:
                hidden_layers.append(nn.Dropout(dropout_rate))
            current_size = network_settings.hidden_units
        
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        # Create action model (handles continuous and discrete actions)
        self.action_model = ActionModel(
            current_size,
            action_spec,
            network_settings
        )
        
        # Memory (for recurrent networks, if needed)
        self.memory_size = network_settings.memory_size if network_settings.use_recurrent else 0
        
        # Value heads (for value estimates, if needed)
        self.value_heads = ValueHeads(
            [name_behavior],
            current_size
        )
        
        # Initialize weights for better regularization
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights with regularization for better generalization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs, masks=None, memories=None):
        """
        Forward pass for the sequential model.
        :param obs: Observations (can be a tensor or list of tensors)
        :param masks: Action masks (optional)
        :param memories: Memories for recurrent networks (optional)
        :return: Predicted actions, values, and log probabilities
        """
        # Flatten and concatenate observations
        if isinstance(obs, torch.Tensor):
            # Single tensor observation
            flat_obs = obs.view(obs.size(0), -1)
        elif isinstance(obs, list):
            # List of tensor observations
            if len(obs) == 1:
                # Case when we have only one observation, use directly
                flat_obs = obs[0].view(obs[0].size(0), -1)
            else:
                # Concatenate multiple observations
                concatenated_obs = [ob.view(ob.size(0), -1) for ob in obs]
                flat_obs = torch.cat(concatenated_obs, dim=1)
        else:
            # Unknown observation type
            raise ValueError(f"Unsupported observation type: {type(obs)}")
        
        # Process observations
        encoded_obs = self.vector_encoder(flat_obs)
        
        # Pass through hidden layers
        hidden_out = self.hidden_layers(encoded_obs)
        
        # Get actions
        action, log_probs = self.action_model(hidden_out)
        
        # Get value estimates
        values = self.value_heads(hidden_out)
        
        return action, values, log_probs
    
    def get_action_and_stats(self, obs, masks=None, memories=None):
        """
        Compatible method with existing ML-Agents system.
        """
        action, values, log_probs = self.forward(obs, masks, memories)
        
        # Prepare output compatible with existing system
        run_out = {
            "value_output": values[self.name_behavior],
            "log_probs": log_probs
        }
        
        # For compatibility with ONNX export system
        if memories is None:
            memories = torch.zeros((len(obs[0]), self.memory_size)) if self.memory_size > 0 else torch.tensor([])
        
        return action, run_out, memories
    
    def get_stats(self, obs):
        """
        Compatible method with existing ML-Agents system.
        """
        action, values, log_probs = self.forward(obs)
        
        return {
            "action": action,
            "value_output": values[self.name_behavior],
            "log_probs": log_probs
        }


class PPOActor(SequentialActor):
    """
    PPO-specific actor model compatible with ML-Agents export system.
    """
    
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        name_behavior: str = "ppo_actor",
        dropout_rate: float = 0.1
    ):
        """
        :param observation_specs: Observation specifications
        :param network_settings: Network settings
        :param action_spec: Action specification
        :param name_behavior: Behavior name
        :param dropout_rate: Dropout rate for regularization
        """
        super().__init__(
            observation_specs=observation_specs,
            network_settings=network_settings,
            action_spec=action_spec,
            name_behavior=name_behavior,
            dropout_rate=dropout_rate
        )


class SACActor(SequentialActor):
    """
    SAC-specific actor model compatible with ML-Agents export system.
    """
    
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        name_behavior: str = "sac_actor",
        dropout_rate: float = 0.1
    ):
        """
        :param observation_specs: Observation specifications
        :param network_settings: Network settings
        :param action_spec: Action specification
        :param name_behavior: Behavior name
        :param dropout_rate: Dropout rate for regularization
        """
        super().__init__(
            observation_specs=observation_specs,
            network_settings=network_settings,
            action_spec=action_spec,
            name_behavior=name_behavior,
            dropout_rate=dropout_rate
        )


class TDSACActor(SequentialActor):
    """
    TDSAC-specific actor model compatible with ML-Agents export system.
    """
    
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        name_behavior: str = "tdsac_actor",
        dropout_rate: float = 0.1
    ):
        """
        :param observation_specs: Observation specifications
        :param network_settings: Network settings
        :param action_spec: Action specification
        :param name_behavior: Behavior name
        :param dropout_rate: Dropout rate for regularization
        """
        super().__init__(
            observation_specs=observation_specs,
            network_settings=network_settings,
            action_spec=action_spec,
            name_behavior=name_behavior,
            dropout_rate=dropout_rate
        )


def create_model_for_algorithm(
    algorithm: str,
    observation_specs: List[ObservationSpec],
    network_settings: NetworkSettings,
    action_spec: ActionSpec,
    name_behavior: str = "supervised_actor",
    dropout_rate: float = 0.1
) -> SequentialActor:
    """
    Creates a model appropriate for the specified algorithm.
    
    :param algorithm: Algorithm name (ppo, sac, tdsac)
    :param observation_specs: Observation specifications
    :param network_settings: Network settings
    :param action_spec: Action specification
    :param name_behavior: Behavior name
    :param dropout_rate: Dropout rate for regularization
    :return: SequentialActor model
    """
    # Create model specific to the algorithm
    if algorithm.lower() == "ppo":
        model = PPOActor(
            observation_specs=observation_specs,
            network_settings=network_settings,
            action_spec=action_spec,
            name_behavior=name_behavior,
            dropout_rate=dropout_rate
        )
    elif algorithm.lower() == "sac":
        model = SACActor(
            observation_specs=observation_specs,
            network_settings=network_settings,
            action_spec=action_spec,
            name_behavior=name_behavior,
            dropout_rate=dropout_rate
        )
    elif algorithm.lower() == "tdsac":
        model = TDSACActor(
            observation_specs=observation_specs,
            network_settings=network_settings,
            action_spec=action_spec,
            name_behavior=name_behavior,
            dropout_rate=dropout_rate
        )
    else:
        # Default to generic SequentialActor
        model = SequentialActor(
            observation_specs=observation_specs,
            network_settings=network_settings,
            action_spec=action_spec,
            name_behavior=name_behavior,
            dropout_rate=dropout_rate
        )
    
    return model