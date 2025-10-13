"""
Sequential model for supervised training in ML-Agents.
This model uses nn.Sequential to create a network compatible with the existing ONNX export system.
"""
import torch
import torch.nn as nn
from typing import List, Tuple
from mlagents_envs.base_env import BehaviorSpec, ActionSpec
from mlagents.trainers.torch_entities.encoders import VectorInput
from mlagents.trainers.torch_entities.decoders import ValueHeads
from mlagents.trainers.torch_entities.action_model import ActionModel
from mlagents.trainers.settings import NetworkSettings


class SequentialActor(nn.Module):
    """
    Sequential actor model that is compatible with the ML-Agents ONNX export system.
    """
    def __init__(
        self,
        observation_specs,
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        name_behavior: str = "sequential_actor",
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.observation_specs = observation_specs
        self.action_spec = action_spec
        self.network_settings = network_settings
        self.name_behavior = name_behavior
        self.dropout_rate = dropout_rate
        
        # Calculate the total size of observations
        total_obs_size = 0
        for obs_spec in observation_specs:
            total_obs_size += int(torch.prod(torch.tensor(obs_spec.shape)))
        
        # Create a sequential encoder for the observations
        # First, an encoder to process the vector observations
        self.vector_encoder = VectorInput(
            total_obs_size, 
            normalize=network_settings.normalize
        )
        
        # Create the sequence of layers for the network body
        hidden_layers = []
        current_size = total_obs_size
        
        # Add sequential hidden layers with dropout for regularization
        for _ in range(network_settings.num_layers):
            hidden_layers.append(nn.Linear(current_size, network_settings.hidden_units))
            hidden_layers.append(nn.ReLU())
            # Adding dropout after each ReLU layer for regularization
            if dropout_rate > 0:
                hidden_layers.append(nn.Dropout(dropout_rate))
            current_size = network_settings.hidden_units
        
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        # Create action model (which handles continuous and discrete actions)
        self.action_model = ActionModel(
            current_size,
            action_spec,
            network_settings
        )
        
        # Memory (for recurrent networks, if necessary)
        self.memory_size = network_settings.memory_size if network_settings.use_recurrent else 0
        
        # Value heads (for value estimates, if necessary)
        self.value_heads = ValueHeads(
            [name_behavior], 
            current_size
        )
        
        # Initialize weights to improve regularization
        self._init_weights()
    
    def _init_weights(self):
        """
        Initializes weights with regularization to improve generalization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs, masks=None, memories=None):
        """
        Standard forward pass for the sequential model.
        """
        # Flattening and concatenating observations
        if len(obs) == 1:
            # Case where we have only one observation, use it directly
            flat_obs = obs[0].view(obs[0].size(0), -1)
        else:
            # Concatenate multiple observations
            concatenated_obs = [ob.view(ob.size(0), -1) for ob in obs]
            flat_obs = torch.cat(concatenated_obs, dim=1)
        
        # Process observations
        encoded_obs = self.vector_encoder(flat_obs)
        
        # Pass through sequential hidden layers
        hidden_out = self.hidden_layers(encoded_obs)
        
        # Get actions
        action, log_probs = self.action_model(hidden_out)
        
        # Get value estimates
        values = self.value_heads(hidden_out)
        
        return action, values, log_probs
    
    def get_action_and_stats(self, obs, masks=None, memories=None):
        """
        Method compatible with the existing ML-Agents system.
        """
        action, values, log_probs = self.forward(obs, masks, memories)
        
        # Prepare output compatible with the existing system
        run_out = {
            "value_output": values[self.name_behavior],
            "log_probs": log_probs
        }
        
        # For compatibility with the ONNX export system
        if memories is None:
            memories = torch.zeros((len(obs[0]), self.memory_size)) if self.memory_size > 0 else torch.tensor([])
        
        return action, run_out, memories
    
    def get_stats(self, obs):
        """
        Method compatible with the existing ML-Agents system.
        """
        action, values, log_probs = self.forward(obs)
        
        return {
            "action": action,
            "value_output": values[self.name_behavior],
            "log_probs": log_probs
        }
