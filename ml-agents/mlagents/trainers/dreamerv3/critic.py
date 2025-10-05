from typing import Tuple, Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import abc

from mlagents.torch_utils import torch as torch_utils
from mlagents.trainers.torch_entities.layers import linear_layer, Initialization
from mlagents.trainers.buffer import AgentBuffer


class DreamerV3Critic(nn.Module):
    """
    Implements the Critic (Value function) for DreamerV3 which estimates
    the value of state-action pairs in the latent space.
    """
    
    def update_normalization(self, buffer: AgentBuffer) -> None:
        """
        Updates normalization based on the buffer.
        For this critic, we don't need normalization since it works on latent states.
        """
        pass
    
    def __init__(
        self,
        latent_state_size: int,
        stochastic_state_size: int,
        hidden_size: int = 400,
        ensemble_size: int = 4  # Use ensemble of critics (common in DreamerV3)
    ):
        super().__init__()
        
        self.latent_state_size = latent_state_size
        self.stochastic_state_size = stochastic_state_size
        self.ensemble_size = ensemble_size
        
        # Input size is the concatenated hidden and stochastic states
        input_size = latent_state_size + stochastic_state_size
        
        # Create ensemble of critics
        self.critics = nn.ModuleList([
            nn.Sequential(
                linear_layer(input_size, hidden_size, kernel_init=Initialization.XavierGlorotUniform),
                nn.ELU(),
                linear_layer(hidden_size, hidden_size, kernel_init=Initialization.XavierGlorotUniform),
                nn.ELU(),
                linear_layer(hidden_size, hidden_size, kernel_init=Initialization.XavierGlorotUniform),
                nn.ELU(),
                linear_layer(hidden_size, 1, kernel_init=Initialization.Zero)
            ) for _ in range(ensemble_size)
        ])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        hidden_state: torch.Tensor, 
        stochastic_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the critic ensemble
        Returns the minimum value among the ensemble (for stability)
        """
        # Concatenate hidden and stochastic states
        latent_input = torch.cat([hidden_state, stochastic_state], dim=-1)
        
        values = []
        for critic in self.critics:
            value = critic(latent_input)
            values.append(value)
        
        # Stack values and take the minimum (common technique to prevent overestimation)
        all_values = torch.stack(values, dim=0)
        min_values = torch.min(all_values, dim=0)[0]
        
        return min_values

    def critic_pass(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get value outputs for the given obs.
        For DreamerV3, the inputs are typically hidden and stochastic states from the world model.
        :param inputs: List of inputs as tensors (in DreamerV3 case, hidden and stochastic states)
        :param memories: Tensor of memories, if using memory. Otherwise, None.
        :returns: Dict of reward stream to output tensor for values and new memories.
        """
        # In DreamerV3, inputs[0] is hidden state, inputs[1] is stochastic state
        if len(inputs) >= 2:
            hidden_state = inputs[0]
            stochastic_state = inputs[1]
        else:
            # If only one tensor is provided, assume it's the concatenated latent state
            concat_state = inputs[0]
            # Split the tensor assuming first part is hidden and second part is stochastic
            total_size = concat_state.shape[-1]
            hidden_size = self.latent_state_size
            stochastic_size = self.stochastic_state_size
            if total_size == hidden_size + stochastic_size:
                hidden_state = concat_state[..., :hidden_size]
                stochastic_state = concat_state[..., hidden_size:]
            else:
                # Default: assume all inputs are hidden and stochastic states are zero
                hidden_state = concat_state[..., :hidden_size] if concat_state.shape[-1] >= hidden_size else concat_state
                stochastic_size = self.stochastic_state_size
                stochastic_state = torch.zeros((*hidden_state.shape[:-1], stochastic_size), device=hidden_state.device, dtype=hidden_state.dtype)
        
        # Get the value estimates
        values = self(hidden_state, stochastic_state)
        
        # For reward streams, we'll return a dictionary with a default stream
        value_estimates = {"extrinsic": values.squeeze(-1) if values.dim() > 1 else values}
        
        # Memories are not typically returned by the critic in DreamerV3, return None
        return value_estimates, memories
    
    def all_values(
        self, 
        hidden_state: torch.Tensor, 
        stochastic_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Return values from all critics in the ensemble
        """
        # Concatenate hidden and stochastic states
        latent_input = torch.cat([hidden_state, stochastic_state], dim=-1)
        
        values = []
        for critic in self.critics:
            value = critic(latent_input)
            values.append(value)
        
        # Stack values from all critics
        return torch.stack(values, dim=0)


class DreamerV3ValueModel(nn.Module):
    """
    Implements a value model that can estimate the value of states for planning
    in the latent space of DreamerV3.
    """
    
    def __init__(
        self,
        latent_state_size: int,
        stochastic_state_size: int,
        hidden_size: int = 400
    ):
        super().__init__()
        
        self.latent_state_size = latent_state_size
        self.stochastic_state_size = stochastic_state_size
        
        # Input size is the concatenated hidden and stochastic states
        input_size = latent_state_size + stochastic_state_size
        
        # Value network
        self.value_network = nn.Sequential(
            linear_layer(input_size, hidden_size, kernel_init=Initialization.XavierGlorotUniform),
            nn.ELU(),
            linear_layer(hidden_size, hidden_size, kernel_init=Initialization.XavierGlorotUniform),
            nn.ELU(),
            linear_layer(hidden_size, hidden_size, kernel_init=Initialization.XavierGlorotUniform),
            nn.ELU(),
            linear_layer(hidden_size, 1, kernel_init=Initialization.Zero)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def update_normalization(self, buffer: AgentBuffer) -> None:
        """
        Updates normalization based on the buffer.
        For this value model, we don't need normalization since it works on latent states.
        """
        pass
            
    def forward(
        self, 
        hidden_state: torch.Tensor, 
        stochastic_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the value model
        """
        # Concatenate hidden and stochastic states
        latent_input = torch.cat([hidden_state, stochastic_state], dim=-1)
        
        # Pass through value network
        value = self.value_network(latent_input)
        
        return value

    def critic_pass(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get value outputs for the given obs.
        For DreamerV3, the inputs are typically hidden and stochastic states from the world model.
        :param inputs: List of inputs as tensors (in DreamerV3 case, hidden and stochastic states)
        :param memories: Tensor of memories, if using memory. Otherwise, None.
        :returns: Dict of reward stream to output tensor for values and new memories.
        """
        # In DreamerV3, inputs[0] is hidden state, inputs[1] is stochastic state
        if len(inputs) >= 2:
            hidden_state = inputs[0]
            stochastic_state = inputs[1]
        else:
            # If only one tensor is provided, assume it's the concatenated latent state
            concat_state = inputs[0]
            # Split the tensor assuming first part is hidden and second part is stochastic
            total_size = concat_state.shape[-1]
            hidden_size = self.latent_state_size
            stochastic_size = self.stochastic_state_size
            if total_size == hidden_size + stochastic_size:
                hidden_state = concat_state[..., :hidden_size]
                stochastic_state = concat_state[..., hidden_size:]
            else:
                # Default: assume all inputs are hidden and stochastic states are zero
                hidden_state = concat_state[..., :hidden_size] if concat_state.shape[-1] >= hidden_size else concat_state
                stochastic_size = self.stochastic_state_size
                stochastic_state = torch.zeros((*hidden_state.shape[:-1], stochastic_size), device=hidden_state.device, dtype=hidden_state.dtype)
        
        # Get the value estimates
        values = self(hidden_state, stochastic_state)
        
        # For reward streams, we'll return a dictionary with a default stream
        value_estimates = {"extrinsic": values.squeeze(-1) if values.dim() > 1 else values}
        
        # Memories are not typically returned by the critic in DreamerV3, return None
        return value_estimates, memories