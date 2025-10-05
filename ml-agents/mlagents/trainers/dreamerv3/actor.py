from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlagents.torch_utils import torch as torch_utils
from mlagents.trainers.torch_entities.layers import linear_layer, Initialization
from mlagents_envs.base_env import ActionSpec


class DreamerV3Actor(nn.Module):
    """
    Implements the Actor (Policy) for DreamerV3 which learns to select actions
    based on the latent states from the world model.
    """
    
    def __init__(
        self,
        action_spec: ActionSpec,
        latent_state_size: int,
        stochastic_state_size: int,
        hidden_size: int = 400,
        discrete_temperature: float = 1.0
    ):
        super().__init__()
        
        self.action_spec = action_spec
        self.latent_state_size = latent_state_size
        self.stochastic_state_size = stochastic_state_size
        self.discrete_temperature = discrete_temperature
        self.is_discrete = action_spec.discrete_size > 0
        self.is_continuous = action_spec.continuous_size > 0
        
        # Input size is the concatenated hidden and stochastic states
        input_size = latent_state_size + stochastic_state_size
        
        # Actor network
        self.actor_network = nn.Sequential(
            linear_layer(input_size, hidden_size, kernel_init=Initialization.XavierGlorotUniform),
            nn.ELU(),
            linear_layer(hidden_size, hidden_size, kernel_init=Initialization.XavierGlorotUniform),
            nn.ELU(),
            linear_layer(hidden_size, hidden_size, kernel_init=Initialization.XavierGlorotUniform),
            nn.ELU()
        )
        
        # Action heads
        if self.is_discrete:
            # Multiple discrete action branches
            self.discrete_heads = nn.ModuleList([
                linear_layer(hidden_size, action_size, kernel_init=Initialization.XavierGlorotUniform)
                for action_size in action_spec.discrete_branches
            ])
        
        if self.is_continuous:
            # Continuous action parameters (mean and std)
            self.continuous_mean = nn.Linear(hidden_size, action_spec.continuous_size)
            self.continuous_std = nn.Linear(hidden_size, action_spec.continuous_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        hidden_state: torch.Tensor, 
        stochastic_state: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the actor network
        """
        # Concatenate hidden and stochastic states
        latent_input = torch.cat([hidden_state, stochastic_state], dim=-1)
        
        # Pass through actor network
        features = self.actor_network(latent_input)
        
        actions = {}
        log_probs = {}
        
        # Handle discrete actions
        if self.is_discrete:
            discrete_actions = []
            discrete_log_probs = []
            
            for i, head in enumerate(self.discrete_heads):
                logits = head(features)
                
                if deterministic:
                    # Take argmax for deterministic action
                    action = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    # Sample from the distribution
                    probs = F.softmax(logits / self.discrete_temperature, dim=-1)
                    action = torch.multinomial(probs, 1)
                
                discrete_actions.append(action)
                discrete_log_probs.append(F.log_softmax(logits, dim=-1).gather(1, action))
            
            actions['discrete'] = torch.cat(discrete_actions, dim=-1)
            log_probs['discrete'] = torch.cat(discrete_log_probs, dim=-1)
        
        # Handle continuous actions
        if self.is_continuous:
            mean = torch.tanh(self.continuous_mean(features))
            std = F.softplus(self.continuous_std(features)) + 1e-6
            
            if deterministic:
                # Use mean for deterministic action
                action = mean
            else:
                # Sample from normal distribution
                normal = torch.randn_like(mean)
                action = mean + std * normal
                action = torch.tanh(action)  # Ensure action is in [-1, 1]
            
            actions['continuous'] = action
            # Calculate log probability for continuous action
            log_prob = -0.5 * (((action - mean) / std) ** 2 + 2.0 * torch.log(std) + torch.log(torch.tensor(2.0 * torch.pi, device=std.device)))
            log_probs['continuous'] = log_prob
        
        return {
            'actions': actions,
            'log_probs': log_probs,
            'continuous_mean': mean if self.is_continuous else None,
            'continuous_std': std if self.is_continuous else None
        }
    
    def action_and_log_prob(
        self, 
        hidden_state: torch.Tensor, 
        stochastic_state: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Return both sampled actions and their log probabilities
        """
        result = self.forward(hidden_state, stochastic_state, deterministic=False)
        return result['actions'], result['log_probs']
    
    def sample_action(
        self, 
        hidden_state: torch.Tensor, 
        stochastic_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Sample an action from the policy
        """
        return self.forward(hidden_state, stochastic_state, deterministic=False)['actions']
    
    def deterministic_action(
        self, 
        hidden_state: torch.Tensor, 
        stochastic_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get the deterministic action (mode) from the policy
        """
        return self.forward(hidden_state, stochastic_state, deterministic=True)['actions']