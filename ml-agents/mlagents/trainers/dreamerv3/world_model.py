from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlagents.torch_utils import torch as torch_utils
from mlagents.trainers.torch_entities.layers import linear_layer, Initialization
from mlagents.trainers.torch_entities.encoders import VectorInput
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents_envs.base_env import ObservationSpec


class DreamerV3WorldModel(nn.Module):
    """
    Implements the World Model for DreamerV3 which includes an RSSM (Recurrent State Space Model)
    and an observation encoder/decoder.
    """
    
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        action_size: int,
        hidden_size: int = 400,
        latent_state_size: int = 60,
        stochastic_state_size: int = 32,
        reward_buckets: int = 1
    ):
        super().__init__()
        
        self.observation_specs = observation_specs
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.latent_state_size = latent_state_size
        self.stochastic_state_size = stochastic_state_size

        # Calculate total observation size
        self.obs_size = sum([np.prod(spec.shape) for spec in observation_specs])
        
        # Observation encoder
        self.obs_encoder = self._build_obs_encoder()
        
        # Observation decoder
        self.obs_decoder = self._build_obs_decoder()
        
        # RSSM components
        self.recurrent_model = self._build_recurrent_model()
        self.representation_model = self._build_representation_model()
        self.transition_model = self._build_transition_model()
        
        # Reward prediction head
        reward_input_size = latent_state_size + stochastic_state_size
        self.reward_head = nn.Sequential(
            linear_layer(reward_input_size, hidden_size, kernel_init=Initialization.XavierGlorotUniform),
            nn.ELU(),
            linear_layer(hidden_size, hidden_size, kernel_init=Initialization.XavierGlorotUniform),
            nn.ELU(),
            linear_layer(hidden_size, reward_buckets, kernel_init=Initialization.Zero)
        )
        
        # Continue reward head for continuous reward
        self.continuous_reward_head = nn.Sequential(
            linear_layer(reward_input_size, hidden_size, kernel_init=Initialization.XavierGlorotUniform),
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
    
    def _build_obs_encoder(self) -> nn.Module:
        """Build visual encoder for observations"""
        # For simplicity, assuming we have one visual observation
        # In a complete implementation, we would handle multiple observation types
        if len(self.observation_specs) == 1:
            spec = self.observation_specs[0]
            if len(spec.shape) == 3:  # Visual observation
                # Use a simple CNN for visual encoding
                return nn.Sequential(
                    nn.Conv2d(spec.shape[0], 32, 4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 4, stride=2),
                    nn.ReLU(),
                    nn.Flatten(),
                    linear_layer(256 * 6 * 6, self.hidden_size, kernel_init=Initialization.XavierGlorotUniform),
                    nn.ELU()
                )
            else:  # Vector observation
                return nn.Sequential(
                    linear_layer(self.obs_size, self.hidden_size, kernel_init=Initialization.XavierGlorotUniform),
                    nn.ELU(),
                    linear_layer(self.hidden_size, self.hidden_size, kernel_init=Initialization.XavierGlorotUniform),
                    nn.ELU()
                )
        else:
            # For multiple observation types, we combine them
            return nn.Sequential(
                linear_layer(self.obs_size, self.hidden_size, kernel_init=Initialization.XavierGlorotUniform),
                nn.ELU(),
                linear_layer(self.hidden_size, self.hidden_size, kernel_init=Initialization.XavierGlorotUniform),
                nn.ELU()
            )
    
    def _build_obs_decoder(self) -> nn.Module:
        """Build visual decoder for observations"""
        # Simple decoder - inverse of encoder
        return nn.Sequential(
            linear_layer(self.latent_state_size + self.stochastic_state_size, 
                        self.hidden_size, kernel_init=Initialization.XavierGlorotUniform),
            nn.ELU(),
            linear_layer(self.hidden_size, self.hidden_size, kernel_init=Initialization.XavierGlorotUniform),
            nn.ELU(),
            linear_layer(self.hidden_size, self.obs_size, kernel_init=Initialization.XavierGlorotUniform),
            nn.Tanh()  # Assuming normalized observation space
        )
    
    def _build_recurrent_model(self) -> nn.Module:
        """Build recurrent model (GRU) for temporal dependencies"""
        return nn.GRUCell(
            input_size=self.hidden_size + self.action_size,
            hidden_size=self.latent_state_size
        )
    
    def _build_representation_model(self) -> nn.Module:
        """Build representation model that infers current state from observation and prior"""
        input_size = self.latent_state_size + self.hidden_size
        return nn.Sequential(
            linear_layer(input_size, self.hidden_size, kernel_init=Initialization.XavierGlorotUniform),
            nn.ELU(),
            linear_layer(self.hidden_size, self.hidden_size, kernel_init=Initialization.XavierGlorotUniform),
            nn.ELU(),
            self._get_dist_layer(self.hidden_size, self.stochastic_state_size)
        )
    
    def _build_transition_model(self) -> nn.Module:
        """Build transition model that predicts next state from current state and action"""
        input_size = self.latent_state_size
        return nn.Sequential(
            linear_layer(input_size, self.hidden_size, kernel_init=Initialization.XavierGlorotUniform),
            nn.ELU(),
            linear_layer(self.hidden_size, self.hidden_size, kernel_init=Initialization.XavierGlorotUniform),
            nn.ELU(),
            self._get_dist_layer(self.hidden_size, self.stochastic_state_size)
        )
    
    def _get_dist_layer(self, input_size: int, output_size: int) -> nn.Module:
        """Create layer to parameterize a distribution"""
        # For DreamerV3, we typically use a layer that outputs mean and std for a normal distribution
        return nn.Linear(input_size, 2 * output_size)
    
    def _sample_stochastic_state(self, dist_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample stochastic state from distribution parameters"""
        mean, std = torch.chunk(dist_params, 2, dim=-1)
        std = F.softplus(std) + 1e-6  # Ensure positive std
        normal = torch.randn_like(mean)
        sample = mean + std * normal
        return sample, mean
    
    def initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize hidden state for RSSM"""
        device = next(self.parameters()).device
        hidden = torch.zeros(batch_size, self.latent_state_size, device=device)
        stochastic = torch.zeros(batch_size, self.stochastic_state_size, device=device)
        mean = torch.zeros(batch_size, self.stochastic_state_size, device=device)
        return hidden, stochastic, mean
    
    def representation_step(
        self, 
        obs_embed: torch.Tensor, 
        prev_action: torch.Tensor,
        prev_hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform representation step to infer current state"""
        # Get hidden state using recurrent model
        hidden = self.recurrent_model(
            torch.cat([obs_embed, prev_action], dim=-1), 
            prev_hidden
        )
        
        # Infer posterior (representation) using current observation
        posterior_input = torch.cat([hidden, obs_embed], dim=-1)
        posterior_params = self.representation_model(posterior_input)
        posterior_sample, posterior_mean = self._sample_stochastic_state(posterior_params)
        
        return hidden, posterior_sample, posterior_mean, posterior_params
    
    def transition_step(
        self, 
        prev_action: torch.Tensor, 
        prev_hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform transition step to predict next state"""
        # Predict prior (transition) using only previous state
        prior_params = self.transition_model(prev_hidden)
        prior_sample, prior_mean = self._sample_stochastic_state(prior_params)
        
        # Get new hidden state (this is the transition)
        hidden = self.recurrent_model(
            torch.cat([torch.zeros((prev_action.shape[0], self.hidden_size), device=prev_action.device, dtype=prev_action.dtype), prev_action], dim=-1),
            prev_hidden
        )
        
        return hidden, prior_sample, prior_params
    
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the world model
        """
        batch_size, seq_len = actions.shape[:2]
        
        if prev_hidden is None:
            prev_hidden = torch.zeros(
                batch_size, self.latent_state_size, 
                device=actions.device, dtype=actions.dtype
            )
        
        # Encode observations
        obs_embed = self.obs_encoder(observations)
        
        # Initialize outputs
        hidden_states = []
        stochastic_states = []
        posterior_params_list = []
        prior_params_list = []
        rewards = []
        cont_rewards = []
        
        for t in range(seq_len):
            # Get current obs embed and action
            obs_t = obs_embed[:, t]
            action_t = actions[:, t]

            # Get prior from previous hidden state
            prior_params = self.transition_model(prev_hidden)
            
            # Representation step
            hidden, stochastic, mean, posterior_params = self.representation_step(
                obs_t, action_t, prev_hidden
            )
            
            # Get reward prediction
            reward_input = torch.cat([hidden, stochastic], dim=-1)
            reward_pred = self.reward_head(reward_input)
            cont_reward_pred = self.continuous_reward_head(reward_input)
            
            # Store results
            hidden_states.append(hidden)
            stochastic_states.append(stochastic)
            posterior_params_list.append(posterior_params)
            prior_params_list.append(prior_params)
            rewards.append(reward_pred)
            cont_rewards.append(cont_reward_pred)
            
            # Update for next step
            prev_hidden = hidden
        
        # Stack results
        hidden_states = torch.stack(hidden_states, dim=1)
        stochastic_states = torch.stack(stochastic_states, dim=1)
        posterior_params = torch.stack(posterior_params_list, dim=1)
        prior_params = torch.stack(prior_params_list, dim=1)
        rewards = torch.stack(rewards, dim=1)
        cont_rewards = torch.stack(cont_rewards, dim=1)
        
        # Decode observations
        latent_input = torch.cat([hidden_states, stochastic_states], dim=-1)
        reconstructed_obs = self.obs_decoder(latent_input)
        
        return {
            'hidden_states': hidden_states,
            'stochastic_states': stochastic_states,
            'rewards': rewards,
            'continuous_rewards': cont_rewards,
            'reconstructions': reconstructed_obs,
            'posterior_params': posterior_params,
            'prior_params': prior_params
        }