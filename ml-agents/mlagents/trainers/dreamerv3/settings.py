from typing import Dict, List, NamedTuple, Optional, Union
import attr
import numpy as np
from mlagents.trainers.settings import (
    OffPolicyHyperparamSettings,
    ScheduleType,
)


@attr.s(auto_attribs=True)
class DreamerV3Settings(OffPolicyHyperparamSettings):
    # World model hyperparameters
    wm_hidden_size: int = 256
    wm_latent_state_size: int = 60
    wm_stochastic_state_size: int = 32
    wm_reward_buckets: int = 1
    
    # Actor hyperparameters
    actor_hidden_size: int = 256
    actor_discrete_temperature: float = 1.0
    
    # Critic hyperparameters
    critic_hidden_size: int = 256
    critic_ensemble_size: int = 4
    
    # Training hyperparameters
    imagination_horizon: int = 15
    discount: float = 0.995
    lambda_return: float = 0.95
    world_model_loss_scale: float = 1.0
    reward_loss_scale: float = 1.0
    continue_loss_scale: float = 1.0
    kl_loss_scale: float = 1.0
    kl_free_nats: float = 3.0
    kl_free_avg: bool = True
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    world_model_lr: float = 1e-4
    grad_clip: float = 100.0
    value_target_tau: float = 0.005  # For target network updates
    train_world_model: bool = True
    train_actor: bool = True
    train_critic: bool = True
    update_actor_every: int = 1  # How often to update the actor
    update_critic_every: int = 1  # How often to update the critic
    update_world_model_every: int = 1  # How often to update the world model
    batch_size: int = 128  # Batch size for sampling from replay buffer
    sequence_length: int = 64  # Length of sequences to sample
    grad_heads: bool = True  # Whether to backprop gradients through heads
    free_nats: float = 3.0  # Free nats for KL regularization
    kl_balance: float = 0.8  # Balance between forward/backward KL
    kl_forward_scale: float = 1.0  # Scale for forward KL
    kl_backward_scale: float = 1.0  # Scale for backward KL
    model_update_frequency: int = 1  # How often to update world model
    
    # Override the default values for the base hyperparameters
    buffer_size: int = 10240  # Match other benchmarks
    steps_per_update: float = 0.5  # Update more frequently
