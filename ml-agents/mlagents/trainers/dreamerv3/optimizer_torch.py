from typing import Dict, cast, List, Tuple, Optional
import attr
import numpy as np
import torch.nn.functional as F

from mlagents.torch_utils import torch, default_device

from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil, AgentBufferField
from mlagents_envs.timers import timed
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import (
    TrainerSettings,
    OffPolicyHyperparamSettings,
    ScheduleType,
)
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.dreamerv3.world_model import DreamerV3WorldModel
from mlagents.trainers.dreamerv3.actor import DreamerV3Actor
from mlagents.trainers.dreamerv3.critic import DreamerV3Critic
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs


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

class TorchDreamerV3Optimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        """
        Creates an optimizer for the DreamerV3 algorithm.
        :param policy: A TorchPolicy object that will be updated by this optimizer.
        :param trainer_params: Trainer parameters dictionary that specifies the properties of the trainer.
        """
        super().__init__(policy, trainer_settings)
        
        self.hyperparameters: DreamerV3Settings = cast(
            DreamerV3Settings, trainer_settings.hyperparameters
        )
        
        # Extract action size from the policy's behavior spec
        action_spec = policy.behavior_spec.action_spec
        if action_spec.continuous_size > 0:
            self.action_size = action_spec.continuous_size
        else:
            self.action_size = sum(action_spec.discrete_branches)
        
        # Create World Model
        self.world_model = DreamerV3WorldModel(
            observation_specs=policy.behavior_spec.observation_specs,
            action_size=self.action_size,
            hidden_size=self.hyperparameters.wm_hidden_size,
            latent_state_size=self.hyperparameters.wm_latent_state_size,
            stochastic_state_size=self.hyperparameters.wm_stochastic_state_size,
            reward_buckets=self.hyperparameters.wm_reward_buckets
        ).to(default_device())
        
        # Create Actor
        self.actor = DreamerV3Actor(
            action_spec=action_spec,
            latent_state_size=self.hyperparameters.wm_latent_state_size,
            stochastic_state_size=self.hyperparameters.wm_stochastic_state_size,
            hidden_size=self.hyperparameters.actor_hidden_size,
            discrete_temperature=self.hyperparameters.actor_discrete_temperature
        ).to(default_device())
        
        # Create Critic
        self._critic = DreamerV3Critic(
            latent_state_size=self.hyperparameters.wm_latent_state_size,
            stochastic_state_size=self.hyperparameters.wm_stochastic_state_size,
            hidden_size=self.hyperparameters.critic_hidden_size,
            ensemble_size=self.hyperparameters.critic_ensemble_size
        ).to(default_device())
        
        # Create target networks for critic (for stable learning)
        self.target_critic = DreamerV3Critic(
            latent_state_size=self.hyperparameters.wm_latent_state_size,
            stochastic_state_size=self.hyperparameters.wm_stochastic_state_size,
            hidden_size=self.hyperparameters.critic_hidden_size,
            ensemble_size=self.hyperparameters.critic_ensemble_size
        ).to(default_device())
        
        # Copy weights to target networks
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self._critic.parameters()):
                target_param.copy_(param)
        
        # Define optimizers for different components
        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(), 
            lr=self.hyperparameters.world_model_lr
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 
            lr=self.hyperparameters.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self._critic.parameters(), 
            lr=self.hyperparameters.critic_lr
        )
        
        # Stats to track during training
        self.stats_name_to_update_name = {
            "Losses/World Model Loss": "world_model_loss",
            "Losses/Actor Loss": "actor_loss", 
            "Losses/Critic Loss": "critic_loss",
            "Policy/World Model Reward Loss": "wm_reward_loss",
            "Policy/World Model Observation Loss": "wm_obs_loss",
            "Policy/KL Divergence": "kl_divergence"
        }

    def _kl_divergence(self, posterior_params: torch.Tensor, prior_params: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between posterior and prior stochastic states"""
        post_mean, post_std = torch.chunk(posterior_params, 2, dim=-1)
        post_std = F.softplus(post_std) + 1e-6
        
        prior_mean, prior_std = torch.chunk(prior_params, 2, dim=-1)
        prior_std = F.softplus(prior_std) + 1e-6
        
        # Compute KL divergence between two normal distributions
        kl = torch.log(prior_std / post_std) + \
             (post_std ** 2 + (post_mean - prior_mean) ** 2) / (2 * prior_std ** 2) - 0.5
        
        # Apply free bits to regularize KL
        if self.hyperparameters.kl_free_avg:
            kl = torch.max(kl, torch.tensor(self.hyperparameters.kl_free_nats, device=kl.device))
        else:
            kl = torch.sum(torch.max(
                kl, 
                torch.tensor(self.hyperparameters.kl_free_nats, device=kl.device)
            ), dim=-1)
            
        return kl

    def compute_world_model_loss(self, batch: AgentBuffer) -> torch.Tensor:
        """Compute loss for the world model component"""
        # Extract observations and actions from buffer
        n_obs = len(self.policy.behavior_spec.observation_specs)
        obs_list = ObsUtil.from_buffer(batch, n_obs)
        
        # Convert observations to tensor format
        obs_tensors = [ModelUtils.list_to_tensor(obs, dtype=torch.float32).to(default_device()) for obs in obs_list]

        # Combine observations if multiple specs exist
        if len(obs_tensors) > 1:
            obs = torch.cat(obs_tensors, dim=-1)
        else:
            obs = obs_tensors[0]
        
        # Get actions from buffer
        actions = AgentAction.from_buffer(batch)
        
        # Process continuous and discrete actions separately
        continuous_actions = actions.continuous_tensor if actions.continuous_tensor is not None else torch.empty(0)
        discrete_actions = actions.discrete_tensor if actions.discrete_list is not None else torch.empty(0)
        
        # Concatenate actions
        if continuous_actions.numel() > 0 and discrete_actions.numel() > 0:
            action_tensor = torch.cat([continuous_actions, discrete_actions.float()], dim=-1).to(default_device())
        elif continuous_actions.numel() > 0:
            action_tensor = continuous_actions.to(default_device())
        elif discrete_actions.numel() > 0:
            action_tensor = discrete_actions.float().to(default_device())
        else:
            # If no actions are available, create a dummy tensor with the right size
            action_spec = self.policy.behavior_spec.action_spec
            action_size = action_spec.continuous_size + sum(action_spec.discrete_branches)
            action_tensor = torch.zeros((len(batch[BufferKey.ENVIRONMENT_REWARDS]), action_size), device=obs.device)
        
        # Reshape to (batch, sequence, features)
        seq_len = self.policy.sequence_length
        batch_size = obs.shape[0] // seq_len
        obs = obs.view(batch_size, seq_len, -1)
        action_tensor = action_tensor.view(batch_size, seq_len, -1)
        
        # Forward pass through world model
        wm_output = self.world_model(
            observations=obs,
            actions=action_tensor
        )
        
        # Compute reconstruction loss for observations
        target_obs = obs
        recon_loss = F.mse_loss(wm_output['reconstructions'], target_obs)
        
        # Compute reward prediction loss
        target_rewards = ModelUtils.list_to_tensor(
            batch[BufferKey.ENVIRONMENT_REWARDS], 
            dtype=torch.float32
        ).view(batch_size, seq_len, -1).to(default_device())
        reward_loss = F.mse_loss(wm_output['continuous_rewards'], target_rewards)
        
        # Compute KL divergence loss between posterior and prior
        kl_loss = self._kl_divergence(
            wm_output['posterior_params'], wm_output['prior_params']
        ).mean()
        
        # Combine losses
        total_loss = (
            self.hyperparameters.world_model_loss_scale * recon_loss +
            self.hyperparameters.reward_loss_scale * reward_loss +
            self.hyperparameters.kl_loss_scale * kl_loss
        )
        
        return total_loss, recon_loss.item(), reward_loss.item(), kl_loss.item()

    def compute_actor_loss(self, batch: AgentBuffer) -> torch.Tensor:
        """Compute loss for the actor component using imagined trajectories"""
        # This is the most complex part of DreamerV3 - using imagined trajectories 
        # to optimize the policy
        with torch.no_grad():
            # Get observations from buffer
            n_obs = len(self.policy.behavior_spec.observation_specs)
            obs_list = ObsUtil.from_buffer(batch, n_obs)
            
            # Combine observations if multiple specs exist
            if len(obs_list) > 1:
                full_obs = torch.tensor(np.concatenate(obs_list, axis=-1), dtype=torch.float32).to(default_device())
            else:
                full_obs = ModelUtils.list_to_tensor(obs_list[0], dtype=torch.float32).to(default_device())
            
            # Reshape observations to get the initial states for imagination
            seq_len = self.policy.sequence_length
            batch_size = full_obs.shape[0] // seq_len
            obs = full_obs.view(batch_size, seq_len, -1)
            
            # Use world model to encode current observations to latent states
            # We'll use the last states from each sequence as initial states for imagination
            actions = AgentAction.from_buffer(batch)
            continuous_actions = actions.continuous_tensor.to(default_device()) if actions.continuous_tensor is not None else torch.empty(0)
            discrete_actions = actions.discrete_tensor.to(default_device()) if actions.discrete_list is not None else torch.empty(0)
            
            if continuous_actions.numel() > 0 and discrete_actions.numel() > 0:
                action_tensor = torch.cat([continuous_actions, discrete_actions.float()], dim=-1)
            elif continuous_actions.numel() > 0:
                action_tensor = continuous_actions
            elif discrete_actions.numel() > 0:
                action_tensor = discrete_actions.float()
            else:
                action_spec = self.policy.behavior_spec.action_spec
                action_size = action_spec.continuous_size + sum(action_spec.discrete_branches)
                action_tensor = torch.zeros((full_obs.shape[0], action_size), device=obs.device)
            
            action_tensor = action_tensor.view(batch_size, seq_len, -1)
            
            # Run world model to get initial latent states
            wm_output = self.world_model(
                observations=obs,
                actions=action_tensor
            )
            initial_hidden = wm_output['hidden_states'][:, -1, :]  # Use last state from each sequence
            initial_stochastic = wm_output['stochastic_states'][:, -1, :]
        
        # Perform imagination (rollout in latent space)
        imagined_hidden = [initial_hidden]
        imagined_stochastic = [initial_stochastic]
        imagined_rewards = []
        imagined_values = []
        
        current_hidden = initial_hidden
        current_stochastic = initial_stochastic
        
        for t in range(self.hyperparameters.imagination_horizon):
            # Get action from actor in the latent state
            with torch.no_grad():
                actor_actions, _ = self.actor.action_and_log_prob(
                    current_hidden, current_stochastic
                )
            
            # Convert actions to tensor format
            continuous_action = actor_actions['continuous'] if actor_actions.get('continuous') is not None else torch.empty(0)
            discrete_action = actor_actions['discrete'] if actor_actions.get('discrete') is not None else torch.empty(0)

            if continuous_action.numel() > 0 and discrete_action.numel() > 0:
                action_tensor = torch.cat([continuous_action, discrete_action], dim=-1)
            elif continuous_action.numel() > 0:
                action_tensor = continuous_action
            elif discrete_action.numel() > 0:
                action_tensor = discrete_action
            else:
                action_tensor = torch.zeros((current_hidden.shape[0], self.action_size), device=current_hidden.device)
            
            # Predict next state using world model transition
            next_hidden, next_stochastic, _ = self.world_model.transition_step(
                action_tensor, 
                current_hidden
            )
            
            # Get reward prediction from world model
            reward_input = torch.cat([next_hidden, next_stochastic], dim=-1)
            pred_rewards = self.world_model.reward_head(reward_input)
            pred_cont_rewards = self.world_model.continuous_reward_head(reward_input)
            
            # Use continuous rewards for actor loss
            rewards = pred_cont_rewards.squeeze(-1)
            
            # Get value prediction for advantage computation
            value_inputs = [torch.cat([next_hidden, next_stochastic], dim=-1)]
            value_outputs, _ = self.critic.critic_pass(value_inputs)
            values = list(value_outputs.values())[0]  # Get first reward stream value
            
            # Store imagined trajectory
            imagined_hidden.append(next_hidden)
            imagined_stochastic.append(next_stochastic)
            imagined_rewards.append(rewards)
            imagined_values.append(values)
            
            # Update current state for next step
            current_hidden = next_hidden
            current_stochastic = next_stochastic
        
        # Compute lambda returns for advantages
        lambda_returns = [imagined_values[-1]]  # Start with last value
        for t in reversed(range(len(imagined_rewards))):
            reward = imagined_rewards[t]
            value = imagined_values[t] if t < len(imagined_values) else 0
            next_return = lambda_returns[-1]
            
            # Compute lambda return: r + gamma * (lambda * next_return + (1-lambda) * V(s'))
            lambda_return = reward + self.hyperparameters.discount * (
                self.hyperparameters.lambda_return * next_return + 
                (1 - self.hyperparameters.lambda_return) * value
            )
            lambda_returns.append(lambda_return)
        
        # Reverse to get correct order
        lambda_returns = list(reversed(lambda_returns[:-1]))  # Remove the initial value
        
        # Compute critic loss
        lambda_returns_tensor = torch.stack(lambda_returns, dim=1).detach()
        imagined_values_tensor = torch.stack(imagined_values, dim=1).squeeze(-1)
        critic_loss = F.mse_loss(imagined_values_tensor, lambda_returns_tensor)

        # Compute actor loss (gradient through policy, but not through world model prediction)
        actor_loss = 0
        for t in range(len(lambda_returns)):
            # Use log probabilities from actor
            actions, log_probs_dict = self.actor.action_and_log_prob(
                imagined_hidden[t], imagined_stochastic[t]
            )

            # Compute log probabilities for the actions taken
            log_probs_tensors = []
            if log_probs_dict.get("continuous") is not None:
                log_probs_tensors.append(log_probs_dict["continuous"])
            if log_probs_dict.get("discrete") is not None:
                log_probs_tensors.append(log_probs_dict["discrete"])
            log_probs = torch.cat(log_probs_tensors, dim=-1)
            
            # Apply lambda return as advantage
            advantage = lambda_returns[t]
            
            # Actor loss is negative objective (we want to maximize expected return)
            actor_loss -= (log_probs.sum(dim=-1) * advantage.detach()).mean()
        
        # Add entropy regularization
        # with torch.no_grad():
        #     entropy_inputs = [torch.cat([initial_hidden, initial_stochastic], dim=-1)]
        #     _, entropy_out, _ = self.actor.get_action_and_stats(
        #         entropy_inputs,
        #         masks=None,
        #         memories=None,
        #         sequence_length=1
        #     )
        # 
        # entropy = entropy_out['entropy']
        # actor_loss -= self.hyperparameters.kl_loss_scale * entropy.mean()
        
        return actor_loss, critic_loss

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Performs update on model using the DreamerV3 approach.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        # Update World Model
        wm_loss, wm_obs_loss, wm_reward_loss, kl_divergence = self.compute_world_model_loss(batch)
        self.world_model_optimizer.zero_grad()
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.hyperparameters.grad_clip)
        self.world_model_optimizer.step()
        
        # Update Actor and Critic from imagination
        actor_loss, critic_loss = self.compute_actor_loss(batch)

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hyperparameters.grad_clip)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hyperparameters.grad_clip)
        self.critic_optimizer.step()
        
        # Soft update target networks
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.copy_(
                    self.hyperparameters.value_target_tau * param +
                    (1 - self.hyperparameters.value_target_tau) * target_param
                )
        
        # Return update statistics
        update_stats = {
            "Losses/World Model Loss": wm_loss.item(),
            "Losses/Critic Loss": critic_loss.item(),
            "Losses/Actor Loss": actor_loss.item() if actor_loss.requires_grad else 0.0,
            "Policy/World Model Observation Loss": wm_obs_loss,
            "Policy/World Model Reward Loss": wm_reward_loss,
            "Policy/KL Divergence": kl_divergence,
            "Policy/Learning Rate": self.hyperparameters.world_model_lr,
        }
        
        return update_stats

    def get_trajectory_value_estimates(
        self,
        batch: AgentBuffer,
        next_obs_list: List[np.ndarray],
        done: bool,
        agent_id: str = "",
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Optional[AgentBufferField]]:

        # 1. Get observations and actions from batch
        n_obs = len(self.policy.behavior_spec.observation_specs)
        obs_list = ObsUtil.from_buffer(batch, n_obs)
        obs_tensors = [ModelUtils.list_to_tensor(obs).to(default_device()) for obs in obs_list]
        if len(obs_tensors) > 1:
            obs_tensor = torch.cat(obs_tensors, dim=-1)
        else:
            obs_tensor = obs_tensors[0]

        actions = AgentAction.from_buffer(batch)
        # The world model expects a flat action tensor
        action_tensor_list = []
        if actions.continuous_tensor is not None:
            action_tensor_list.append(actions.continuous_tensor)
        if actions.discrete_tensor is not None and actions.discrete_tensor.numel() > 0:
            action_tensor_list.append(actions.discrete_tensor.float())
        
        if len(action_tensor_list) > 0:
            action_tensor = torch.cat(action_tensor_list, dim=1).to(default_device())
        else:
            action_tensor = torch.empty((obs_tensor.shape[0], 0), device=obs_tensor.device)


        # 2. Encode trajectory observations to latent states
        with torch.no_grad():
            # Reshape to (batch_size=1, seq_len, features) for world model
            obs_tensor_seq = obs_tensor.unsqueeze(0)
            action_tensor_seq = action_tensor.unsqueeze(0)

            wm_output = self.world_model(
                observations=obs_tensor_seq,
                actions=action_tensor_seq
            )
            hidden_states = wm_output['hidden_states'].squeeze(0)
            stochastic_states = wm_output['stochastic_states'].squeeze(0)

        # 3. Pass latent states to the critic to get value estimates for the trajectory
        with torch.no_grad():
            # The DreamerV3Critic expects a list of tensors.
            # The first is hidden, second is stochastic.
            value_estimates, _ = self.critic.critic_pass([hidden_states, stochastic_states])

        # 4. Handle next_obs for bootstrapping value
        with torch.no_grad():
            # Get last hidden state from the sequence
            last_hidden = wm_output['hidden_states'][:, -1, :] # Shape (1, latent_state_size)

            # Encode the next_obs to get its embedding
            next_obs_tensors = [ModelUtils.list_to_tensor(obs).to(default_device()) for obs in next_obs_list]
            flattened_tensors = [obs.reshape(-1) for obs in next_obs_tensors]
            next_obs_tensor = torch.cat(flattened_tensors, dim=-1)
            next_obs_tensor = next_obs_tensor.reshape(1, -1)
            next_obs_embed = self.world_model.obs_encoder(next_obs_tensor)

            # Get last action from the trajectory
            last_action = action_tensor[-1, :].unsqueeze(0)  # Shape (1, action_size)

            # ``representation_step`` expects both ``obs_embed`` and ``prev_action``
            # to share the same batch dimension. ``next_obs_embed`` is already
            # batched, so we pass it directly (no extra ``unsqueeze``).
            next_hidden, next_stochastic, _, _ = self.world_model.representation_step(
                next_obs_embed,
                last_action,
                last_hidden,
            )

            # Get value for this next latent state
            next_value_estimate, _ = self.critic.critic_pass([next_hidden, next_stochastic])

        # Convert to numpy and handle done state
        for name, estimate in value_estimates.items():
            value_estimates[name] = ModelUtils.to_numpy(estimate)
        for name, estimate in next_value_estimate.items():
            next_value_estimate[name] = ModelUtils.to_numpy(estimate).item()

        if done:
            for k in next_value_estimate:
                if not self.reward_signals[k].ignore_done:
                    next_value_estimate[k] = 0.0

        # DreamerV3 critic doesn't use memories in the same way as LSTM-based critics
        return value_estimates, next_value_estimate, None

    def get_modules(self):
        """Return the modules managed by this optimizer"""
        modules = {
            "Optimizer/world_model_optimizer": self.world_model_optimizer,
            "Optimizer/actor_optimizer": self.actor_optimizer,
            "Optimizer/critic_optimizer": self.critic_optimizer,
            "Optimizer/world_model": self.world_model,
            "Optimizer/actor": self.actor,
            "Optimizer/critic": self._critic,
            "Optimizer/target_critic": self.target_critic,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules

    @property
    def critic(self):
        return self._critic
