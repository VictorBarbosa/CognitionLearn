import numpy as np
from typing import Dict, List, NamedTuple, cast, Tuple, Optional
import attr

from mlagents.torch_utils import torch, nn, default_device

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.networks import ValueNetwork, SharedActorCritic
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents_envs.timers import timed
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.settings import TrainerSettings, OffPolicyHyperparamSettings
from contextlib import ExitStack
from mlagents.trainers.trajectory import ObsUtil

EPSILON = 1e-6  # Small value to avoid divide by zero

logger = get_logger(__name__)

def quantile_huber_loss(
    quantiles: torch.Tensor, samples: torch.Tensor, huber_param: float = 1.0
) -> torch.Tensor:
    """
    Calculates the quantile huber loss.
    :param quantiles: The quantiles that are being learned.
    :param samples: The samples that the quantiles are learning from.
    :return: The quantile huber loss.
    """
    # Add a dimension to samples and quantiles tensors
    samples = samples.unsqueeze(dim=2)
    quantiles = quantiles.unsqueeze(dim=1)

    # Get the absolute errors and huber loss
    abs_errors = torch.abs(samples - quantiles)
    huber_loss = torch.where(
        abs_errors < huber_param,
        0.5 * abs_errors ** 2,
        huber_param * (abs_errors - 0.5 * huber_param),
    )

    # Get the huber loss
    n_quantiles = quantiles.shape[2]
    tau = (
        torch.arange(n_quantiles, device=quantiles.device).float() + 0.5
    ) / n_quantiles
    huber_loss = (
        torch.abs(tau - (samples - quantiles < 0).float()) * huber_loss
    ).mean()
    return huber_loss


@attr.s(auto_attribs=True)
class TQCSettings(OffPolicyHyperparamSettings):
    batch_size: int = 128
    buffer_size: int = 50000
    buffer_init_steps: int = 0
    tau: float = 0.005
    steps_per_update: float = 1
    save_replay_buffer: bool = False
    init_entcoef: float = 1.0
    reward_signal_steps_per_update: float = attr.ib()
    n_quantiles: int = 25
    n_to_drop: int = 2

    @reward_signal_steps_per_update.default
    def _reward_signal_steps_per_update_default(self):
        return self.steps_per_update


class TorchTQCOptimizer(TorchOptimizer):
    class PolicyValueNetwork(nn.Module):
        def __init__(
            self,
            stream_names: List[str],
            observation_specs: List[ObservationSpec],
            network_settings: NetworkSettings,
            action_spec: ActionSpec,
            n_quantiles: int,
        ):
            super().__init__()
            num_value_outs = max(sum(action_spec.discrete_branches), 1) * n_quantiles
            num_action_ins = int(action_spec.continuous_size)

            self.q1_network = ValueNetwork(
                stream_names,
                observation_specs,
                network_settings,
                num_action_ins,
                num_value_outs,
            )
            self.q2_network = ValueNetwork(
                stream_names,
                observation_specs,
                network_settings,
                num_action_ins,
                num_value_outs,
            )

        def forward(
            self,
            inputs: List[torch.Tensor],
            actions: Optional[torch.Tensor] = None,
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
            q1_out, _ = self.q1_network(
                inputs,
                actions=actions,
                memories=memories,
                sequence_length=sequence_length,
            )
            q2_out, _ = self.q2_network(
                inputs,
                actions=actions,
                memories=memories,
                sequence_length=sequence_length,
            )
            return q1_out, q2_out

    class TargetEntropy(NamedTuple):
        discrete: List[float] = []  # One per branch
        continuous: float = 0.0

    class LogEntCoef(nn.Module):
        def __init__(self, discrete, continuous):
            super().__init__()
            self.discrete = discrete
            self.continuous = continuous

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]
        if isinstance(policy.actor, SharedActorCritic):
            raise UnityTrainerException("TQC does not support SharedActorCritic")

        hyperparameters: TQCSettings = cast(
            TQCSettings, trainer_settings.hyperparameters
        )

        self.n_quantiles = hyperparameters.n_quantiles
        self.n_to_drop = hyperparameters.n_to_drop

        self.tau = hyperparameters.tau
        self.init_entcoef = hyperparameters.init_entcoef

        self.policy = policy
        policy_network_settings = policy.network_settings

        self.burn_in_ratio = 0.0

        # Non-exposed TQC parameters
        self.discrete_target_entropy_scale = 0.2  # Roughly equal to e-greedy 0.05
        self.continuous_target_entropy_scale = 1.0

        self.stream_names = list(self.reward_signals.keys())
        self.gammas = [_val.gamma for _val in trainer_settings.reward_signals.values()]
        self.use_dones_in_backup = {
            name: int(not self.reward_signals[name].ignore_done)
            for name in self.stream_names
        }
        self._action_spec = self.policy.behavior_spec.action_spec

        self.q_network = self.PolicyValueNetwork(
            self.stream_names,
            self.policy.behavior_spec.observation_specs,
            policy_network_settings,
            self._action_spec,
            self.n_quantiles,
        )

        self.target_q_network = self.PolicyValueNetwork(
            self.stream_names,
            self.policy.behavior_spec.observation_specs,
            policy_network_settings,
            self._action_spec,
            self.n_quantiles,
        )

        self.target_actor = type(self.policy.actor)(
            self.policy.behavior_spec.observation_specs,
            self.policy.network_settings,
            self.policy.behavior_spec.action_spec,
            **self.policy.actor_kwargs
        ).to(default_device())

        ModelUtils.soft_update(self.q_network, self.target_q_network, 1.0)
        ModelUtils.soft_update(self.policy.actor, self.target_actor, 1.0)

        _disc_log_ent_coef = torch.nn.Parameter(
            torch.log(
                torch.as_tensor(
                    [self.init_entcoef] * len(self._action_spec.discrete_branches)
                )
            ),
            requires_grad=True,
        )
        _cont_log_ent_coef = torch.nn.Parameter(
            torch.log(torch.as_tensor([self.init_entcoef])), requires_grad=True
        )
        self._log_ent_coef = self.LogEntCoef(
            discrete=_disc_log_ent_coef, continuous=_cont_log_ent_coef
        )
        _cont_target = (
            -1
            * self.continuous_target_entropy_scale
            * np.prod(self._action_spec.continuous_size).astype(np.float32)
        )
        _disc_target = [
            self.discrete_target_entropy_scale * np.log(i).astype(np.float32)
            for i in self._action_spec.discrete_branches
        ]
        self.target_entropy = self.TargetEntropy(
            continuous=_cont_target, discrete=_disc_target
        )
        policy_params = list(self.policy.actor.parameters())
        value_params = list(self.q_network.parameters())

        self.decay_learning_rate = ModelUtils.DecayedValue(
            hyperparameters.learning_rate_schedule,
            hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.policy_optimizer = torch.optim.Adam(
            policy_params, lr=hyperparameters.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            value_params, lr=hyperparameters.learning_rate
        )
        self.entropy_optimizer = torch.optim.Adam(
            self._log_ent_coef.parameters(), lr=hyperparameters.learning_rate
        )
        self._move_to_device(default_device())

    @property
    def critic(self):
        # TQC does not have a separate V-network like SAC, so we return one of the Q-networks for normalization purposes.
        return self.q_network.q1_network

    def _move_to_device(self, device: torch.device) -> None:
        self._log_ent_coef.to(device)
        self.q_network.to(device)
        self.target_q_network.to(device)
        self.target_actor.to(device)

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        rewards = {}
        for name in self.reward_signals:
            rewards[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.rewards_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        next_obs = ObsUtil.from_buffer_next(batch, n_obs)
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        memories = None # TQC doesn't support recurrence yet

        # Update Critic
        with torch.no_grad():
            next_action, next_log_prob_dict, _ = self.target_actor.get_action_and_stats(next_obs, memories=memories)
            next_log_probs = next_log_prob_dict['log_probs'].continuous_tensor
            target_q1_out, target_q2_out = self.target_q_network(next_obs, next_action.continuous_tensor, memories=memories)
            
            target_q_cat = torch.cat(
                (target_q1_out[self.stream_names[0]].unsqueeze(1), target_q2_out[self.stream_names[0]].unsqueeze(1)), dim=1
            )
            
            # Sort and drop top quantiles
            sorted_quantiles, _ = torch.sort(target_q_cat.reshape(actions.continuous_tensor.shape[0], -1))
            quantiles_to_keep = self.n_quantiles * 2 - self.n_to_drop
            truncated_quantiles = sorted_quantiles[:, :quantiles_to_keep]

            target_q = truncated_quantiles
            
            # TD target
            ent_term = self._log_ent_coef.continuous.exp() * -next_log_probs.sum(dim=1)
            target_q += ent_term.unsqueeze(1)
            q_backup = rewards[self.stream_names[0]].unsqueeze(1) + self.gammas[0] * (1 - ModelUtils.list_to_tensor(batch[BufferKey.DONE])).unsqueeze(1) * target_q

        q1_out, q2_out = self.q_network(current_obs, actions.continuous_tensor, memories=memories)
        q1_quantiles = q1_out[self.stream_names[0]]
        q2_quantiles = q2_out[self.stream_names[0]]

        q1_loss = quantile_huber_loss(q1_quantiles, q_backup)
        q2_loss = quantile_huber_loss(q2_quantiles, q_backup)
        total_value_loss = q1_loss + q2_loss

        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        self.value_optimizer.step()

        # Update Actor and Entropy
        sampled_actions, log_probs_dict, _ = self.policy.actor.get_action_and_stats(current_obs, memories=memories, masks=act_masks)
        log_probs = log_probs_dict['log_probs'].continuous_tensor
        
        q1_policy, q2_policy = self.q_network(current_obs, sampled_actions.continuous_tensor, memories=memories)
        # Take mean over quantiles to get a single Q value
        q1_policy_mean = q1_policy[self.stream_names[0]].mean(dim=1).squeeze()
        q2_policy_mean = q2_policy[self.stream_names[0]].mean(dim=1).squeeze()
        min_q = torch.min(q1_policy_mean, q2_policy_mean)

        policy_loss = (self._log_ent_coef.continuous.exp() * log_probs.sum(dim=1) - min_q).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update entropy coefficient
        with torch.no_grad():
            target_current_diff = (log_probs.sum(dim=1) + self.target_entropy.continuous)
        entropy_loss = -(self._log_ent_coef.continuous * target_current_diff).mean()

        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_optimizer.step()

        # Update target networks
        ModelUtils.soft_update(self.q_network, self.target_q_network, self.tau)
        ModelUtils.soft_update(self.policy.actor, self.target_actor, self.tau)

        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        ModelUtils.update_learning_rate(self.policy_optimizer, decay_lr)
        ModelUtils.update_learning_rate(self.value_optimizer, decay_lr)
        ModelUtils.update_learning_rate(self.entropy_optimizer, decay_lr)

        update_stats = {
            "Losses/Policy Loss": policy_loss.item(),
            "Losses/Q1 Loss": q1_loss.item(),
            "Losses/Q2 Loss": q2_loss.item(),
            "Policy/Continuous Entropy Coeff": self._log_ent_coef.continuous.exp().item(),
            "Policy/Learning Rate": decay_lr,
        }

        return update_stats

    def get_modules(self):
        modules = {
            "Optimizer:q_network": self.q_network,
            "Optimizer:target_q_network": self.target_q_network,
            "Optimizer:policy_optimizer": self.policy_optimizer,
            "Optimizer:value_optimizer": self.value_optimizer,
            "Optimizer:entropy_optimizer": self.entropy_optimizer,
            "Optimizer:target_actor": self.target_actor,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules