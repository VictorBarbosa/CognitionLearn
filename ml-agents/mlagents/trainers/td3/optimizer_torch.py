
import numpy as np
from typing import Dict, List, cast, Tuple, Optional
import attr

from mlagents.torch_utils import torch, nn, default_device

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.networks import ValueNetwork, SharedActorCritic
from mlagents.trainers.torch_entities.agent_action import AgentAction
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


@attr.s(auto_attribs=True)
class TD3Settings(OffPolicyHyperparamSettings):
    batch_size: int = 128
    buffer_size: int = 50000
    buffer_init_steps: int = 0
    tau: float = 0.005
    steps_per_update: float = 1
    save_replay_buffer: bool = False
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    noise_clip: float = 0.5
    reward_signal_steps_per_update: float = attr.ib()

    @reward_signal_steps_per_update.default
    def _reward_signal_steps_per_update_default(self):
        return self.steps_per_update


class TorchTD3Optimizer(TorchOptimizer):
    class PolicyValueNetwork(nn.Module):
        def __init__(
            self,
            stream_names: List[str],
            observation_specs: List[ObservationSpec],
            network_settings: NetworkSettings,
            action_spec: ActionSpec,
        ):
            super().__init__()
            num_value_outs = max(sum(action_spec.discrete_branches), 1)
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
            q1_grad: bool = True,
            q2_grad: bool = True,
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
            """
            Performs a forward pass on the value network, which consists of a Q1 and Q2
            network. Optionally does not evaluate gradients for either the Q1, Q2, or both.
            :param inputs: List of observation tensors.
            :param actions: For a continuous Q function (has actions), tensor of actions.
                Otherwise, None.
            :param memories: Initial memories if using memory. Otherwise, None.
            :param sequence_length: Sequence length if using memory.
            :param q1_grad: Whether or not to compute gradients for the Q1 network.
            :param q2_grad: Whether or not to compute gradients for the Q2 network.
            :return: Tuple of two dictionaries, which both map {reward_signal: Q} for Q1 and Q2,
                respectively.
            """
            # ExitStack allows us to enter the torch.no_grad() context conditionally
            with ExitStack() as stack:
                if not q1_grad:
                    stack.enter_context(torch.no_grad())
                q1_out, _ = self.q1_network(
                    inputs,
                    actions=actions,
                    memories=memories,
                    sequence_length=sequence_length,
                )
            with ExitStack() as stack:
                if not q2_grad:
                    stack.enter_context(torch.no_grad())
                q2_out, _ = self.q2_network(
                    inputs,
                    actions=actions,
                    memories=memories,
                    sequence_length=sequence_length,
                )
            return q1_out, q2_out

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]
        if isinstance(policy.actor, SharedActorCritic):
            raise UnityTrainerException("TD3 does not support SharedActorCritic")

        hyperparameters: TD3Settings = cast(
            TD3Settings, trainer_settings.hyperparameters
        )

        self.tau = hyperparameters.tau
        self.policy_delay = hyperparameters.policy_delay
        self.target_policy_noise = hyperparameters.target_policy_noise
        self.noise_clip = hyperparameters.noise_clip

        self.policy = policy
        policy_network_settings = policy.network_settings

        self.burn_in_ratio = 0.0

        self.stream_names = list(self.reward_signals.keys())
        # Use to reduce "survivor bonus" when using Curiosity or GAIL.
        self.gammas = [_val.gamma for _val in trainer_settings.reward_signals.values()]
        self.use_dones_in_backup = {
            name: int(not self.reward_signals[name].ignore_done)
            for name in self.stream_names
        }
        self._action_spec = self.policy.behavior_spec.action_spec

        self.q_network = TorchTD3Optimizer.PolicyValueNetwork(
            self.stream_names,
            self.policy.behavior_spec.observation_specs,
            policy_network_settings,
            self._action_spec,
        )
        self.target_q_network = TorchTD3Optimizer.PolicyValueNetwork(
            self.stream_names,
            self.policy.behavior_spec.observation_specs,
            policy_network_settings,
            self._action_spec,
        )
        ModelUtils.soft_update(self.q_network, self.target_q_network, 1.0)

        self.target_actor = type(self.policy.actor)(
            self.policy.behavior_spec.observation_specs,
            self.policy.network_settings,
            self.policy.behavior_spec.action_spec,
            **self.policy.actor_kwargs
        ).to(default_device())
        ModelUtils.soft_update(self.policy.actor, self.target_actor, 1.0)

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
        self._move_to_device(default_device())
        self.update_counter = 0

    def _move_to_device(self, device: torch.device) -> None:
        self.q_network.to(device)
        self.target_q_network.to(device)
        if hasattr(self, "target_actor"):
            self.target_actor.to(device)

    def q_loss(
        self,
        q1_out: Dict[str, torch.Tensor],
        q2_out: Dict[str, torch.Tensor],
        target_values: Dict[str, torch.Tensor],
        dones: torch.Tensor,
        rewards: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1_losses = []
        q2_losses = []
        # Multiple q losses per stream
        for i, name in enumerate(q1_out.keys()):
            q1_stream = q1_out[name].squeeze()
            q2_stream = q2_out[name].squeeze()
            with torch.no_grad():
                q_backup = rewards[name] + (
                    (1.0 - self.use_dones_in_backup[name] * dones)
                    * self.gammas[i]
                    * target_values[name]
                )
            _q1_loss = 0.5 * ModelUtils.masked_mean(
                torch.nn.functional.mse_loss(q_backup, q1_stream), loss_masks
            )
            _q2_loss = 0.5 * ModelUtils.masked_mean(
                torch.nn.functional.mse_loss(q_backup, q2_stream), loss_masks
            )

            q1_losses.append(_q1_loss)
            q2_losses.append(_q2_loss)
        q1_loss = torch.mean(torch.stack(q1_losses))
        q2_loss = torch.mean(torch.stack(q2_losses))
        return q1_loss, q2_loss

    def policy_loss(
        self,
        q1p_outs: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
    ) -> torch.Tensor:
        mean_q1 = torch.mean(torch.stack(list(q1p_outs.values())), axis=0)
        policy_loss = -ModelUtils.masked_mean(mean_q1, loss_masks)
        return policy_loss

    def _condense_q_streams(
        self, q_output: Dict[str, torch.Tensor], discrete_actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        condensed_q_output = {}
        onehot_actions = ModelUtils.actions_to_onehot(
            discrete_actions, self._action_spec.discrete_branches
        )
        for key, item in q_output.items():
            branched_q = ModelUtils.break_into_branches(
                item, self._action_spec.discrete_branches
            )
            only_action_qs = torch.stack(
                [
                    torch.sum(_act * _q, dim=1, keepdim=True)
                    for _act, _q in zip(onehot_actions, branched_q)
                ]
            )

            condensed_q_output[key] = torch.mean(only_action_qs, dim=0)
        return condensed_q_output

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        self.update_counter += 1
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

        actions = AgentAction.from_buffer(batch)

        memories_list = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]

        if len(memories_list) > 0:
            memories = torch.stack(memories_list).unsqueeze(0)
        else:
            memories = None

        q_memories = (
            torch.zeros_like(memories) if memories is not None else None
        )

        self.q_network.q1_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.q_network.q2_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.target_q_network.q1_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.target_q_network.q2_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )

        with torch.no_grad():
            next_action, _, _ = self.target_actor.get_action_and_stats(
                next_obs, memories=memories, sequence_length=self.policy.sequence_length
            )
            noise = torch.randn_like(next_action.continuous_tensor) * self.target_policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            noisy_continuous_action = next_action.continuous_tensor + noise
            noisy_continuous_action = torch.clamp(noisy_continuous_action, -1, 1)

            target_q1, target_q2 = self.target_q_network(
                next_obs, noisy_continuous_action, memories=q_memories, sequence_length=self.policy.sequence_length
            )
            target_q = torch.min(list(target_q1.values())[0], list(target_q2.values())[0])

        q1_out, q2_out = self.q_network(
            current_obs,
            actions.continuous_tensor,
            memories=q_memories,
            sequence_length=self.policy.sequence_length,
        )

        masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)
        dones = ModelUtils.list_to_tensor(batch[BufferKey.DONE])

        q1_loss, q2_loss = self.q_loss(
            q1_out, q2_out, {self.stream_names[0]: target_q}, dones, rewards, masks
        )
        total_value_loss = q1_loss + q2_loss

        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        ModelUtils.update_learning_rate(self.value_optimizer, decay_lr)
        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        self.value_optimizer.step()

        update_stats = {
            "Losses/Value Loss": total_value_loss.item(),
            "Losses/Q1 Loss": q1_loss.item(),
            "Losses/Q2 Loss": q2_loss.item(),
            "Policy/Learning Rate": decay_lr,
        }

        if self.update_counter % self.policy_delay == 0:
            sampled_actions, _, _ = self.policy.actor.get_action_and_stats(
                current_obs,
                memories=memories,
                sequence_length=self.policy.sequence_length,
            )
            q1p_out, _ = self.q_network(
                current_obs,
                sampled_actions.continuous_tensor,
                memories=q_memories,
                sequence_length=self.policy.sequence_length,
                q2_grad=False,
            )
            p_loss = self.policy_loss(q1p_out, masks)
            ModelUtils.update_learning_rate(self.policy_optimizer, decay_lr)
            self.policy_optimizer.zero_grad()
            p_loss.backward()
            self.policy_optimizer.step()
            ModelUtils.soft_update(self.policy.actor, self.target_actor, self.tau)
            ModelUtils.soft_update(self.q_network, self.target_q_network, self.tau)
            update_stats["Losses/Policy Loss"] = p_loss.item()

        return update_stats

    def get_modules(self):
        modules = {
            "Optimizer:q_network": self.q_network,
            "Optimizer:target_q_network": self.target_q_network,
            "Optimizer:policy_optimizer": self.policy_optimizer,
            "Optimizer:value_optimizer": self.value_optimizer,
            "Optimizer:target_actor": self.target_actor,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules
