
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
from mlagents.trainers.sac.optimizer_torch import TorchSACOptimizer, SACSettings

EPSILON = 1e-6  # Small value to avoid divide by zero

logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class TDSACSettings(SACSettings):
    pass


class TorchTDSACOptimizer(TorchSACOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        self.target_q_network = TorchSACOptimizer.PolicyValueNetwork(
            self.stream_names,
            self.policy.behavior_spec.observation_specs,
            policy.network_settings,
            self._action_spec,
        )
        ModelUtils.soft_update(self.q_network, self.target_q_network, 1.0)
        self._move_to_device(default_device())

    def _move_to_device(self, device: torch.device) -> None:
        super()._move_to_device(device)
        self.target_q_network.to(device)

    def sac_value_loss(
        self,
        log_probs: ActionLogProbs,
        values: Dict[str, torch.Tensor],
        q1p_out: Dict[str, torch.Tensor],
        q2p_out: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
    ) -> torch.Tensor:
        min_policy_qs = {}
        with torch.no_grad():
            _cont_ent_coef = self._log_ent_coef.continuous.exp()
            _disc_ent_coef = self._log_ent_coef.discrete.exp()
            for name in values.keys():
                if self._action_spec.discrete_size <= 0:
                    min_policy_qs[name] = torch.min(q1p_out[name], q2p_out[name])
                else:
                    disc_action_probs = log_probs.all_discrete_tensor.exp()
                    _branched_q1p = ModelUtils.break_into_branches(
                        q1p_out[name] * disc_action_probs,
                        self._action_spec.discrete_branches,
                    )
                    _branched_q2p = ModelUtils.break_into_branches(
                        q2p_out[name] * disc_action_probs,
                        self._action_spec.discrete_branches,
                    )
                    _q1p_mean = torch.mean(
                        torch.stack(
                            [
                                torch.sum(_br, dim=1, keepdim=True)
                                for _br in _branched_q1p
                            ]
                        ),
                        dim=0,
                    )
                    _q2p_mean = torch.mean(
                        torch.stack(
                            [
                                torch.sum(_br, dim=1, keepdim=True)
                                for _br in _branched_q2p
                            ]
                        ),
                        dim=0,
                    )

                    min_policy_qs[name] = torch.min(_q1p_mean, _q2p_mean)

        value_losses = []
        if self._action_spec.discrete_size <= 0:
            for name in values.keys():
                with torch.no_grad():
                    v_backup = min_policy_qs[name] - torch.sum(
                        _cont_ent_coef * log_probs.continuous_tensor, dim=1
                    )
                value_loss = 0.5 * ModelUtils.masked_mean(
                    torch.nn.functional.mse_loss(values[name], v_backup), loss_masks
                )
                value_losses.append(value_loss)
        else:
            disc_log_probs = log_probs.all_discrete_tensor
            branched_per_action_ent = ModelUtils.break_into_branches(
                disc_log_probs * disc_log_probs.exp(),
                self._action_spec.discrete_branches,
            )
            # We have to do entropy bonus per action branch
            branched_ent_bonus = torch.stack(
                [
                    torch.sum(_disc_ent_coef[i] * _lp, dim=1, keepdim=True)
                    for i, _lp in enumerate(branched_per_action_ent)
                ]
            )
            for name in values.keys():
                with torch.no_grad():
                    v_backup = min_policy_qs[name] - torch.mean(
                        branched_ent_bonus, axis=0
                    )
                    # Add continuous entropy bonus to minimum Q
                    if self._action_spec.continuous_size > 0:
                        v_backup += torch.sum(
                            _cont_ent_coef * log_probs.continuous_tensor,
                            dim=1,
                            keepdim=True,
                        )
                value_loss = 0.5 * ModelUtils.masked_mean(
                    torch.nn.functional.mse_loss(values[name], v_backup.squeeze()),
                    loss_masks,
                )
                value_losses.append(value_loss)
        value_loss = torch.mean(torch.stack(value_losses))
        if torch.isinf(value_loss).any() or torch.isnan(value_loss).any():
            raise UnityTrainerException("Inf found")
        return value_loss

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Updates model using buffer.
        :param num_sequences: Number of trajectories in batch.
        :param batch: Experience mini-batch.
        :return: Output from update process.
        """
        rewards = {}
        for name in self.reward_signals:
            rewards[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.rewards_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        next_obs = ObsUtil.from_buffer_next(batch, n_obs)
        # Convert to tensors
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        memories_list = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        # LSTM shouldn't have sequence length <1, but stop it from going out of the index if true.
        value_memories_list = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(
                0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length
            )
        ]

        if len(memories_list) > 0:
            memories = torch.stack(memories_list).unsqueeze(0)
            value_memories = torch.stack(value_memories_list).unsqueeze(0)
        else:
            memories = None
            value_memories = None

        # Q and V network memories are 0'ed out, since we don't have them during inference.
        q_memories = (
            torch.zeros_like(value_memories) if value_memories is not None else None
        )

        # Copy normalizers from policy
        self.q_network.q1_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.q_network.q2_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.target_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.target_q_network.q1_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.target_q_network.q2_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self._critic.network_body.copy_normalization(self.policy.actor.network_body)
        sampled_actions, run_out, _, = self.policy.actor.get_action_and_stats(
            current_obs,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )
        log_probs = run_out["log_probs"]
        value_estimates, _ = self._critic.critic_pass(
            current_obs, value_memories, sequence_length=self.policy.sequence_length
        )

        cont_sampled_actions = sampled_actions.continuous_tensor
        cont_actions = actions.continuous_tensor
        q1p_out, q2p_out = self.q_network(
            current_obs,
            cont_sampled_actions,
            memories=q_memories,
            sequence_length=self.policy.sequence_length,
            q2_grad=False,
        )
        q1_out, q2_out = self.q_network(
            current_obs,
            cont_actions,
            memories=q_memories,
            sequence_length=self.policy.sequence_length,
        )

        if self._action_spec.discrete_size > 0:
            disc_actions = actions.discrete_tensor
            q1_stream = self._condense_q_streams(q1_out, disc_actions)
            q2_stream = self._condense_q_streams(q2_out, disc_actions)
        else:
            q1_stream, q2_stream = q1_out, q2_out

        with torch.no_grad():
            # Since we didn't record the next value memories, evaluate one step in the critic to
            # get them.
            if value_memories is not None:
                # Get the first observation in each sequence
                just_first_obs = [
                    _obs[:: self.policy.sequence_length] for _obs in current_obs
                ]
                _, next_value_memories = self._critic.critic_pass(
                    just_first_obs, value_memories, sequence_length=1
                )
            else:
                next_value_memories = None
            target_values, _ = self.target_network(
                next_obs,
                memories=next_value_memories,
                sequence_length=self.policy.sequence_length,
            )
            target_q1, target_q2 = self.target_q_network(
                next_obs,
                memories=next_value_memories,
                sequence_length=self.policy.sequence_length,
            )
            target_q = torch.min(list(target_q1.values())[0], list(target_q2.values())[0])
            for name in target_values:
                target_values[name] = target_q

        masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)
        dones = ModelUtils.list_to_tensor(batch[BufferKey.DONE])

        q1_loss, q2_loss = self.sac_q_loss(
            q1_stream, q2_stream, target_values, dones, rewards, masks
        )
        value_loss = self.sac_value_loss(
            log_probs, value_estimates, q1p_out, q2p_out, masks
        )
        policy_loss = self.sac_policy_loss(log_probs, q1p_out, masks)
        entropy_loss = self.sac_entropy_loss(log_probs, masks)

        total_value_loss = q1_loss + q2_loss + value_loss

        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        ModelUtils.update_learning_rate(self.policy_optimizer, decay_lr)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        ModelUtils.update_learning_rate(self.value_optimizer, decay_lr)
        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        self.value_optimizer.step()

        ModelUtils.update_learning_rate(self.entropy_optimizer, decay_lr)
        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_optimizer.step()

        # Update target network
        ModelUtils.soft_update(self._critic, self.target_network, self.tau)
        ModelUtils.soft_update(self.q_network, self.target_q_network, self.tau)
        update_stats = {
            "Losses/Policy Loss": policy_loss.item(),
            "Losses/Value Loss": value_loss.item(),
            "Losses/Q1 Loss": q1_loss.item(),
            "Losses/Q2 Loss": q2_loss.item(),
            "Policy/Discrete Entropy Coeff": torch.mean(
                torch.exp(self._log_ent_coef.discrete)
            ).item(),
            "Policy/Continuous Entropy Coeff": torch.mean(
                torch.exp(self._log_ent_coef.continuous)
            ).item(),
            "Policy/Learning Rate": decay_lr,
        }

        return update_stats

    def get_modules(self):
        modules = super().get_modules()
        modules["Optimizer:target_q_network"] = self.target_q_network
        return modules
