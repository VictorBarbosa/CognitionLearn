
import numpy as np
from typing import Dict, List, NamedTuple, cast, Tuple, Optional
import attr

from mlagents.torch_utils import torch, nn, default_device

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.networks import ValueNetwork, SharedActorCritic, MultiAgentNetworkBody
from mlagents.trainers.torch_entities.decoders import ValueHeads
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents_envs.timers import timed
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.settings import TrainerSettings, OffPolicyHyperparamSettings
from contextlib import ExitStack
from mlagents.trainers.trajectory import ObsUtil, GroupObsUtil
from mlagents.trainers.sac.optimizer_torch import TorchSACOptimizer, SACSettings

EPSILON = 1e-6  # Small value to avoid divide by zero

logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class MASACSettings(SACSettings):
    pass


class TorchMASACOptimizer(TorchSACOptimizer):

    class CentralizedValueNetwork(ValueNetwork):
        def __init__(
            self,
            stream_names: List[str],
            observation_specs: List[ObservationSpec],
            network_settings: NetworkSettings,
            action_spec: ActionSpec,
            outputs_per_stream: int = 1,
        ):
            # This is not a typo, we want to call __init__ of nn.Module
            nn.Module.__init__(self)
            self.network_body = MultiAgentNetworkBody(
                observation_specs, network_settings, action_spec
            )
            if network_settings.memory is not None:
                encoding_size = network_settings.memory.memory_size // 2
            else:
                encoding_size = network_settings.hidden_units
            self.value_heads = ValueHeads(stream_names, encoding_size + 1, outputs_per_stream)

        def critic_pass(
            self,
            obs: List[List[torch.Tensor]],
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
        ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
            """
            A centralized value function. It calls the forward pass of MultiAgentNetworkBody
            with just the states of all agents.
            :param obs: List of observations for all agents in group
            :param memories: If using memory, a Tensor of initial memories.
            :param sequence_length: If using memory, the sequence length.
            :return: A Tuple of Dict of reward stream to tensor and critic memories.
            """
            encoding, memories = self.network_body(
                obs_only=obs,
                obs=[],
                actions=[],
                memories=memories,
                sequence_length=sequence_length,
            )

            value_outputs, critic_mem_out = self.forward(
                encoding, memories, sequence_length
            )
            return value_outputs, critic_mem_out

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        reward_signal_names = [key for key, _ in self.reward_signals.items()]
        self._critic = self.CentralizedValueNetwork(
            reward_signal_names,
            self.policy.behavior_spec.observation_specs,
            self.policy.network_settings,
            self.policy.behavior_spec.action_spec,
        )
        self.target_network = self.CentralizedValueNetwork(
            reward_signal_names,
            self.policy.behavior_spec.observation_specs,
            self.policy.network_settings,
            self.policy.behavior_spec.action_spec,
        )
        ModelUtils.soft_update(self._critic, self.target_network, 1.0)
        self._move_to_device(default_device())

    def get_trajectory_value_estimates(
        self,
        batch: AgentBuffer,
        next_obs: List[np.ndarray],
        next_group_obs: List[List[np.ndarray]],
        done: bool,
        agent_id: str = "",
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Optional[AgentBufferField]]:
        """
        Get value estimates and memories for a trajectory, in batch form.
        :param batch: An AgentBuffer that consists of a trajectory.
        :param next_obs: the next observation (after the trajectory). Used for bootstrapping
            if this is not a terminal trajectory.
        :param next_group_obs: The next group observations.
        :param done: Set true if this is a terminal trajectory.
        :param agent_id: Agent ID of the agent that this trajectory belongs to.
        :returns: A Tuple of the Value Estimates as a Dict of [name, np.ndarray(trajectory_len)],
            the final value estimate as a Dict of [name, float], and optionally (if using memories)
            an AgentBufferField of initial critic memories to be used during update.
        """
        n_obs = len(self.policy.behavior_spec.observation_specs)

        current_obs = ObsUtil.from_buffer(batch, n_obs)
        group_obs = GroupObsUtil.from_buffer(batch, n_obs)
        all_obs = [current_obs] + group_obs

        # Convert to tensors
        all_obs_tensors = [[ModelUtils.list_to_tensor(o) for o in obs_list] for obs_list in all_obs]
        next_obs_tensors = [ModelUtils.list_to_tensor(o) for o in next_obs]
        next_group_obs_tensors = [[ModelUtils.list_to_tensor(o) for o in obs_list] for obs_list in next_group_obs]
        all_next_obs_tensors = [next_obs_tensors] + next_group_obs_tensors

        if agent_id in self.critic_memory_dict:
            memory = self.critic_memory_dict[agent_id]
        else:
            memory = (
                torch.zeros((1, 1, self.critic.memory_size))
                if self.policy.use_recurrent
                else None
            )

        all_next_memories: Optional[AgentBufferField] = None

        with torch.no_grad():
            if self.policy.use_recurrent:
                raise NotImplementedError("Recurrent MASAC not implemented.")
            else:
                value_estimates, next_memory = self.critic.critic_pass(
                    all_obs_tensors, memory, sequence_length=batch.num_experiences
                )

        self.critic_memory_dict[agent_id] = next_memory

        next_value_estimate, _ = self.critic.critic_pass(
            all_next_obs_tensors, next_memory, sequence_length=1
        )

        for name, estimate in value_estimates.items():
            value_estimates[name] = ModelUtils.to_numpy(estimate)
            next_value_estimate[name] = ModelUtils.to_numpy(next_value_estimate[name])

        if done:
            for k in next_value_estimate:
                if not self.reward_signals[k].ignore_done:
                    next_value_estimate[k] = 0.0
            if agent_id in self.critic_memory_dict:
                self.critic_memory_dict.pop(agent_id)
        return value_estimates, next_value_estimate, all_next_memories

    import numpy as np
from typing import Dict, List, NamedTuple, cast, Tuple, Optional
import attr

from mlagents.torch_utils import torch, nn, default_device

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.networks import ValueNetwork, SharedActorCritic, MultiAgentNetworkBody
from mlagents.trainers.torch_entities.decoders import ValueHeads
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil, AgentBufferField
from mlagents_envs.timers import timed
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.settings import TrainerSettings, OffPolicyHyperparamSettings
from contextlib import ExitStack
from mlagents.trainers.trajectory import ObsUtil, GroupObsUtil
from mlagents.trainers.sac.optimizer_torch import TorchSACOptimizer, SACSettings

EPSILON = 1e-6  # Small value to avoid divide by zero

logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class MASACSettings(SACSettings):
    pass


class TorchMASACOptimizer(TorchSACOptimizer):

    class CentralizedValueNetwork(ValueNetwork):
        def __init__(
            self,
            stream_names: List[str],
            observation_specs: List[ObservationSpec],
            network_settings: NetworkSettings,
            action_spec: ActionSpec,
            outputs_per_stream: int = 1,
        ):
            # This is not a typo, we want to call __init__ of nn.Module
            nn.Module.__init__(self)
            self.network_body = MultiAgentNetworkBody(
                observation_specs, network_settings, action_spec
            )
            if network_settings.memory is not None:
                encoding_size = network_settings.memory.memory_size // 2
            else:
                encoding_size = network_settings.hidden_units
            self.value_heads = ValueHeads(stream_names, encoding_size + 1, outputs_per_stream)

        def critic_pass(
            self,
            obs: List[List[torch.Tensor]],
            memories: Optional[torch.Tensor] = None,
            sequence_length: int = 1,
        ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
            """
            A centralized value function. It calls the forward pass of MultiAgentNetworkBody
            with just the states of all agents.
            :param obs: List of observations for all agents in group
            :param memories: If using memory, a Tensor of initial memories.
            :param sequence_length: If using memory, the sequence length.
            :return: A Tuple of Dict of reward stream to tensor and critic memories.
            """
            encoding, memories = self.network_body(
                obs_only=obs,
                obs=[],
                actions=[],
                memories=memories,
                sequence_length=sequence_length,
            )

            value_outputs, critic_mem_out = self.forward(
                encoding, memories, sequence_length
            )
            return value_outputs, critic_mem_out

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        reward_signal_names = [key for key, _ in self.reward_signals.items()]
        self._critic = self.CentralizedValueNetwork(
            reward_signal_names,
            self.policy.behavior_spec.observation_specs,
            self.policy.network_settings,
            self.policy.behavior_spec.action_spec,
        )
        self.target_network = self.CentralizedValueNetwork(
            reward_signal_names,
            self.policy.behavior_spec.observation_specs,
            self.policy.network_settings,
            self.policy.behavior_spec.action_spec,
        )
        ModelUtils.soft_update(self._critic, self.target_network, 1.0)
        self._move_to_device(default_device())

    def _move_to_device(self, device: torch.device) -> None:
        super()._move_to_device(device)

    def get_trajectory_value_estimates(
        self,
        batch: AgentBuffer,
        next_obs: List[np.ndarray],
        next_group_obs: List[List[np.ndarray]],
        done: bool,
        agent_id: str = "",
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Optional[AgentBufferField]]:
        """
        Get value estimates and memories for a trajectory, in batch form.
        :param batch: An AgentBuffer that consists of a trajectory.
        :param next_obs: the next observation (after the trajectory). Used for bootstrapping
            if this is not a terminal trajectory.
        :param next_group_obs: The next group observations.
        :param done: Set true if this is a terminal trajectory.
        :param agent_id: Agent ID of the agent that this trajectory belongs to.
        :returns: A Tuple of the Value Estimates as a Dict of [name, np.ndarray(trajectory_len)],
            the final value estimate as a Dict of [name, float], and optionally (if using memories)
            an AgentBufferField of initial critic memories to be used during update.
        """
        n_obs = len(self.policy.behavior_spec.observation_specs)

        current_obs = ObsUtil.from_buffer(batch, n_obs)
        group_obs = GroupObsUtil.from_buffer(batch, n_obs)
        all_obs = [current_obs] + group_obs

        # Convert to tensors
        all_obs_tensors = [[ModelUtils.list_to_tensor(o) for o in obs_list] for obs_list in all_obs]
        next_obs_tensors = [ModelUtils.list_to_tensor(o) for o in next_obs]
        next_group_obs_tensors = [[ModelUtils.list_to_tensor(o) for o in obs_list] for obs_list in next_group_obs]
        all_next_obs_tensors = [next_obs_tensors] + next_group_obs_tensors

        if agent_id in self.critic_memory_dict:
            memory = self.critic_memory_dict[agent_id]
        else:
            memory = (
                torch.zeros((1, 1, self.critic.memory_size))
                if self.policy.use_recurrent
                else None
            )

        all_next_memories: Optional[AgentBufferField] = None

        with torch.no_grad():
            if self.policy.use_recurrent:
                raise NotImplementedError("Recurrent MASAC not implemented.")
            else:
                value_estimates, next_memory = self.critic.critic_pass(
                    all_obs_tensors, memory, sequence_length=batch.num_experiences
                )

        self.critic_memory_dict[agent_id] = next_memory

        next_value_estimate, _ = self.critic.critic_pass(
            all_next_obs_tensors, next_memory, sequence_length=1
        )

        for name, estimate in value_estimates.items():
            value_estimates[name] = ModelUtils.to_numpy(estimate)
            next_value_estimate[name] = ModelUtils.to_numpy(next_value_estimate[name])

        if done:
            for k in next_value_estimate:
                if not self.reward_signals[k].ignore_done:
                    next_value_estimate[k] = 0.0
            if agent_id in self.critic_memory_dict:
                self.critic_memory_dict.pop(agent_id)
        return value_estimates, next_value_estimate, all_next_memories

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

        group_obs = GroupObsUtil.from_buffer(batch, n_obs)
        group_obs = [
            [ModelUtils.list_to_tensor(obs) for obs in _group_obs]
            for _group_obs in group_obs
        ]
        all_obs = [current_obs] + group_obs

        next_obs = ObsUtil.from_buffer_next(batch, n_obs)
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]
        next_group_obs = GroupObsUtil.from_buffer_next(batch, n_obs)
        next_group_obs = [
            [ModelUtils.list_to_tensor(obs) for obs in _group_obs]
            for _group_obs in next_group_obs
        ]
        all_next_obs = [next_obs] + next_group_obs


        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        memories_list = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
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

        q_memories = (
            torch.zeros_like(value_memories) if value_memories is not None else None
        )

        self.q_network.q1_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.q_network.q2_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.target_network.network_body.copy_normalization(
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
            all_obs, value_memories, sequence_length=self.policy.sequence_length
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
            if value_memories is not None:
                just_first_obs = [
                    _obs[:: self.policy.sequence_length] for _obs in all_obs
                ]
                _, next_value_memories = self._critic.critic_pass(
                    just_first_obs, value_memories, sequence_length=1
                )
            else:
                next_value_memories = None
            target_values, _ = self.target_network.critic_pass(
                all_next_obs,
                memories=next_value_memories,
                sequence_length=self.policy.sequence_length,
            )

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

        ModelUtils.soft_update(self._critic, self.target_network, self.tau)
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
