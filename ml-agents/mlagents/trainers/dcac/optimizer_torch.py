# ## ML-Agent Learning (DCAC)
# Contains an implementation of DCAC as described in https://arxiv.org/abs/2302.03771
# This is a modification of the SAC implementation, with a check for destructive critic updates.

from typing import cast, Dict

import attr
from mlagents.torch_utils import torch

from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents.trainers.sac.optimizer_torch import TorchSACOptimizer, SACSettings
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.torch_entities.agent_action import AgentAction


@attr.s(auto_attribs=True)
class DCACSettings(SACSettings):
    """
    DCAC-specific hyperparameters.
    """

    destructive_threshold: float = 0.0


class TorchDCACOptimizer(TorchSACOptimizer):
    """
    This is a modification of the SAC optimizer that checks for destructive critic updates
    before updating the policy, as described in the DCAC paper.
    """

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        self.hyperparameters: DCACSettings = cast(
            DCACSettings, trainer_settings.hyperparameters
        )
        self.destructive_threshold = self.hyperparameters.destructive_threshold

    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Updates model using buffer. Overrides the SAC optimizer to apply the DCAC check.
        """
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
            if value_memories is not None:
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
        masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)
        dones = ModelUtils.list_to_tensor(batch[BufferKey.DONE])

        # Calculate losses
        q1_loss, q2_loss = self.sac_q_loss(
            q1_stream, q2_stream, target_values, dones, rewards, masks
        )
        value_loss = self.sac_value_loss(
            log_probs, value_estimates, q1p_out, q2p_out, masks
        )
        policy_loss = self.sac_policy_loss(log_probs, q1p_out, masks)
        entropy_loss = self.sac_entropy_loss(log_probs, masks)

        total_value_loss = q1_loss + q2_loss + value_loss

        # DCAC Check
        with torch.no_grad():
            q_old = torch.min(list(q1_out.values())[0], list(q2_out.values())[0])
            q_new = torch.min(list(q1p_out.values())[0], list(q2p_out.values())[0])
            advantage = (q_new - q_old).mean()

        # --- OPTIMIZER STEPS ---
        # Initialize destructive_updates_skipped before the conditional assignment
        destructive_updates_skipped = 0.0

        # Zero gradients for all optimizers that might be stepped
        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        self.entropy_optimizer.zero_grad()

        # Perform backward passes
        total_value_loss.backward() # Gradients for Q-net and V-net

        # Conditionally perform policy and entropy loss backward
        # Both policy and entropy coefficient updates should be skipped together
        if advantage.item() < self.destructive_threshold:
            destructive_updates_skipped = 1.0
        else:
            policy_loss.backward()
            entropy_loss.backward()  # Only update entropy coefficient when updating policy

        # Perform optimizer steps
        self.value_optimizer.step()
        if advantage.item() < self.destructive_threshold:
            pass # Policy and entropy coefficient updates skipped
        else:
            self.policy_optimizer.step()
            self.entropy_optimizer.step()

        # Update target network
        ModelUtils.soft_update(self._critic, self.target_network, self.tau)
        
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        ModelUtils.update_learning_rate(self.policy_optimizer, decay_lr)
        ModelUtils.update_learning_rate(self.value_optimizer, decay_lr)
        ModelUtils.update_learning_rate(self.entropy_optimizer, decay_lr)

        return {
            "Losses/Policy Loss": policy_loss.item(),
            "Losses/Value Loss": value_loss.item(),
            "Losses/Q1 Loss": q1_loss.item(),
            "Losses/Q2 Loss": q2_loss.item(),
            "Policy/Destructive Updates Skipped": destructive_updates_skipped,
            "Policy/Continuous Entropy Coeff": torch.mean(torch.exp(self._log_ent_coef.continuous)).item(),
            "Policy/Discrete Entropy Coeff": torch.mean(torch.exp(self._log_ent_coef.discrete)).item(),
            "Policy/Learning Rate": decay_lr,
        }
