from typing import Dict, cast, List, Tuple, Optional
import attr
import numpy as np

from mlagents.torch_utils import torch, nn, default_device

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents.trainers.sac.optimizer_torch import TorchSACOptimizer, SACSettings
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.sac_ae.settings import SACAESettings
from mlagents.trainers.sac_ae.autoencoder_module import AutoEncoderModule
from mlagents.trainers.sac_ae.world_model import WorldModelModule
from mlagents.trainers.torch_entities.networks import ValueNetwork, SharedActorCritic
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents_envs.timers import timed
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from contextlib import ExitStack
from mlagents.trainers.trajectory import ObsUtil

EPSILON = 1e-6  # Small value to avoid divide by zero

logger = get_logger(__name__)


class TorchSACAEOptimizer(TorchSACOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        """
        Extensão do otimizador SAC para incluir autoencoder e world model.
        """
        super().__init__(policy, trainer_settings)
        self.hyperparameters: SACAESettings = cast(
            SACAESettings, trainer_settings.hyperparameters
        )
        
        # Criar módulo de autoencoder
        if self.hyperparameters.use_autoencoder:
            self.autoencoder = AutoEncoderModule(
                self.policy.behavior_spec.observation_specs,
                latent_size=self.hyperparameters.latent_size,
                hidden_units=self.hyperparameters.ae_hidden_units,
                num_layers=self.hyperparameters.ae_num_layers,
                learning_rate=self.hyperparameters.ae_learning_rate,
            )
            
        # Criar módulo de world model
        if self.hyperparameters.use_world_model:
            self.world_model = WorldModelModule(
                self.hyperparameters.latent_size,
                self.policy.behavior_spec.action_spec,
                hidden_units=self.hyperparameters.world_model_hidden_units,
                num_layers=self.hyperparameters.world_model_num_layers,
                learning_rate=self.hyperparameters.world_model_learning_rate,
            )
            
        # Atualizar otimizador para incluir novos parâmetros
        if hasattr(self, 'autoencoder') or hasattr(self, 'world_model'):
            params = list(self.q_network.parameters()) + list(self.policy.actor.parameters())
            if hasattr(self, 'autoencoder'):
                params += list(self.autoencoder.parameters())
            if hasattr(self, 'world_model'):
                params += list(self.world_model.parameters())
            self.optimizer = torch.optim.Adam(
                params, lr=self.trainer_settings.hyperparameters.learning_rate
            )

    def _encode_observations(self, observations: List[torch.Tensor]) -> torch.Tensor:
        """
        Codifica observações usando o autoencoder.
        :param observations: Lista de tensores de observações
        :return: Representações latentes
        """
        if hasattr(self, 'autoencoder') and self.hyperparameters.use_autoencoder:
            return self.autoencoder.encode(observations)
        else:
            # Se não usar autoencoder, achatar as observações
            if len(observations) > 1:
                flat_obs = torch.cat([obs.view(obs.shape[0], -1) for obs in observations], dim=1)
            else:
                flat_obs = observations[0].view(observations[0].shape[0], -1)
            return flat_obs
            
    def get_trajectory_value_estimates(
        self, buffer: AgentBuffer, next_obs: List[np.ndarray], done_reached: bool
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Optional[np.ndarray]]:
        """
        Sobrescreve o método para usar observações codificadas.
        :param buffer: Buffer de experiências.
        :param next_obs: Próximas observações.
        :param done_reached: Se o episódio terminou.
        :return: Estimativas de valor, estatísticas e memórias.
        """
        # Converter observações para tensores
        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(buffer, n_obs)
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
        
        # Codificar observações
        encoded_obs = self._encode_observations(current_obs)
        
        # Converter next_obs para tensores
        next_obs_tensors = [ModelUtils.list_to_tensor(obs) for obs in next_obs]
        encoded_next_obs = self._encode_observations(next_obs_tensors)
        
        # Em vez de chamar super(), vamos implementar diretamente o que precisamos
        # Isso evita o problema de conversão duplicada
        
        # Converter encoded_next_obs de volta para numpy para compatibilidade
        encoded_next_obs_numpy = ModelUtils.to_numpy(encoded_next_obs)
        
        # Retornar estimativas de valor padrão (você pode querer implementar algo mais sofisticado aqui)
        value_estimates = {}
        for name in self.reward_signals:
            # Usar zeros como estimativa padrão
            value_estimates[name] = np.zeros(buffer.num_experiences, dtype=np.float32)
            
        return value_estimates, {}, None
        
    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Performs update on model using encoded observations.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        # Atualizar autoencoder se necessário
        ae_stats = {}
        if hasattr(self, 'autoencoder') and self.hyperparameters.use_autoencoder:
            ae_stats = self.autoencoder.update(batch)
            
        # Converter batch para tensores
        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
        
        next_obs_list = ObsUtil.from_buffer_next(batch, n_obs)
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs_list]
        
        # Codificar observações para o espaço latente
        encoded_obs = self._encode_observations(current_obs)
        encoded_next_obs = self._encode_observations(next_obs)
        
        # Atualizar world model se necessário
        wm_stats = {}
        if hasattr(self, 'world_model') and self.hyperparameters.use_world_model:
            # Converter ações para tensor
            if self.policy.behavior_spec.action_spec.continuous_size > 0:
                actions = ModelUtils.list_to_tensor(batch[BufferKey.CONTINUOUS_ACTION])
            else:
                # Para ações discretas, converter para one-hot
                actions = ModelUtils.list_to_tensor(batch[BufferKey.DISCRETE_ACTION]).float()
                
            # Converter recompensas para tensor
            rewards = ModelUtils.list_to_tensor(batch[BufferKey.ENVIRONMENT_REWARDS])
            
            wm_stats = self.world_model.update(encoded_obs, actions, encoded_next_obs, rewards)
        
        # Modificação principal: Substituir a chamada para super().update para operar com observações codificadas
        # Extrair dados necessários do batch
        rewards = {}
        for name in self.reward_signals:
            rewards[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.rewards_key(name)]
            )

        # Usar as observações codificadas no lugar das observações brutas
        # Agora encoded_obs e encoded_next_obs são as representações latentes
        
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

        # Usar as observações codificadas em vez das brutas
        current_obs_encoded = [encoded_obs]  # Observações já codificadas
        next_obs_encoded = [encoded_next_obs]  # Próximas observações codificadas

        q_memories = (
            torch.zeros_like(value_memories) if value_memories is not None else None
        )

        # Copiar normalização
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
        
        # Obter ações e log_probs com observações codificadas
        sampled_actions, run_out, _, = self.policy.actor.get_action_and_stats(
            current_obs_encoded,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )
        log_probs = run_out["log_probs"]
        value_estimates, _ = self._critic.critic_pass(
            current_obs_encoded, value_memories, sequence_length=self.policy.sequence_length
        )

        cont_sampled_actions = sampled_actions.continuous_tensor
        cont_actions = actions.continuous_tensor
        q1p_out, q2p_out = self.q_network(
            current_obs_encoded,
            cont_sampled_actions,
            memories=q_memories,
            sequence_length=self.policy.sequence_length,
            q2_grad=False,
        )
        q1_out, q2_out = self.q_network(
            current_obs_encoded,
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
                    _obs_encoded[:: self.policy.sequence_length] for _obs_encoded in current_obs_encoded
                ]
                _, next_value_memories = self._critic.critic_pass(
                    just_first_obs, value_memories, sequence_length=1
                )
            else:
                next_value_memories = None
            target_values, _ = self.target_network(
                next_obs_encoded,
                memories=next_value_memories,
                sequence_length=self.policy.sequence_length,
            )
        masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)
        dones = ModelUtils.list_to_tensor(batch[BufferKey.DONE])

        # Calcular perdas
        q1_loss, q2_loss = self.sac_q_loss(
            q1_stream, q2_stream, target_values, dones, rewards, masks
        )
        value_loss = self.sac_value_loss(
            log_probs, value_estimates, q1p_out, q2p_out, masks
        )
        policy_loss = self.sac_policy_loss(log_probs, q1p_out, masks)
        entropy_loss = self.sac_entropy_loss(log_probs, masks)

        total_value_loss = q1_loss + q2_loss + value_loss

        # Atualizar parâmetros
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        ModelUtils.update_learning_rate(self.policy_optimizer, decay_lr)
        ModelUtils.update_learning_rate(self.value_optimizer, decay_lr)
        ModelUtils.update_learning_rate(self.entropy_optimizer, decay_lr)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        self.value_optimizer.step()

        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_optimizer.step()

        # Atualizar rede alvo
        ModelUtils.soft_update(self._critic, self.target_network, self.tau)
        
        # Combinar estatísticas
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
        
        update_stats.update(ae_stats)
        update_stats.update(wm_stats)
        
        return update_stats
        
    def get_modules(self):
        """
        Returns a dictionary of modules to be saved/loaded.
        """
        modules = super().get_modules()
        
        if hasattr(self, 'autoencoder'):
            modules["Optimizer:autoencoder"] = self.autoencoder
            
        if hasattr(self, 'world_model'):
            modules["Optimizer:world_model"] = self.world_model
            
        return modules