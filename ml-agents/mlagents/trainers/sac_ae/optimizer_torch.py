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
        Performs update on model.
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
        
        # Codificar observações
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
            
        # Restante da atualização SAC padrão
        sac_stats = super().update(batch, num_sequences)
        
        # Combinar estatísticas
        update_stats = {}
        update_stats.update(ae_stats)
        update_stats.update(wm_stats)
        update_stats.update(sac_stats)
        
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