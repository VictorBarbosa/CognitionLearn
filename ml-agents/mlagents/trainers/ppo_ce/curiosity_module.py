from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import numpy as np
from mlagents.torch_utils import torch, default_device
from mlagents.trainers.torch_entities.layers import linear_layer, Initialization, Swish
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents_envs.base_env import ObservationSpec, ActionSpec
from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.trajectory import ObsUtil


class CuriosityModule(nn.Module):
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        action_spec: ActionSpec,
        hidden_units: int = 128,
        num_layers: int = 2,
        learning_rate: float = 3e-4,
    ):
        """
        Módulo de curiosidade para PPO-CE.
        :param observation_specs: Especificações das observações
        :param action_spec: Especificações das ações
        :param hidden_units: Número de unidades ocultas
        :param num_layers: Número de camadas
        :param learning_rate: Taxa de aprendizado
        """
        super().__init__()
        
        self.observation_specs = observation_specs
        self.action_spec = action_spec
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        # Calcular o tamanho total da entrada (observações concatenadas)
        self.observation_size = sum(np.prod(obs.shape) for obs in observation_specs)
        
        # Tamanho da ação
        self.action_size = int(action_spec.continuous_size)
        if action_spec.discrete_size > 0:
            self.action_size += sum(action_spec.discrete_branches)
            
        # Encoder para as observações
        self.obs_encoder = self._create_encoder(self.observation_size)
        
        # Encoder para ações
        self.action_encoder = self._create_encoder(self.action_size)
        
        # Rede de transição (estado + ação -> próximo estado)
        self.transition_network = self._create_transition_network()
        
        # Otimizador para o módulo de curiosidade
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        self.to(default_device())
        
    def _create_encoder(self, input_size: int) -> nn.Module:
        """Cria um encoder para observações ou ações."""
        layers = []
        last_size = input_size
        
        for _ in range(self.num_layers):
            layers.append(linear_layer(last_size, self.hidden_units))
            layers.append(Swish())
            last_size = self.hidden_units
            
        return nn.Sequential(*layers)
        
    def _create_transition_network(self) -> nn.Module:
        """Cria a rede de transição que prediz o próximo estado."""
        input_size = self.hidden_units + self.hidden_units  # estado codificado + ação codificada
        layers = []
        last_size = input_size
        
        for _ in range(self.num_layers):
            layers.append(linear_layer(last_size, self.hidden_units))
            layers.append(Swish())
            last_size = self.hidden_units
            
        # Camada de saída para predizer o próximo estado codificado
        layers.append(linear_layer(last_size, self.hidden_units))
        
        return nn.Sequential(*layers)
        
    def forward(self, observations: List[torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass do módulo de curiosidade.
        :param observations: Lista de tensores de observações
        :param actions: Tensor de ações
        :return: (estado_codificado, proximo_estado_predito)
        """
        # Concatenar observações
        if len(observations) > 1:
            flat_obs = torch.cat([obs.view(obs.shape[0], -1) for obs in observations], dim=1)
        else:
            flat_obs = observations[0].view(observations[0].shape[0], -1)
            
        # Codificar observações e ações
        encoded_obs = self.obs_encoder(flat_obs)
        encoded_actions = self.action_encoder(actions)
        
        # Concatenar estado codificado e ação codificada
        transition_input = torch.cat([encoded_obs, encoded_actions], dim=1)
        
        # Predizer próximo estado
        predicted_next_state = self.transition_network(transition_input)
        
        return encoded_obs, predicted_next_state
        
    def compute_curiosity_reward(
        self, 
        observations: List[torch.Tensor], 
        actions: torch.Tensor, 
        next_observations: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Calcula a recompensa de curiosidade.
        :param observations: Observações atuais
        :param actions: Ações tomadas
        :param next_observations: Próximas observações
        :return: Recompensa de curiosidade
        """
        # Codificar observações atuais
        encoded_obs, predicted_next_state = self.forward(observations, actions)
        
        # Codificar próximas observações
        if len(next_observations) > 1:
            flat_next_obs = torch.cat([obs.view(obs.shape[0], -1) for obs in next_observations], dim=1)
        else:
            flat_next_obs = next_observations[0].view(next_observations[0].shape[0], -1)
            
        actual_next_state = self.obs_encoder(flat_next_obs)
        
        # Calcular a recompensa de curiosidade (erro de predição)
        curiosity_reward = torch.mean((predicted_next_state - actual_next_state) ** 2, dim=1)
        
        return curiosity_reward
        
    def update(self, batch: AgentBuffer) -> Dict[str, float]:
        """
        Atualiza o módulo de curiosidade com um batch de experiências.
        :param batch: Batch de experiências
        :return: Estatísticas de atualização
        """
        # Converter batch para tensores
        n_obs = len(self.observation_specs)
        current_obs_list = ObsUtil.from_buffer(batch, n_obs)
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs_list]
        next_obs_list = ObsUtil.from_buffer_next(batch, n_obs)
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs_list]
        
        # Converter ações para tensor
        if self.action_spec.continuous_size > 0:
            actions = ModelUtils.list_to_tensor(batch[BufferKey.CONTINUOUS_ACTION])
        else:
            # Para ações discretas, precisamos converter para one-hot
            actions = ModelUtils.list_to_tensor(batch[BufferKey.DISCRETE_ACTION])
            # Aqui precisaríamos fazer o one-hot encoding, mas vamos simplificar por agora
            actions = actions.float()
            
        # Forward pass
        encoded_obs, predicted_next_state = self.forward(current_obs, actions)
        
        # Codificar próximas observações
        if len(next_obs) > 1:
            flat_next_obs = torch.cat([obs.view(obs.shape[0], -1) for obs in next_obs], dim=1)
        else:
            flat_next_obs = next_obs[0].view(next_obs[0].shape[0], -1)
            
        actual_next_state = self.obs_encoder(flat_next_obs)
        
        # Calcular perda
        loss = torch.mean((predicted_next_state - actual_next_state) ** 2)
        
        # Atualizar pesos
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "Curiosity/Loss": loss.item(),
            "Curiosity/Mean Prediction Error": torch.mean(torch.abs(predicted_next_state - actual_next_state)).item()
        }