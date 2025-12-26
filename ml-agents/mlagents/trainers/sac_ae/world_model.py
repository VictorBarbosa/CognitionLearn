from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import numpy as np
from mlagents.torch_utils import torch, default_device
from mlagents.trainers.torch_entities.layers import linear_layer, Initialization, Swish
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.buffer import AgentBuffer, BufferKey


class WorldModelModule(nn.Module):
    def __init__(
        self,
        latent_size: int,
        action_spec: ActionSpec,
        hidden_units: int = 256,
        num_layers: int = 2,
        learning_rate: float = 3e-4,
    ):
        """
        Módulo de World Model para SAC-AE.
        :param latent_size: Tamanho do espaço latente
        :param action_spec: Especificações das ações
        :param hidden_units: Número de unidades ocultas
        :param num_layers: Número de camadas
        :param learning_rate: Taxa de aprendizado
        """
        super().__init__()
        
        self.latent_size = latent_size
        self.action_spec = action_spec
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        # Tamanho da ação
        self.action_size = int(action_spec.continuous_size)
        if action_spec.discrete_size > 0:
            self.action_size += sum(action_spec.discrete_branches)
            
        # Rede de transição (estado latente + ação -> próximo estado latente)
        self.transition_network = self._create_transition_network()
        
        # Rede de recompensa (estado latente + ação -> recompensa)
        self.reward_network = self._create_reward_network()
        
        # Otimizador para o módulo de world model
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        self.to(default_device())
        
    def _create_transition_network(self) -> nn.Module:
        """Cria a rede de transição que prediz o próximo estado latente."""
        input_size = self.latent_size + self.action_size
        layers = []
        last_size = input_size
        
        for _ in range(self.num_layers):
            layers.append(linear_layer(last_size, self.hidden_units))
            layers.append(Swish())
            last_size = self.hidden_units
            
        # Camada de saída para predizer o próximo estado latente
        layers.append(linear_layer(last_size, self.latent_size))
        
        return nn.Sequential(*layers)
        
    def _create_reward_network(self) -> nn.Module:
        """Cria a rede de recompensa que prediz a recompensa."""
        input_size = self.latent_size + self.action_size
        layers = []
        last_size = input_size
        
        for _ in range(self.num_layers):
            layers.append(linear_layer(last_size, self.hidden_units))
            layers.append(Swish())
            last_size = self.hidden_units
            
        # Camada de saída para predizer a recompensa
        layers.append(linear_layer(last_size, 1))
        
        return nn.Sequential(*layers)
        
    def forward(self, latent: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass do world model.
        :param latent: Estado latente atual
        :param actions: Ações
        :return: (próximo estado latente predito, recompensa predita)
        """
        # Concatenar estado latente e ação
        transition_input = torch.cat([latent, actions], dim=1)
        
        # Predizer próximo estado latente
        predicted_next_latent = self.transition_network(transition_input)
        
        # Predizer recompensa
        predicted_reward = self.reward_network(transition_input)
        
        return predicted_next_latent, predicted_reward
        
    def compute_world_model_loss(
        self, 
        latent: torch.Tensor, 
        actions: torch.Tensor, 
        next_latent: torch.Tensor, 
        rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula a perda do world model.
        :param latent: Estado latente atual
        :param actions: Ações
        :param next_latent: Próximo estado latente real
        :param rewards: Recompensas reais
        :return: (perda de transição, perda de recompensa)
        """
        predicted_next_latent, predicted_reward = self.forward(latent, actions)
        
        # Calcular perda de transição
        transition_loss = torch.mean((predicted_next_latent - next_latent) ** 2)
        
        # Calcular perda de recompensa
        reward_loss = torch.mean((predicted_reward.squeeze() - rewards) ** 2)
        
        return transition_loss, reward_loss
        
    def update(
        self, 
        latent: torch.Tensor, 
        actions: torch.Tensor, 
        next_latent: torch.Tensor, 
        rewards: torch.Tensor
    ) -> Dict[str, float]:
        """
        Atualiza o módulo de world model.
        :param latent: Estado latente atual
        :param actions: Ações
        :param next_latent: Próximo estado latente real
        :param rewards: Recompensas reais
        :return: Estatísticas de atualização
        """
        # Calcular perdas
        transition_loss, reward_loss = self.compute_world_model_loss(latent, actions, next_latent, rewards)
        
        total_loss = transition_loss + reward_loss
        
        # Atualizar pesos
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "WorldModel/Transition Loss": transition_loss.item(),
            "WorldModel/Reward Loss": reward_loss.item(),
            "WorldModel/Total Loss": total_loss.item(),
        }