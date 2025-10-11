"""
Modelo sequencial para treinamento supervisionado no ML-Agents.
Este modelo usa nn.Sequential para criar uma rede compatível com o sistema de exportação ONNX existente.
"""
import torch
import torch.nn as nn
from typing import List, Tuple
from mlagents_envs.base_env import BehaviorSpec, ActionSpec
from mlagents.trainers.torch_entities.encoders import VectorInput
from mlagents.trainers.torch_entities.decoders import ValueHeads
from mlagents.trainers.torch_entities.action_model import ActionModel
from mlagents.trainers.settings import NetworkSettings


class SequentialActor(nn.Module):
    """
    Modelo de ator sequencial que é compatível com o sistema de exportação ONNX do ML-Agents.
    """
    def __init__(
        self,
        observation_specs,
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        name_behavior: str = "sequential_actor",
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.observation_specs = observation_specs
        self.action_spec = action_spec
        self.network_settings = network_settings
        self.name_behavior = name_behavior
        self.dropout_rate = dropout_rate
        
        # Calcular o tamanho total das observações
        total_obs_size = 0
        for obs_spec in observation_specs:
            total_obs_size += int(torch.prod(torch.tensor(obs_spec.shape)))
        
        # Criar encoder sequencial para as observações
        # Primeiro, um encoder para processar as observações vetoriais
        self.vector_encoder = VectorInput(
            total_obs_size, 
            normalize=network_settings.normalize
        )
        
        # Criar a sequência de camadas para o corpo da rede
        hidden_layers = []
        current_size = total_obs_size
        
        # Adicionar camadas ocultas sequenciais com dropout para regularização
        for _ in range(network_settings.num_layers):
            hidden_layers.append(nn.Linear(current_size, network_settings.hidden_units))
            hidden_layers.append(nn.ReLU())
            # Adicionando dropout após cada camada ReLU para regularização
            if dropout_rate > 0:
                hidden_layers.append(nn.Dropout(dropout_rate))
            current_size = network_settings.hidden_units
        
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        # Criar modelo de ação (que lida com ações contínuas e discretas)
        self.action_model = ActionModel(
            current_size,
            action_spec,
            network_settings
        )
        
        # Memória (para redes recorrentes, se necessário)
        self.memory_size = network_settings.memory_size if network_settings.use_recurrent else 0
        
        # Valor heads (para estimativas de valor, se necessário)
        self.value_heads = ValueHeads(
            [name_behavior], 
            current_size
        )
        
        # Inicializar pesos para melhorar a regularização
        self._init_weights()
    
    def _init_weights(self):
        """
        Inicializa os pesos com regularização para melhorar a generalização.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, obs, masks=None, memories=None):
        """
        Forward pass padrão para o modelo sequencial.
        """
        # Achatando e concatenando observações
        if len(obs) == 1:
            # Caso tenhamos apenas uma observação, usamos diretamente
            flat_obs = obs[0].view(obs[0].size(0), -1)
        else:
            # Concatenar múltiplas observações
            concatenated_obs = [ob.view(ob.size(0), -1) for ob in obs]
            flat_obs = torch.cat(concatenated_obs, dim=1)
        
        # Processar observações
        encoded_obs = self.vector_encoder(flat_obs)
        
        # Passar pelas camadas ocultas sequenciais
        hidden_out = self.hidden_layers(encoded_obs)
        
        # Obter ações
        action, log_probs = self.action_model(hidden_out)
        
        # Obter estimativas de valor
        values = self.value_heads(hidden_out)
        
        return action, values, log_probs
    
    def get_action_and_stats(self, obs, masks=None, memories=None):
        """
        Método compatível com o sistema existente do ML-Agents.
        """
        action, values, log_probs = self.forward(obs, masks, memories)
        
        # Preparar saída compatível com o sistema existente
        run_out = {
            "value_output": values[self.name_behavior],
            "log_probs": log_probs
        }
        
        # Para compatibilidade com o sistema de exportação ONNX
        if memories is None:
            memories = torch.zeros((len(obs[0]), self.memory_size)) if self.memory_size > 0 else torch.tensor([])
        
        return action, run_out, memories
    
    def get_stats(self, obs):
        """
        Método compatível com o sistema existente do ML-Agents.
        """
        action, values, log_probs = self.forward(obs)
        
        return {
            "action": action,
            "value_output": values[self.name_behavior],
            "log_probs": log_probs
        }