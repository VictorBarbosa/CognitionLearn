from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from mlagents.torch_utils import torch, default_device
from mlagents.trainers.torch_entities.layers import linear_layer, Initialization, Swish
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents_envs.base_env import ObservationSpec
from mlagents.trainers.buffer import AgentBuffer, BufferKey


class AutoEncoderModule(nn.Module):
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        latent_size: int = 512,
        hidden_units: int = 256,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
    ):
        """
        Módulo de AutoEncoder para SAC-AE.
        :param observation_specs: Especificações das observações
        :param latent_size: Tamanho do espaço latente
        :param hidden_units: Número de unidades ocultas
        :param num_layers: Número de camadas
        :param learning_rate: Taxa de aprendizado
        """
        super().__init__()
        
        self.observation_specs = observation_specs
        self.latent_size = latent_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        # Identificar observações visuais
        self.visual_obs_specs = [spec for spec in observation_specs if len(spec.shape) >= 3]
        self.vector_obs_specs = [spec for spec in observation_specs if len(spec.shape) < 3]
        
        # Calcular o tamanho total das observações visuais
        self.visual_obs_size = 0
        for spec in self.visual_obs_specs:
            self.visual_obs_size += int(np.prod(spec.shape))
            
        # Calcular o tamanho total das observações vetoriais
        self.vector_obs_size = 0
        for spec in self.vector_obs_specs:
            self.vector_obs_size += int(np.prod(spec.shape))
            
        # Encoder para observações visuais
        if self.visual_obs_size > 0:
            self.visual_encoder = self._create_encoder(self.visual_obs_size, latent_size // 2)
            
        # Encoder para observações vetoriais
        if self.vector_obs_size > 0:
            # print(f"Creating vector encoder with input_size={self.vector_obs_size}, output_size={latent_size // 2}")
            self.vector_encoder = self._create_encoder(self.vector_obs_size, latent_size // 2)
            
        # Camada de combinação para obter representação latente completa
        if self.visual_obs_size > 0 and self.vector_obs_size > 0:
            self.combination_layer = linear_layer(latent_size, latent_size)
        elif self.visual_obs_size > 0:
            self.combination_layer = linear_layer(latent_size // 2, latent_size)
        elif self.vector_obs_size > 0:
            self.combination_layer = linear_layer(latent_size // 2, latent_size)
            
        # Decoder para reconstruir observações
        self.visual_decoder = None
        if self.visual_obs_size > 0:
            self.visual_decoder = self._create_decoder(latent_size, self.visual_obs_size)
            
        self.vector_decoder = None
        if self.vector_obs_size > 0:
            self.vector_decoder = self._create_decoder(latent_size, self.vector_obs_size)
        
        # Otimizador para o módulo de autoencoder
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        self.to(default_device())
            
        # Encoder para observações visuais
        if self.visual_obs_size > 0:
            self.visual_encoder = self._create_encoder(self.visual_obs_size, latent_size // 2)
            
        # Encoder para observações vetoriais
        if self.vector_obs_size > 0:
            self.vector_encoder = self._create_encoder(self.vector_obs_size, latent_size // 2)
            
        # Camada de combinação para obter representação latente completa
        if self.visual_obs_size > 0 and self.vector_obs_size > 0:
            self.combination_layer = linear_layer(latent_size, latent_size)
        elif self.visual_obs_size > 0:
            self.combination_layer = linear_layer(latent_size // 2, latent_size)
        elif self.vector_obs_size > 0:
            self.combination_layer = linear_layer(latent_size // 2, latent_size)
            
        # Decoder para reconstruir observações
        self.visual_decoder = None
        if self.visual_obs_size > 0:
            self.visual_decoder = self._create_decoder(latent_size, self.visual_obs_size)
            
        self.vector_decoder = None
        if self.vector_obs_size > 0:
            self.vector_decoder = self._create_decoder(latent_size, self.vector_obs_size)
        
        # Otimizador para o módulo de autoencoder
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        self.to(default_device())
        
    def _create_encoder(self, input_size: int, output_size: int) -> nn.Module:
        """Cria um encoder."""
        layers = []
        last_size = input_size
        
        # Camadas ocultas
        for _ in range(self.num_layers):
            layers.append(linear_layer(last_size, self.hidden_units))
            layers.append(Swish())
            last_size = self.hidden_units
            
        # Camada de saída
        layers.append(linear_layer(last_size, output_size))
        
        return nn.Sequential(*layers)
        
    def _create_decoder(self, input_size: int, output_size: int) -> nn.Module:
        """Cria um decoder."""
        layers = []
        last_size = input_size
        
        # Camadas ocultas
        for _ in range(self.num_layers):
            layers.append(linear_layer(last_size, self.hidden_units))
            layers.append(Swish())
            last_size = self.hidden_units
            
        # Camada de saída
        layers.append(linear_layer(last_size, output_size))
        
        return nn.Sequential(*layers)
        
    def encode(self, observations: List[torch.Tensor]) -> torch.Tensor:
        """
        Codifica observações em representações latentes.
        :param observations: Lista de tensores de observações
        :return: Representação latente
        """
        visual_features = []
        vector_features = []
        
        obs_idx = 0
        # Processar observações visuais
        for spec in self.visual_obs_specs:
            obs_tensor = observations[obs_idx]
            # print(f"Processing visual obs {obs_idx}: shape={obs_tensor.shape}")
            # Achatar a observação
            flat_obs = obs_tensor.view(obs_tensor.shape[0], -1)
            encoded = self.visual_encoder(flat_obs)
            visual_features.append(encoded)
            obs_idx += 1
            
        # Processar observações vetoriais
        for spec in self.vector_obs_specs:
            obs_tensor = observations[obs_idx]
            # Achatar a observação
            flat_obs = obs_tensor.view(obs_tensor.shape[0], -1)
            # print(f"Flattened obs shape: {flat_obs.shape}")
            
            # Verificar dimensões
            batch_size = flat_obs.shape[0]
            feature_size = flat_obs.shape[1]
            expected_feature_size = int(np.prod(spec.shape))
            
            # print(f"Feature size check: expected={expected_feature_size}, actual={feature_size}")
            
            if feature_size != expected_feature_size:
                # print(f"Mismatch! Expected {expected_feature_size} features but got {feature_size}")
                # print(f"Obs tensor shape: {obs_tensor.shape}")
                # print(f"Spec shape: {spec.shape}")
                
                # Lidar com o mismatch
                if feature_size == 1 and expected_feature_size > 1:
                    # Repetir o elemento para ter o tamanho esperado
                    # print(f"Expanding from {feature_size} to {expected_feature_size}")
                    flat_obs = flat_obs.expand(batch_size, expected_feature_size)
                elif feature_size > expected_feature_size:
                    # Cortar para o tamanho esperado
                    # print(f"Trimming from {feature_size} to {expected_feature_size}")
                    flat_obs = flat_obs[:, :expected_feature_size]
                else:
                    # Caso contrário, vamos tentar outro approach
                    # print("Unhandled case, using zero padding")
                    padded_obs = torch.zeros(batch_size, expected_feature_size, device=flat_obs.device)
                    padded_obs[:, :feature_size] = flat_obs[:, :min(feature_size, expected_feature_size)]
                    flat_obs = padded_obs
                    
            encoded = self.vector_encoder(flat_obs)
            vector_features.append(encoded)
            obs_idx += 1
            
        # Combinar features
        all_features = visual_features + vector_features
        if len(all_features) > 1:
            combined = torch.cat(all_features, dim=1)
        else:
            combined = all_features[0] if all_features else torch.zeros(observations[0].shape[0], self.latent_size, device=default_device())
            
        # Aplicar camada de combinação
        if hasattr(self, 'combination_layer'):
            latent = self.combination_layer(combined)
        else:
            latent = combined
            
        return latent
        
    def decode(self, latent: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Decodifica representações latentes em observações.
        :param latent: Representação latente
        :return: (observações visuais reconstruídas, observações vetoriais reconstruídas)
        """
        visual_reconstruction = None
        vector_reconstruction = None
        
        if self.visual_decoder is not None:
            visual_reconstruction = self.visual_decoder(latent)
            
        if self.vector_decoder is not None:
            vector_reconstruction = self.vector_decoder(latent)
            
        return visual_reconstruction, vector_reconstruction
        
    def forward(self, observations: List[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass do autoencoder.
        :param observations: Lista de tensores de observações
        :return: (representação latente, observações visuais reconstruídas, observações vetoriais reconstruídas)
        """
        latent = self.encode(observations)
        visual_reconstruction, vector_reconstruction = self.decode(latent)
        
        return latent, visual_reconstruction, vector_reconstruction
        
    def compute_reconstruction_loss(
        self, 
        observations: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Calcula a perda de reconstrução.
        :param observations: Observações originais
        :return: Perda de reconstrução
        """
        latent, visual_reconstruction, vector_reconstruction = self.forward(observations)
        
        total_loss = 0.0
        
        obs_idx = 0
        # Calcular perda para observações visuais
        for spec in self.visual_obs_specs:
            obs_tensor = observations[obs_idx]
            flat_obs = obs_tensor.view(obs_tensor.shape[0], -1)
            if visual_reconstruction is not None:
                loss = torch.mean((visual_reconstruction - flat_obs) ** 2)
                total_loss += loss
            obs_idx += 1
            
        # Calcular perda para observações vetoriais
        for spec in self.vector_obs_specs:
            obs_tensor = observations[obs_idx]
            flat_obs = obs_tensor.view(obs_tensor.shape[0], -1)
            if vector_reconstruction is not None:
                loss = torch.mean((vector_reconstruction - flat_obs) ** 2)
                total_loss += loss
            obs_idx += 1
            
        return total_loss
        
    def update(self, batch: AgentBuffer) -> Dict[str, float]:
        """
        Atualiza o módulo de autoencoder com um batch de experiências.
        :param batch: Batch de experiências
        :return: Estatísticas de atualização
        """
        # Converter batch para tensores
        from mlagents.trainers.trajectory import ObsUtil
        n_obs = len(self.observation_specs)
        current_obs_list = ObsUtil.from_buffer(batch, n_obs)
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs_list]
        
        # Calcular perda de reconstrução
        reconstruction_loss = self.compute_reconstruction_loss(current_obs)
        
        # Atualizar pesos
        self.optimizer.zero_grad()
        reconstruction_loss.backward()
        self.optimizer.step()
        
        return {
            "AutoEncoder/Reconstruction Loss": reconstruction_loss.item(),
        }