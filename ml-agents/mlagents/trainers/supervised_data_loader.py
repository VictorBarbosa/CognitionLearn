"""
Módulo para carregar e processar dados de treinamento supervisionado a partir de um arquivo CSV.
"""
import csv
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.torch_utils import torch, default_device


class SupervisedDataLoader:
    """
    Classe para carregar e processar dados de treinamento supervisionado a partir de um CSV.
    """
    
    def __init__(
        self,
        csv_path: str,
        observation_columns: List[str],
        action_columns: List[str],
        validation_split: float = 0.2,
        shuffle: bool = True,
        augment_noise: float = 0.01,
        action_spec: Optional[ActionSpec] = None
    ):
        """
        :param csv_path: Caminho para o arquivo CSV
        :param observation_columns: Lista de nomes das colunas que contêm observações
        :param action_columns: Lista de nomes das colunas que contêm ações
        :param validation_split: Fração dos dados a ser usada para validação
        :param shuffle: Se os dados devem ser embaralhados
        :param augment_noise: Nível de ruído para aumentação de dados
        :param action_spec: Especificação das ações (opcional, para validação)
        """
        self.csv_path = csv_path
        self.observation_columns = observation_columns
        self.action_columns = action_columns
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.augment_noise = augment_noise
        self.action_spec = action_spec
        
        # Carregar dados
        self.data = self._load_data()
        
        # Dividir dados em treino e validação
        self.train_data, self.val_data = self._split_data()
        
        # Validar se o número de ações corresponde à especificação, se fornecida
        # Essa validação pode ser opcional para manter compatibilidade com diferentes configurações
        if self.action_spec is not None:
            if self.action_spec.is_discrete:
                expected_action_size = sum(self.action_spec.discrete_branches)
            else:
                expected_action_size = self.action_spec.continuous_size

            if expected_action_size != len(self.action_columns):
                import warnings
                # warnings.warn(
                #     f"Número de colunas de ação ({len(self.action_columns)}) não corresponde "
                #     f"à especificação de ações ({expected_action_size}). "
                #     f"Continuando de qualquer forma, mas verifique se as colunas de ação estão corretas."
                # )
    
    def _load_data(self) -> pd.DataFrame:
        """
        Carrega os dados do arquivo CSV.
        """
        try:
            data = pd.read_csv(self.csv_path)
            # Verificar se as colunas especificadas existem
            required_columns = set(self.observation_columns + self.action_columns)
            available_columns = set(data.columns)
            
            missing_cols = required_columns - available_columns
            if missing_cols:
                raise ValueError(f"Colunas ausentes no CSV: {missing_cols}")
                
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo CSV não encontrado: {self.csv_path}")
        except Exception as e:
            raise ValueError(f"Erro ao carregar CSV: {str(e)}")
    
    def _split_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Divide os dados em conjuntos de treinamento e validação.
        """
        if self.shuffle:
            self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n_samples = len(self.data)
        n_val = int(n_samples * self.validation_split)
        
        val_indices = self.data.index[:n_val].tolist()
        train_indices = self.data.index[n_val:].tolist()
        
        # Separar observações e ações
        train_data = {
            'observations': self.data.loc[train_indices, self.observation_columns].values.astype(np.float32),
            'actions': self.data.loc[train_indices, self.action_columns].values.astype(np.float32)
        }
        
        val_data = {
            'observations': self.data.loc[val_indices, self.observation_columns].values.astype(np.float32),
            'actions': self.data.loc[val_indices, self.action_columns].values.astype(np.float32)
        }
        
        return train_data, val_data
    
    def add_noise_to_observations(self, observations: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Adiciona ruído às observações para aumentação de dados.
        """
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, observations.shape).astype(observations.dtype)
            return observations + noise
        return observations
    
    def get_train_loader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """
        Retorna um DataLoader PyTorch para os dados de treinamento.
        """
        return self._create_loader(self.train_data, batch_size)
    
    def get_validation_loader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """
        Retorna um DataLoader PyTorch para os dados de validação.
        """
        return self._create_loader(self.val_data, batch_size, shuffle=False)
    
    def _create_loader(self, data: Dict[str, np.ndarray], batch_size: int, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """
        Cria um DataLoader PyTorch com os dados especificados.
        """
        # Adicionar ruído de aumento de dados apenas ao treinamento
        observations = data['observations']
        if self.augment_noise > 0 and data is self.train_data:
            observations = self.add_noise_to_observations(observations, self.augment_noise)
        
        observations_tensor = torch.tensor(observations)
        actions_tensor = torch.tensor(data['actions'])
        
        dataset = torch.utils.data.TensorDataset(observations_tensor, actions_tensor)
        # Criar um gerador compatível com o dispositivo definido para o modelo
        generator = torch.Generator(device=default_device())
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if data is self.train_data else False,
            generator=generator
        )
    
    def get_num_features(self) -> int:
        """
        Retorna o número de features nas observações.
        """
        return len(self.observation_columns)
    
    def get_num_actions(self) -> int:
        """
        Retorna o número de ações.
        """
        return len(self.action_columns)