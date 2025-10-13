"""
Supervised trainer for ML-Agents standalone learning.
"""

import os
import numpy as np
from typing import Dict, Any, List
import torch
from torch import nn
from torch.utils.data import DataLoader

from mlagents_envs.base_env import ActionSpec
from mlagents.trainers.settings import NetworkSettings

from standalone_supervised.data_loader import SupervisedDataLoader
from standalone_supervised.models import create_model_for_algorithm


class SupervisedTrainer:
    """
    Treinador para aprendizado supervisionado standalone compatível com ML-Agents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        :param config: Dicionário de configuração
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")

        # Configurações de modelo
        self.algorithm = config["model"].get("algorithm", "ppo")
        self.network_settings = self._create_network_settings()
        
        # Configurações de treinamento
        self.epochs = config["training"].get("epochs", 10)
        self.batch_size = config["training"].get("batch_size", 128)
        self.learning_rate = config["training"].get("learning_rate", 3e-4)
        self.validation_split = config["training"].get("validation_split", 0.2)
        self.shuffle = config["training"].get("shuffle", True)
        self.augment_noise = config["training"].get("augment_noise", 0.01)
        self.early_stopping = config["training"].get("early_stopping", True)
        self.patience = config["training"].get("patience", 5)
        self.min_delta = config["training"].get("min_delta", 0.001)
        self.dropout_rate = config["training"].get("dropout_rate", 0.1)
        self.weight_decay = config["training"].get("weight_decay", 1e-4)
        self.lr_patience = config["training"].get("lr_patience", 5)
    
        # Configurações de saída
        self.output_dir = config["output"].get("dir", "./results")
        self.checkpoint_interval = config["output"].get("checkpoint_interval", 1000)
        
        # Criar diretório de saída se não existir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Carregar dados
        self.data_loader = self._load_data()
        
        # Criar modelo
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Criar otimizador e função de perda
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.criterion = nn.MSELoss()
        
        # Criar scheduler de taxa de aprendizado
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.lr_patience
        )
        
        # Estado para early stopping
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        print(f"Treinador configurado para algoritmo: {self.algorithm}")
        print(f"Dimensões de entrada: {self.data_loader.get_num_features()}")
        print(f"Dimensões de saída: {self.data_loader.get_num_actions()}")
    
    def _create_network_settings(self) -> NetworkSettings:
        """
        Cria configurações de rede a partir da configuração.
        """
        network_config = self.config["model"].get("network_settings", {})
        return NetworkSettings(
            hidden_units=network_config.get("hidden_units", 256),
            num_layers=network_config.get("num_layers", 2),
            normalize=network_config.get("normalize", True),
            memory_size=network_config.get("memory_size", 0),
            use_recurrent=network_config.get("use_recurrent", False)
        )
    
    def _load_data(self) -> SupervisedDataLoader:
        """
        Carrega dados de treinamento supervisionado a partir do CSV.
        """
        data_config = self.config["data"]
        csv_path = data_config["csv_path"]
        observation_columns = data_config.get("observation_columns", [])
        action_columns = data_config.get("action_columns", [])
        
        return SupervisedDataLoader(
            csv_path=csv_path,
            observation_columns=observation_columns,
            action_columns=action_columns,
            validation_split=self.validation_split,
            shuffle=self.shuffle,
            augment_noise=self.augment_noise
        )
    
    def _create_model(self):
        """
        Cria o modelo apropriado para o algoritmo selecionado.
        """
        return create_model_for_algorithm(
            algorithm=self.algorithm,
            observation_specs=self.data_loader.observation_specs,
            network_settings=self.network_settings,
            action_spec=self.data_loader.action_spec,
            name_behavior="supervised_actor",
            dropout_rate=self.dropout_rate
        )
    
    def train_epoch(self, data_loader: DataLoader) -> float:
        """
        Executa uma época de treinamento.
        :param data_loader: DataLoader com os dados de treinamento
        :return: Perda média na época
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (observations, actions) in enumerate(data_loader):
            observations = observations.to(self.device)
            actions = actions.to(self.device)
            
            # Forward pass
            predicted_actions = self.model(observations)
            
            # Calcular perda
            loss = self.criterion(predicted_actions, actions)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, data_loader: DataLoader) -> float:
        """
        Valida o modelo no conjunto de validação.
        :param data_loader: DataLoader com os dados de validação
        :return: Perda média na validação
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for observations, actions in data_loader:
                observations = observations.to(self.device)
                actions = actions.to(self.device)
                
                # Forward pass
                predicted_actions = self.model(observations)
                
                # Calcular perda
                loss = self.criterion(predicted_actions, actions)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def should_stop_early(self, val_loss: float, epoch: int) -> bool:
        """
        Verifica se o treinamento deve parar cedo.
        :param val_loss: Perda atual na validação
        :param epoch: Época atual
        :return: True se o treinamento deve parar cedo
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.patience
    
    def train(self):
        """
        Executa o treinamento supervisionado completo.
        """
        print("Iniciando treinamento supervisionado...")
        
        # Carregar dados
        train_loader = self.data_loader.get_train_loader(self.batch_size)
        val_loader = self.data_loader.get_validation_loader(self.batch_size)
        
        for epoch in range(self.epochs):
            # Treinamento
            train_loss = self.train_epoch(train_loader)
            
            # Validação
            val_loss = self.validate(val_loader)
            
            # Atualizar scheduler
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Salvar checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch + 1)
            
            # Early stopping
            if self.early_stopping and self.should_stop_early(val_loss, epoch):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Salvar modelo final
        self._save_final_model()
        print("Treinamento supervisionado concluído.")
    
    def _save_checkpoint(self, epoch: int):
        """
        Salva um checkpoint do modelo.
        :param epoch: Número da época
        """
        checkpoint_path = os.path.join(self.output_dir, f"supervised_checkpoint_{epoch}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint salvo: {checkpoint_path}")
    
    def _save_final_model(self):
        """
        Salva o modelo final.
        """
        # Salvar modelo PyTorch
        pt_path = os.path.join(self.output_dir, "supervised_model.pt")
        torch.save(self.model.state_dict(), pt_path)
        print(f"Modelo PyTorch salvo: {pt_path}")
        
        # Salvar modelo ONNX
        onnx_path = os.path.join(self.output_dir, "supervised_model.onnx")
        self._export_to_onnx(onnx_path)
        print(f"Modelo ONNX salvo: {onnx_path}")
    
    def _export_to_onnx(self, output_path: str):
        """
        Exporta o modelo para formato ONNX.
        :param output_path: Caminho para salvar o modelo ONNX
        """
        self.model.eval()
        
        # Criar entrada dummy
        dummy_input = torch.randn(1, self.data_loader.get_num_features()).to(self.device)
        
        # Exportar para ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['vector_observation', 'action_masks', 'recurrent_in'],
            output_names=['version_number', 'memory_size'],
            dynamic_axes={
                'vector_observation': {0: 'batch_size'},
                'action_masks': {0: 'batch_size'},
                'recurrent_in': {0: 'batch_size'}
            }
        )
