"""
Módulo para implementar o otimizador de aprendizado supervisionado.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict

import torch
from torch import nn

from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import SupervisedLearningSettings, NetworkSettings
from mlagents.trainers.supervised_data_loader import SupervisedDataLoader
from mlagents.trainers.trajectory import ObsUtil
from mlagents.torch_utils import torch as torch_module, default_device, set_torch_config
from mlagents.trainers.torch_entities.model_serialization import ModelSerializer
from mlagents.trainers.torch_entities.sequential_model import SequentialActor


class SupervisedTorchOptimizer:
    """
    Otimizador para treinamento supervisionado de redes neurais de políticas.
    """
    
    def __init__(
        self,
        policy: TorchPolicy,
        supervised_settings: SupervisedLearningSettings,
        stats_reporter=None,
        use_sequential_model: bool = False
    ):
        """
        :param policy: Política que será treinada com aprendizado supervisionado
        :param supervised_settings: Configurações do treinamento supervisionado
        :param stats_reporter: Reporter de estatísticas para TensorBoard
        :param use_sequential_model: Se deve usar o modelo sequencial em vez do modelo original
        """
        self.policy = policy
        self.settings = supervised_settings
        self.stats_reporter = stats_reporter
        self.use_sequential_model = use_sequential_model
        
        # print("Network settings:",policy.network_settings)
        
        if use_sequential_model:
            # Criar um modelo sequencial compatível com o behavior_spec existente
            self.sequential_actor = SequentialActor(
                observation_specs=self.policy.behavior_spec.observation_specs,
                network_settings=self.policy.network_settings,
                action_spec=self.policy.behavior_spec.action_spec,
                name_behavior=self.policy.behavior_spec.name,
                dropout_rate=getattr(self.settings, 'dropout_rate', 0.1)  # Adicionando dropout
            )
            
            # Substituir o otimizador para usar o novo modelo sequencial com weight decay para regularização
            self.optimizer = torch_module.optim.Adam(
                self.sequential_actor.parameters(),
                lr=self.settings.learning_rate,
                weight_decay=getattr(self.settings, 'weight_decay', 1e-4)  # Adicionando weight decay
            )
            
            # Atualizar a política para usar o modelo sequencial (isso afetará a exportação)
            self.original_actor = self.policy.actor
            self.policy.actor = self.sequential_actor
        else:
            # Configurar o otimizador e a função de perda para o modelo original com weight decay
            self.optimizer = torch_module.optim.Adam(
                self.policy.actor.parameters(),
                lr=self.settings.learning_rate,
                weight_decay=getattr(self.settings, 'weight_decay', 1e-4)  # Adicionando weight decay
            )
        
        # Usar MSE para regressão de ações
        self.criterion = nn.MSELoss()
        
        # Configurações para early stopping
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_model_state = None
        self.patience_counter = 0
        self.min_delta = self.settings.min_delta
        self.patience = self.settings.patience
        self.use_early_stopping = self.settings.early_stopping if hasattr(self.settings, 'early_stopping') else True  # Verificando se early stopping está ativado
        
        # Adicionando scheduler de taxa de aprendizado
        self.scheduler = torch_module.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=getattr(self.settings, 'lr_patience', 5)  # Paciência para redução da taxa de aprendizado
        )
        
        # Exporter para ONNX
        self.exporter = ModelSerializer(self.policy)
        
        self.device = default_device()
        if use_sequential_model:
            self.sequential_actor.to(self.device)
        else:
            self.policy.actor.to(self.device)
    
    def train_epoch(
        self,
        data_loader: torch_module.utils.data.DataLoader
    ) -> float:
        """
        Executa uma época de treinamento.
        :param data_loader: DataLoader com os dados de treinamento
        :return: Perda média na época
        """
        self.policy.actor.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (observations, actions) in enumerate(data_loader):
            observations = observations.to(self.device)
            actions = actions.to(self.device)
            
            # Preparar observações para o formato esperado pela política do ML-Agents
            processed_obs = self._prepare_observations(observations)
            
            # Obter as ações previstas pela política atual
            # Usando get_action_and_stats que retorna (action, run_out, memories)
            try:
                predicted_action_tuple, run_out, _ = self.policy.actor.get_action_and_stats(
                    processed_obs, masks=None  # masks=None para ações determinísticas
                )
                
                # Extrair ações da tupla retornada
                if hasattr(predicted_action_tuple, 'continuous_tensor') and predicted_action_tuple.continuous_tensor is not None:
                    predicted_actions = predicted_action_tuple.continuous_tensor
                elif hasattr(predicted_action_tuple, 'discrete_tensor') and predicted_action_tuple.discrete_tensor is not None:
                    predicted_actions = predicted_action_tuple.discrete_tensor
                else:
                    # Caso não encontremos ações contínuas ou discretas, tentamos outra abordagem
                    predicted_actions = None
            except:
                predicted_actions = None
            
            # Se ainda não encontrarmos ações, tentamos get_stats
            if predicted_actions is None:
                try:
                    run_out = self.policy.actor.get_stats(processed_obs)
                    # Extrair ações do run_out que é um dicionário de saída
                    if 'action' in run_out:
                        action_tuple = run_out['action']
                        if hasattr(action_tuple, 'continuous_tensor') and action_tuple.continuous_tensor is not None:
                            predicted_actions = action_tuple.continuous_tensor
                        elif hasattr(action_tuple, 'discrete_tensor') and action_tuple.discrete_tensor is not None:
                            predicted_actions = action_tuple.discrete_tensor
                except:
                    predicted_actions = None
            
            # Assegurar que as ações previstas e reais tenham o mesmo formato
            if predicted_actions is not None:
                # Converter ações para o formato correto se necessário
                if predicted_actions.shape != actions.shape:
                    # Se ações previstas têm mais dimensões do que as reais, reduzir
                    if len(predicted_actions.shape) > len(actions.shape):
                        # Talvez tenhamos ações extra ou dimensões extras que precisam ser ajustadas
                        if predicted_actions.shape[0] == actions.shape[0]:
                            # Mesmo número de amostras, ajustar as outras dimensões
                            if len(actions.shape) == 2 and actions.shape[1] == 1 and len(predicted_actions.shape) >= 2:
                                # Ações reais são colunas, achatamos as previstas para terem formato compatível
                                predicted_actions = predicted_actions.view(actions.shape)
                
                # Calcular a perda entre as ações previstas e as ações do CSV
                loss = self.criterion(predicted_actions, actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            else:
                print(f"WARNING: Não foi possível obter ações previstas para o batch {batch_idx}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(
        self,
        data_loader: torch_module.utils.data.DataLoader
    ) -> float:
        """
        Valida o modelo no conjunto de validação.
        :param data_loader: DataLoader com os dados de validação
        :return: Perda média na validação
        """
        self.policy.actor.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch_module.no_grad():
            for observations, actions in data_loader:
                observations = observations.to(self.device)
                actions = actions.to(self.device)
                
                # Preparar observações e obter previsões
                processed_obs = self._prepare_observations(observations)
                
                # Obter as ações previstas pela política atual
                # Usando get_action_and_stats que retorna (action, run_out, memories)
                try:
                    predicted_action_tuple, run_out, _ = self.policy.actor.get_action_and_stats(
                        processed_obs, masks=None  # masks=None para ações determinísticas
                    )
                    
                    # Extrair ações da tupla retornada
                    if hasattr(predicted_action_tuple, 'continuous_tensor') and predicted_action_tuple.continuous_tensor is not None:
                        predicted_actions = predicted_action_tuple.continuous_tensor
                    elif hasattr(predicted_action_tuple, 'discrete_tensor') and predicted_action_tuple.discrete_tensor is not None:
                        predicted_actions = predicted_action_tuple.discrete_tensor
                    else:
                        predicted_actions = None
                except:
                    predicted_actions = None
                
                # Se ainda não encontrarmos ações, tentamos get_stats
                if predicted_actions is None:
                    try:
                        run_out = self.policy.actor.get_stats(processed_obs)
                        # Extrair ações do run_out que é um dicionário de saída
                        if 'action' in run_out:
                            action_tuple = run_out['action']
                            if hasattr(action_tuple, 'continuous_tensor') and action_tuple.continuous_tensor is not None:
                                predicted_actions = action_tuple.continuous_tensor
                            elif hasattr(action_tuple, 'discrete_tensor') and action_tuple.discrete_tensor is not None:
                                predicted_actions = action_tuple.discrete_tensor
                    except:
                        predicted_actions = None
                
                if predicted_actions is not None:
                    # Ajustar formato das ações se necessário
                    if predicted_actions.shape != actions.shape:
                        if len(predicted_actions.shape) > len(actions.shape):
                            predicted_actions = predicted_actions.view(actions.shape)
                    
                    loss = self.criterion(predicted_actions, actions)
                    total_loss += loss.item()
                    num_batches += 1
                else:
                    print("WARNING: Não foi possível obter ações previstas para validação")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def _prepare_observations(self, observations: torch_module.Tensor) -> List[torch_module.Tensor]:
        """
        Prepara as observações para entrada na política do ML-Agents.
        Converte de um tensor simples para a lista de tensores que a política espera.
        """
        # Obter as especificações de observação da política
        obs_specs = self.policy.behavior_spec.observation_specs
        
        # Dividir o tensor plano nas diferentes observações esperadas
        prepared_obs = []
        current_idx = 0
        
        for obs_spec in obs_specs:
            # Calcular o tamanho da observação esperada
            obs_size = int(np.prod(obs_spec.shape))
            # Extrair os dados para esta observação específica
            obs_tensor = observations[:, current_idx:current_idx + obs_size].contiguous()
            # Redimensionar para a forma esperada: (batch_size, *obs_shape)
            obs_tensor = obs_tensor.view(observations.size(0), *obs_spec.shape)
            prepared_obs.append(obs_tensor)
            current_idx += obs_size
        
        return prepared_obs
    
    def should_stop_early(self, val_loss: float, current_epoch: int, metrics: Dict[str, Any]) -> bool:
        """
        Verifica se o treinamento deve parar cedo devido à falta de melhoria.
        :param val_loss: Perda atual na validação
        :param current_epoch: Época atual (0-indexada)
        :param metrics: Dicionário de métricas a ser atualizado
        :return: True se o treinamento deve parar cedo
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = current_epoch
            self.patience_counter = 0
            # Atualizar o best_epoch quando encontramos uma melhor perda
            metrics['best_epoch'] = current_epoch + 1  # 1-indexado para consistência com relatórios
            # Salvar o estado do melhor modelo
            self.best_model_state = {key: value.clone() for key, value in self.policy.actor.state_dict().items()}
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.patience
    
    def train(
        self,
        csv_path: str,
        observation_columns: List[str],
        action_columns: List[str],
        batch_size: Optional[int] = None,
        num_epochs: Optional[int] = None,
        validation_split: Optional[float] = None,
        shuffle: Optional[bool] = None,
        augment_noise: Optional[float] = None,
        checkpoint_interval: Optional[int] = None,
        artifact_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executa o treinamento supervisionado completo.
        :param csv_path: Caminho para o arquivo CSV de treinamento
        :param observation_columns: Colunas de observação no CSV
        :param action_columns: Colunas de ação no CSV
        :param batch_size: Tamanho do batch (usa configuração se não fornecido)
        :param num_epochs: Número de épocas de treinamento (usa configuração se não fornecido)
        :param validation_split: Fração de dados para validação (usa configuração se não fornecido)
        :param shuffle: Se os dados devem ser embaralhados (usa configuração se não fornecido)
        :param augment_noise: Nível de ruído para aumentação de dados (usa configuração se não fornecido)
        :param checkpoint_interval: Intervalo de épocas para salvar ONNX (usa configuração se não fornecido)
        :param artifact_path: Caminho para salvar artefatos (modelos ONNX)
        :return: Métricas do treinamento
        """
        # Usar configurações padrão se não forem fornecidas
        batch_size = batch_size or self.settings.batch_size
        num_epochs = num_epochs or self.settings.num_epoch
        validation_split = validation_split or self.settings.validation_split
        shuffle = shuffle or self.settings.shuffle
        augment_noise = augment_noise or self.settings.augment_noise
        checkpoint_interval = checkpoint_interval or self.settings.checkpoint_interval
        # Carregar dados
        data_loader = SupervisedDataLoader(
            csv_path=csv_path,
            observation_columns=observation_columns,
            action_columns=action_columns,
            validation_split=validation_split,
            shuffle=shuffle,
            augment_noise=augment_noise,
            action_spec=self.policy.behavior_spec.action_spec  # Adicionando action_spec para validação
        )
        train_loader = data_loader.get_train_loader(batch_size)
        val_loader = data_loader.get_validation_loader(batch_size)
        
        metrics = {
            'train_losses': [],
            'val_losses': [],
            'best_epoch': 0
        }
        
        print("Iniciando treinamento supervisionado...")
        print(
            """


  ░▒▓███████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓████████▓▒░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓███████▓▒░▒▓████████▓▒░▒▓███████▓▒░  
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
 ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓██████▓▒░ ░▒▓███████▓▒░ ░▒▓█▓▒▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓██████▓▒░ ░▒▓█▓▒░░▒▓█▓▒░ 
       ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▓█▓▒░ ░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
       ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▓█▓▒░ ░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓██▓▒░  ░▒▓█▓▒░▒▓███████▓▒░░▒▓████████▓▒░▒▓███████▓▒░  
                                                                                                                             
                                                                                                                                                                                         



        """
        )

        
        for epoch in range(num_epochs):
            # Treinamento
            train_loss = self.train_epoch(train_loader)
            
            # Validação
            val_loss = self.validate(val_loader)
            
            metrics['train_losses'].append(train_loss)
            metrics['val_losses'].append(val_loss)
            
            # Atualizar o scheduler com base na perda de validação
            self.scheduler.step(val_loss)
            
            # # Reportar estatísticas ao TensorBoard
            # if self.stats_reporter is not None:
            #     self.stats_reporter.add_stat("Supervised/Train_Loss", train_loss)
            #     self.stats_reporter.add_stat("Supervised/Validation_Loss", val_loss)
            #     # Escrever as estatísticas imediatamente para garantir que apareçam no TensorBoard
            #     # Usar o número da época como step para manter a consistência
            #     self.stats_reporter.write_stats(epoch + 1)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Exportar ONNX em intervalos definidos
            if artifact_path and (epoch + 1) % checkpoint_interval == 0:
                try:
                    checkpoint_path = f"{artifact_path}/supervised_{epoch+1}"
                    self.exporter.export_policy_model(checkpoint_path)
                    print(f"Modelo ONNX exportado: {checkpoint_path}.onnx")
                except Exception as e:
                    print(f"Erro ao exportar modelo ONNX na época {epoch+1}: {e}")
            
            # Verificar early stopping
            if self.use_early_stopping and self.should_stop_early(val_loss, epoch, metrics):
                print(f"Early stopping at epoch {epoch+1}")
                # metrics['best_epoch'] já foi atualizado corretamente na função should_stop_early
                break
        
        # Após o treinamento, restaurar o melhor modelo se early stopping estiver ativado
        if self.use_early_stopping and self.best_model_state is not None:
            self.restore_best_model()
        
        # Exportar modelo final (se early stopping estiver ativado, será o melhor modelo encontrado)
        if artifact_path:
            try:
                final_path = f"{artifact_path}/supervised_final"
                self.exporter.export_policy_model(final_path)
                print(f"Modelo ONNX final exportado: {final_path}.onnx")
            except Exception as e:
                print(f"Erro ao exportar modelo ONNX final: {e}")
            
            # Exportar o melhor modelo encontrado durante o treinamento
            if self.best_model_state is not None:
                try:
                    best_path = f"{artifact_path}/supervised_best"
                    self.exporter.export_policy_model(best_path)
                    print(f"Melhor modelo ONNX exportado: {best_path}.onnx (época {self.best_epoch + 1})")
                except Exception as e:
                    print(f"Erro ao exportar melhor modelo ONNX: {e}")
        
        print("Treinamento supervisionado concluído.")

        # Salvar pesos treinados para verificação posterior
        self._save_trained_weights(artifact_path)
        
        # Avaliação adicional para verificar a qualidade do modelo
        if artifact_path:
            self._evaluate_model_quality(val_loader, artifact_path)
        
        # Restaurar o modelo original se estiver usando modelo sequencial
        # Isso é crucial para garantir que o treinamento de RL não seja afetado
        if self.use_sequential_model and hasattr(self, 'original_actor'):
            self.policy.actor = self.original_actor
            print("[AUDIT] Modelo original restaurado para treinamento de RL")


 

                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                    
        return metrics

    def _save_trained_weights(self, artifact_path: Optional[str]) -> None:
        """
        Salva os pesos treinados para verificação posterior.
        :param artifact_path: Caminho para salvar os pesos
        """
        if artifact_path:
            try:
                import os
                weights_path = f"{artifact_path}/supervised_weights.pth"
                # Criar diretório se não existir
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                torch_module.save(self.policy.actor.state_dict(), weights_path)
                print(f"[AUDIT] Pesos supervisionados salvos: {weights_path}")
                # Calcular e registrar hash dos pesos para verificação
                weights_hash = self._calculate_weights_hash()
                print(f"[AUDIT] Hash dos pesos supervisionados: {weights_hash}")
            except Exception as e:
                print(f"Erro ao salvar pesos supervisionados: {e}")
    
    def _calculate_weights_hash(self) -> str:
        """
        Calcula um hash dos pesos atuais para verificação.
        :return: Hash MD5 dos pesos atuais
        """
        try:
            import hashlib
            import pickle
            
            # Serializar os pesos atuais
            state_dict = self.policy.actor.state_dict()
            # Ordenar as chaves para garantir consistência
            sorted_items = sorted(state_dict.items())
            weights_bytes = pickle.dumps(sorted_items)
            # Calcular hash
            return hashlib.md5(weights_bytes).hexdigest()
        except Exception as e:
            print(f"Erro ao calcular hash dos pesos: {e}")
            return "unknown"
    
    def restore_best_model(self):
        """
        Restaura o estado do modelo com o melhor desempenho encontrado durante o treinamento.
        """
        if self.best_model_state is not None:
            if self.use_sequential_model and hasattr(self, 'sequential_actor'):
                # Carregar o melhor estado no modelo sequencial
                self.sequential_actor.load_state_dict(self.best_model_state)
                # Atualizar a política atual para usar o modelo com o melhor estado
                self.policy.actor = self.sequential_actor
            else:
                self.policy.actor.load_state_dict(self.best_model_state)
            print(f"Melhor modelo restaurado da época {self.best_epoch + 1} com perda {self.best_loss:.6f}")
        else:
            print("Nenhum melhor modelo encontrado para restaurar")
    
    def get_trained_weights_hash(self) -> str:
        """
        Retorna o hash dos pesos treinados para verificação.
        :return: Hash MD5 dos pesos treinados
        """
        return self._calculate_weights_hash()
    
    def compare_weights_with_saved(self, weights_file_path: str) -> bool:
        """
        Compara os pesos atuais com pesos salvos em arquivo.
        :param weights_file_path: Caminho para arquivo de pesos
        :return: True se os pesos são iguais, False caso contrário
        """
        try:
            import os
            if not os.path.exists(weights_file_path):
                print(f"Arquivo de pesos não encontrado: {weights_file_path}")
                return False
                
            current_state = self.policy.actor.state_dict()
            saved_state = torch_module.load(weights_file_path)
            
            # Comparar cada tensor de peso
            for key in current_state.keys():
                if key not in saved_state:
                    print(f"Chave {key} não encontrada nos pesos salvos")
                    return False
                    
                if not torch_module.equal(current_state[key], saved_state[key]):
                    print(f"Pesos diferentes na chave {key}")
                    return False
                    
            print("Pesos verificados: IGUAIS")
            return True
            
        except Exception as e:
            print(f"Erro ao comparar pesos: {e}")
            return False
    
    def _evaluate_model_quality(self, val_loader, artifact_path: str) -> None:
        """
        Avaliação adicional para verificar a qualidade do modelo.
        """
        print("Avaliando qualidade do modelo...")
        
        # Calcular métricas adicionais
        self.policy.actor.eval()
        all_predictions = []
        all_targets = []
        
        with torch_module.no_grad():
            for observations, actions in val_loader:
                observations = observations.to(self.device)
                actions = actions.to(self.device)
                
                # Preparar observações e obter previsões
                processed_obs = self._prepare_observations(observations)
                
                # Obter as ações previstas pela política atual
                try:
                    predicted_action_tuple, run_out, _ = self.policy.actor.get_action_and_stats(
                        processed_obs, masks=None
                    )
                    
                    # Extrair ações da tupla retornada
                    if hasattr(predicted_action_tuple, 'continuous_tensor') and predicted_action_tuple.continuous_tensor is not None:
                        predicted_actions = predicted_action_tuple.continuous_tensor
                    elif hasattr(predicted_action_tuple, 'discrete_tensor') and predicted_action_tuple.discrete_tensor is not None:
                        predicted_actions = predicted_action_tuple.discrete_tensor
                    else:
                        continue
                except:
                    continue
                
                # Ajustar formato das ações se necessário
                if predicted_actions.shape != actions.shape:
                    if len(predicted_actions.shape) > len(actions.shape):
                        predicted_actions = predicted_actions.view(actions.shape)
                
                all_predictions.append(predicted_actions.cpu())
                all_targets.append(actions.cpu())
        
        if all_predictions and all_targets:
            all_predictions = torch_module.cat(all_predictions, dim=0)
            all_targets = torch_module.cat(all_targets, dim=0)
            
            # Calcular métricas adicionais
            mse = torch_module.mean((all_predictions - all_targets) ** 2).item()
            mae = torch_module.mean(torch_module.abs(all_predictions - all_targets)).item()
            
            # Calcular R² (coeficiente de determinação)
            ss_res = torch_module.sum((all_targets - all_predictions) ** 2)
            ss_tot = torch_module.sum((all_targets - torch_module.mean(all_targets)) ** 2)
            r2 = (1 - ss_res / ss_tot).item() if ss_tot != 0 else 0
            
            print(f"Métricas de Avaliação - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
            
            # Salvar métricas em arquivo
            import os
            metrics_path = f"{artifact_path}/supervised_metrics.txt"
            with open(metrics_path, 'w') as f:
                f.write(f"MSE: {mse:.6f}\n")
                f.write(f"MAE: {mae:.6f}\n")
                f.write(f"R²: {r2:.6f}\n")
                f.write(f"Best Epoch: {self.best_epoch + 1 if self.best_epoch is not None else 'N/A'}\n")
                f.write(f"Best Val Loss: {self.best_loss:.6f}\n")
            
            print(f"Métricas salvas em: {metrics_path}")
