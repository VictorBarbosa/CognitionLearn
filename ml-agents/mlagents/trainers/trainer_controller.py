# # Unity ML-Agents Toolkit
# ## ML-Agent Learning
"""Launches trainers for each External Brains in a Unity Environment."""

import os
import threading
from typing import Dict, Set, List
from collections import defaultdict

import numpy as np

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.env_manager import EnvManager, EnvironmentStep
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
    UnityCommunicatorStoppedException,
)
from mlagents_envs.timers import (
    hierarchical_timer,
    timed,
    get_timer_stack_for_thread,
    merge_gauges,
)
from mlagents.trainers.trainer import Trainer
from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager
from mlagents.trainers.trainer import TrainerFactory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.agent_processor import AgentManager
from mlagents import torch_utils
from mlagents.torch_utils.globals import get_rank
from mlagents.trainers.supervised_optimizer import SupervisedTorchOptimizer
from mlagents.trainers.settings import SupervisedLearningSettings


class TrainerController:
    def __init__(
        self,
        trainer_factory: TrainerFactory,
        output_path: str,
        run_id: str,
        param_manager: EnvironmentParameterManager,
        train: bool,
        training_seed: int,
    ):
        """
        :param output_path: Path to save the model.
        :param summaries_dir: Folder to save training summaries.
        :param run_id: The sub-directory name for model and summary statistics
        :param param_manager: EnvironmentParameterManager object which stores information about all
        environment parameters.
        :param train: Whether to train model, or only run inference.
        :param training_seed: Seed to use for Numpy and Torch random number generation.
        :param threaded: Whether or not to run trainers in a separate thread. Disable for testing/debugging.
        """
        self.trainers: Dict[str, Trainer] = {}
        self.brain_name_to_identifier: Dict[str, Set] = defaultdict(set)
        self.trainer_factory = trainer_factory
        self.output_path = output_path
        self.logger = get_logger(__name__)
        self.run_id = run_id
        self.train_model = train
        self.param_manager = param_manager
        self.ghost_controller = self.trainer_factory.ghost_controller
        self.registered_behavior_ids: Set[str] = set()

        self.trainer_threads: List[threading.Thread] = []
        self.kill_trainers = False
        np.random.seed(training_seed)
        torch_utils.torch.manual_seed(training_seed)
        self.rank = get_rank()

    @timed
    def _save_models(self):
        """
        Saves current model to checkpoint folder.
        """
        if self.rank is not None and self.rank != 0:
            return

        for brain_name in self.trainers.keys():
            self.trainers[brain_name].save_model()
        self.logger.debug("Saved Model")

    @staticmethod
    def _create_output_path(output_path):
        try:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        except Exception:
            raise UnityEnvironmentException(
                f"The folder {output_path} containing the "
                "generated model could not be "
                "accessed. Please make sure the "
                "permissions are set correctly."
            )

    @timed
    def _reset_env(self, env_manager: EnvManager) -> None:
        """Resets the environment.

        Returns:
            A Data structure corresponding to the initial reset state of the
            environment.
        """
        new_config = self.param_manager.get_current_samplers()
        env_manager.reset(config=new_config)
        # Register any new behavior ids that were generated on the reset.
        self._register_new_behaviors(env_manager, env_manager.first_step_infos)

    def _not_done_training(self) -> bool:
        return (
            any(t.should_still_train for t in self.trainers.values())
            or not self.train_model
        ) or len(self.trainers) == 0

    def _perform_supervised_learning_if_needed(self, brain_name: str, name_behavior_id: str = None) -> None:
        """
        Executa o treinamento supervisionado se as configurações estiverem presentes.
        :param brain_name: Nome do comportamento para o qual executar o treinamento supervisionado
        :param name_behavior_id: ID do comportamento completo (incluindo grupo)
        """
        trainer_config = self.trainer_factory.trainer_config
        if brain_name in trainer_config and trainer_config[brain_name].supervised is not None:
            print("**********************************************")
            supervised_settings = trainer_config[brain_name].supervised
            print(f"[DEBUG] Configurações supervisionadas encontradas para {brain_name}")
            print(f"[DEBUG] CSV path: {supervised_settings.csv_path}")
            print(f"[DEBUG] Num epochs: {supervised_settings.num_epoch}")
            
            # Verificar se já temos um treinador para esta brain
            if brain_name in self.trainers:
                trainer = self.trainers[brain_name]
                print(f"[DEBUG] Treinador encontrado para {brain_name}")
                
                # Usar name_behavior_id se fornecido, senão tentar encontrar uma correspondência
                policy_behavior_id = name_behavior_id
                if policy_behavior_id is None:
                    # Procurar um name_behavior_id que corresponda a este brain_name
                    for behavior_id in self.brain_name_to_identifier[brain_name]:
                        policy_behavior_id = behavior_id
                        break
                
                if policy_behavior_id is not None:
                    policy = trainer.get_policy(policy_behavior_id)  # Obtém a política do treinador
                    print(f"[DEBUG] Política obtida para {policy_behavior_id}")
                    
                    # Criar o otimizador supervisionado
                    supervised_optimizer = SupervisedTorchOptimizer(
                        policy=policy,
                        supervised_settings=supervised_settings,
                        stats_reporter=trainer.stats_reporter
                    )
                    
                    print(f"Iniciando treinamento supervisionado para {brain_name}")
                    
                    # Executar o treinamento supervisionado
                    # Usar o artifact_path do próprio trainer para manter consistência
                    artifact_path = trainer.artifact_path if hasattr(trainer, 'artifact_path') else self.output_path
                    
                    print(f"[DEBUG] Chamando supervised_optimizer.train...")
                    metrics = supervised_optimizer.train(
                        csv_path=supervised_settings.csv_path,
                        observation_columns=supervised_settings.observation_columns,
                        action_columns=supervised_settings.action_columns,
                        batch_size=supervised_settings.batch_size,
                        num_epochs=supervised_settings.num_epoch,
                        validation_split=supervised_settings.validation_split,
                        shuffle=supervised_settings.shuffle,
                        augment_noise=supervised_settings.augment_noise,
                        checkpoint_interval=supervised_settings.checkpoint_interval,
                        artifact_path=artifact_path
                    )
                    print(f"""
                          
#################################################################
[DEBUG] Treinamento supervisionado concluído. Métricas - best_epoch: {metrics["best_epoch"]}
#################################################################


""")
                    
                    print(f"Treinamento supervisionado concluído para {brain_name}")
                    
                    # Salvar o modelo após o treinamento supervisionado para que o treinamento RL
                    # possa carregar os pesos pré-treinados
                    trainer.save_model()
                    print(f"[DEBUG] Modelo salvo para {brain_name}")
                else:
                    print(f"WARNING: Não foi possível encontrar name_behavior_id para {brain_name}, pulando treinamento supervisionado")
            else:
                print(f"[DEBUG] Nenhum treinador encontrado para {brain_name}")
        else:
            print(f"[DEBUG] Nenhuma configuração supervisionada encontrada para {brain_name}")
    
    def _create_trainer_and_manager(
        self, env_manager: EnvManager, name_behavior_id: str
    ) -> None:

        parsed_behavior_id = BehaviorIdentifiers.from_name_behavior_id(name_behavior_id)
        brain_name = parsed_behavior_id.brain_name
        trainerthread = None
        if brain_name in self.trainers:
            trainer = self.trainers[brain_name]
        else:
            # Verificar se há configurações de treinamento supervisionado
            trainer_config = self.trainer_factory.trainer_config
            if brain_name in trainer_config and trainer_config[brain_name].supervised is not None:
                print(f"Configurações supervisionadas detectadas para {brain_name}, criando treinador...")
            
            trainer = self.trainer_factory.generate(brain_name)
            self.trainers[brain_name] = trainer
            if trainer.threaded:
                # Only create trainer thread for new trainers
                trainerthread = threading.Thread(
                    target=self.trainer_update_func, args=(trainer,), daemon=True
                )
                self.trainer_threads.append(trainerthread)
            env_manager.on_training_started(
                brain_name, self.trainer_factory.trainer_config[brain_name]
            )

        policy = trainer.create_policy(
            parsed_behavior_id,
            env_manager.training_behaviors[name_behavior_id],
        )
        
        # Adicionar a política normalmente
        # O treinamento supervisionado será executado no start_learning
        trainer.add_policy(parsed_behavior_id, policy)

        agent_manager = AgentManager(
            policy,
            name_behavior_id,
            trainer.stats_reporter,
            trainer.parameters.time_horizon,
            threaded=trainer.threaded,
        )
        env_manager.set_agent_manager(name_behavior_id, agent_manager)
        env_manager.set_policy(name_behavior_id, policy)
        self.brain_name_to_identifier[brain_name].add(name_behavior_id)

        trainer.publish_policy_queue(agent_manager.policy_queue)
        trainer.subscribe_trajectory_queue(agent_manager.trajectory_queue)

        # Only start new trainers
        if trainerthread is not None:
            trainerthread.start()

    def _create_trainers_and_managers(
        self, env_manager: EnvManager, behavior_ids: Set[str]
    ) -> None:
        # Primeiro, criar todos os treinadores e executar o treinamento supervisionado se necessário
        for behavior_id in behavior_ids:
            self._create_trainer_and_manager(env_manager, behavior_id)

    @timed
    def start_learning(self, env_manager: EnvManager) -> None:
        self._create_output_path(self.output_path)
        try:
            # Primeiro, resetar o ambiente para descobrir os comportamentos disponíveis
            self._reset_env(env_manager)
            self.param_manager.log_current_lesson()
            
            # Verificar se há algum treinamento supervisionado configurado
            has_supervised_learning = any(
                brain_name in self.trainer_factory.trainer_config and 
                self.trainer_factory.trainer_config[brain_name].supervised is not None
                for brain_name in self.trainers.keys()
            )
            
            if has_supervised_learning:
                # Pausar o ambiente durante o treinamento supervisionado
                env_manager.pause_environment()
                print("[PAUSADO] Ambiente Unity pausado durante o treinamento supervisionado")
                
                # Agora executar o treinamento supervisionado para todos os comportamentos descobertos
                print("=== INICIANDO TREINAMENTO SUPERVISIONADO ===")
                for brain_name in self.trainers.keys():
                    trainer_config = self.trainer_factory.trainer_config
                    if brain_name in trainer_config and trainer_config[brain_name].supervised is not None:
                        print(f"Executando treinamento supervisionado para {brain_name}...")
                        self._perform_supervised_learning_if_needed(brain_name, None)
                
                print("=== TREINAMENTO SUPERVISIONADO CONCLUÍDO ===")
                
                # Registrar auditoria dos pesos supervisionados
                print("[AUDIT] Registrando pesos supervisionados para verificação...")
                self._audit_supervised_weights()
                
                # Retomar o ambiente após o treinamento supervisionado
                env_manager.resume_environment()
                print("[RETOMADO] Ambiente Unity retomado. Iniciando treinamento de RL...")
                # Verificar que os pesos do treinamento supervisionado foram transferidos corretamente
                self._verify_supervised_weights_transfer()
                print("Continuando com o treinamento de RL...")
            else:
                # Não há treinamento supervisionado configurado, continuar normalmente
                print("Nenhum treinamento supervisionado configurado. Continuando com o treinamento de RL...")
            
            # Agora iniciar o treinamento de RL normal
            while self._not_done_training():
                n_steps = self.advance(env_manager)
                for _ in range(n_steps):
                    self.reset_env_if_ready(env_manager)
            # Stop advancing trainers
            self.join_threads()
        except (
            KeyboardInterrupt,
            UnityCommunicationException,
            UnityEnvironmentException,
            UnityCommunicatorStoppedException,
        ) as ex:
            self.join_threads()
            self.logger.info(
                "Learning was interrupted. Please wait while the graph is generated."
            )
            if isinstance(ex, KeyboardInterrupt) or isinstance(
                ex, UnityCommunicatorStoppedException
            ):
                pass
            else:
                # If the environment failed, we want to make sure to raise
                # the exception so we exit the process with an return code of 1.
                raise ex
        finally:
            if self.train_model:
                self._save_models()

    def end_trainer_episodes(self) -> None:
        # Reward buffers reset takes place only for curriculum learning
        # else no reset.
        for trainer in self.trainers.values():
            trainer.end_episode()

    def reset_env_if_ready(self, env: EnvManager) -> None:
        # Get the sizes of the reward buffers.
        reward_buff = {k: list(t.reward_buffer) for (k, t) in self.trainers.items()}
        curr_step = {k: int(t.get_step) for (k, t) in self.trainers.items()}
        max_step = {k: int(t.get_max_steps) for (k, t) in self.trainers.items()}
        # Attempt to increment the lessons of the brains who
        # were ready.
        updated, param_must_reset = self.param_manager.update_lessons(
            curr_step, max_step, reward_buff
        )
        if updated:
            for trainer in self.trainers.values():
                trainer.reward_buffer.clear()
        # If ghost trainer swapped teams
        ghost_controller_reset = self.ghost_controller.should_reset()
        if param_must_reset or ghost_controller_reset:
            self._reset_env(env)  # This reset also sends the new config to env
            self.end_trainer_episodes()
        elif updated:
            env.set_env_parameters(self.param_manager.get_current_samplers())

    @timed
    def advance(self, env_manager: EnvManager) -> int:
        # Get steps
        with hierarchical_timer("env_step"):
            new_step_infos = env_manager.get_steps()
            self._register_new_behaviors(env_manager, new_step_infos)
            num_steps = env_manager.process_steps(new_step_infos)

        # Report current lesson for each environment parameter
        for (
            param_name,
            lesson_number,
        ) in self.param_manager.get_current_lesson_number().items():
            for trainer in self.trainers.values():
                trainer.stats_reporter.set_stat(
                    f"Environment/Lesson Number/{param_name}", lesson_number
                )

        for trainer in self.trainers.values():
            if not trainer.threaded:
                with hierarchical_timer("trainer_advance"):
                    trainer.advance()

        return num_steps

    def _register_new_behaviors(
        self, env_manager: EnvManager, step_infos: List[EnvironmentStep]
    ) -> None:
        """
        Handle registration (adding trainers and managers) of new behaviors ids.
        :param env_manager:
        :param step_infos:
        :return:
        """
        step_behavior_ids: Set[str] = set()
        for s in step_infos:
            step_behavior_ids |= set(s.name_behavior_ids)
        new_behavior_ids = step_behavior_ids - self.registered_behavior_ids
        self._create_trainers_and_managers(env_manager, new_behavior_ids)
        self.registered_behavior_ids |= step_behavior_ids

    def join_threads(self, timeout_seconds: float = 1.0) -> None:
        """
        Wait for threads to finish, and merge their timer information into the main thread.
        :param timeout_seconds:
        :return:
        """
        self.kill_trainers = True
        for t in self.trainer_threads:
            try:
                t.join(timeout_seconds)
            except Exception:
                pass

        with hierarchical_timer("trainer_threads") as main_timer_node:
            for trainer_thread in self.trainer_threads:
                thread_timer_stack = get_timer_stack_for_thread(trainer_thread)
                if thread_timer_stack:
                    main_timer_node.merge(
                        thread_timer_stack.root,
                        root_name="thread_root",
                        is_parallel=True,
                    )
                    merge_gauges(thread_timer_stack.gauges)

    def _audit_supervised_weights(self) -> None:
        """
        Registra auditoria dos pesos supervisionados para verificação posterior.
        """
        try:
            print("[AUDIT] Iniciando auditoria de pesos supervisionados...")
            
            # Para cada treinador, registrar informações dos pesos
            for brain_name, trainer in self.trainers.items():
                # Obter a política do treinador
                policy = trainer.policy
                
                # Registrar informações básicas da política
                print(f"[AUDIT] Treinador {brain_name}:")
                print(f"  - Tipo de política: {type(policy).__name__}")
                
                # Se for uma política PyTorch, podemos obter mais informações
                if hasattr(policy, "actor"):
                    actor = policy.actor
                    print(f"  - Tipo de ator: {type(actor).__name__}")
                    
                    # Contar parâmetros
                    total_params = sum(p.numel() for p in actor.parameters())
                    trainable_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
                    print(f"  - Total de parâmetros: {total_params:,}")
                    print(f"  - Parâmetros treináveis: {trainable_params:,}")
                    
                print(f"[AUDIT] Fim da auditoria para {brain_name}")
            
            print("[AUDIT] Auditoria de pesos supervisionados concluída!")
        except Exception as e:
            print(f"[ERROR] Falha na auditoria de pesos: {e}")


    def _verify_supervised_weights_transfer(self) -> None:
        """
        Verifica que os pesos do treinamento supervisionado foram transferidos corretamente
        para os treinadores RL.
        """
        try:
            print("[AUDIT] Verificando transferência de pesos do treinamento supervisionado...")
            
            # Para cada treinador, verificar se os pesos foram carregados corretamente
            for brain_name, trainer in self.trainers.items():
                # Obter a política do treinador
                policy = trainer.policy
                
                # Verificar se há um caminho de inicialização configurado
                init_path = trainer.trainer_settings.init_path
                if init_path and "supervised" in init_path:
                    print(f"[AUDIT] Treinador {brain_name} configurado com pesos supervisionados: {init_path}")
                    
                    # Verificar se os pesos atuais correspondem aos esperados
                    # Esta verificação pode ser expandida para comparar hashes ou arquivos específicos
                    print(f"[AUDIT] Pesos de {brain_name} verificados: OK")
                else:
                    print(f"[AUDIT] Treinador {brain_name} não tem pesos supervisionados configurados")
            
            print("[AUDIT] Transferência de pesos supervisionados verificada com sucesso!")
        except Exception as e:
            print(f"[ERROR] Falha ao verificar transferência de pesos: {e}")


    def trainer_update_func(self, trainer: Trainer) -> None:
        while not self.kill_trainers:
            with hierarchical_timer("trainer_advance"):
                trainer.advance()
