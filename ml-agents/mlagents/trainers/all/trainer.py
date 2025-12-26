
import os
import cattr
import copy
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.sac.trainer import SACTrainer
from mlagents.trainers.td3.trainer import TD3Trainer
from mlagents.trainers.tdsac.trainer import TDSACTrainer
from mlagents.trainers.tqc.trainer import TQCTrainer
from mlagents.trainers.poca.trainer import POCATrainer
from mlagents.trainers.drqv2.trainer import DrQv2Trainer
from mlagents.trainers.dcac.trainer import DCACTrainer
from mlagents.trainers.crossq.trainer import CrossQTrainer
from mlagents.trainers.ppo_et.trainer import PPOETTrainer
from mlagents.trainers.ppo_ce.trainer import PPOCETrainer
from mlagents.trainers.sac_ae.trainer import SACAETrainer

from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.policy import Policy
from mlagents.trainers.all.policy import AllPolicy
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.settings import TrainerSettings, deep_update_dict

class AllTrainer(RLTrainer):
    def __init__(self, behavior_name, reward_buff_cap, trainer_settings, training, load, seed, artifact_path):
        super().__init__(behavior_name, trainer_settings, training, load, artifact_path, reward_buff_cap)
        self.seed = seed
        
        base_settings = trainer_settings.as_dict()
        # Pop all possible trainer keys to create a clean base config
        for trainer_name in ["ppo", "sac", "td3", "tdsac", "tqc", "poca", "drqv2", "dcac", "crossq","ppo_et", "ppo_ce", "sac_ae",  "trainer_type"]:
            base_settings.pop(trainer_name, None)

        self.trainers = []

        # PPO Trainer Setup
        if hasattr(trainer_settings, "ppo") and trainer_settings.ppo is not None:
            ppo_config = cattr.unstructure(trainer_settings.ppo)
            ppo_full_config = copy.deepcopy(base_settings)
            deep_update_dict(ppo_full_config, ppo_config)
            ppo_full_config["trainer_type"] = "ppo"
            ppo_trainer_settings = cattr.structure(ppo_full_config, TrainerSettings)
            ppo_artifact_path = os.path.join(artifact_path, "ppo")
            ppo_brain_name = f"{behavior_name}_ppo"
            self.ppo_trainer = PPOTrainer(ppo_brain_name, reward_buff_cap, ppo_trainer_settings, training, load, seed, ppo_artifact_path)
            self.trainers.append(self.ppo_trainer)

        # SAC Trainer Setup
        if hasattr(trainer_settings, "sac") and trainer_settings.sac is not None:
            sac_config = cattr.unstructure(trainer_settings.sac)
            sac_full_config = copy.deepcopy(base_settings)
            deep_update_dict(sac_full_config, sac_config)
            sac_full_config["trainer_type"] = "sac"
            sac_trainer_settings = cattr.structure(sac_full_config, TrainerSettings)
            sac_artifact_path = os.path.join(artifact_path, "sac")
            sac_brain_name = f"{behavior_name}_sac"
            self.sac_trainer = SACTrainer(sac_brain_name, reward_buff_cap, sac_trainer_settings, training, load, seed, sac_artifact_path)
            self.trainers.append(self.sac_trainer)

        # TD3 Trainer Setup
        if hasattr(trainer_settings, "td3") and trainer_settings.td3 is not None:
            td3_config = cattr.unstructure(trainer_settings.td3)
            td3_full_config = copy.deepcopy(base_settings)
            deep_update_dict(td3_full_config, td3_config)
            td3_full_config["trainer_type"] = "td3"
            td3_trainer_settings = cattr.structure(td3_full_config, TrainerSettings)
            td3_artifact_path = os.path.join(artifact_path, "td3")
            td3_brain_name = f"{behavior_name}_td3"
            self.td3_trainer = TD3Trainer(td3_brain_name, reward_buff_cap, td3_trainer_settings, training, load, seed, td3_artifact_path)
            self.trainers.append(self.td3_trainer)

        # TDSAC Trainer Setup
        if hasattr(trainer_settings, "tdsac") and trainer_settings.tdsac is not None:
            tdsac_config = cattr.unstructure(trainer_settings.tdsac)
            tdsac_full_config = copy.deepcopy(base_settings)
            deep_update_dict(tdsac_full_config, tdsac_config)
            tdsac_full_config["trainer_type"] = "tdsac"
            tdsac_trainer_settings = cattr.structure(tdsac_full_config, TrainerSettings)
            tdsac_artifact_path = os.path.join(artifact_path, "tdsac")
            tdsac_brain_name = f"{behavior_name}_tdsac"
            self.tdsac_trainer = TDSACTrainer(tdsac_brain_name, reward_buff_cap, tdsac_trainer_settings, training, load, seed, tdsac_artifact_path)
            self.trainers.append(self.tdsac_trainer)

        # TQC Trainer Setup
        if hasattr(trainer_settings, "tqc") and trainer_settings.tqc is not None:
            tqc_config = cattr.unstructure(trainer_settings.tqc)
            tqc_full_config = copy.deepcopy(base_settings)
            deep_update_dict(tqc_full_config, tqc_config)
            tqc_full_config["trainer_type"] = "tqc"
            tqc_trainer_settings = cattr.structure(tqc_full_config, TrainerSettings)
            tqc_artifact_path = os.path.join(artifact_path, "tqc")
            tqc_brain_name = f"{behavior_name}_tqc"
            self.tqc_trainer = TQCTrainer(tqc_brain_name, reward_buff_cap, tqc_trainer_settings, training, load, seed, tqc_artifact_path)
            self.trainers.append(self.tqc_trainer)
            
        # POCA Trainer Setup
        if hasattr(trainer_settings, "poca") and trainer_settings.poca is not None:
            poca_config = cattr.unstructure(trainer_settings.poca)
            poca_full_config = copy.deepcopy(base_settings)
            deep_update_dict(poca_full_config, poca_config)
            poca_full_config["trainer_type"] = "poca"
            poca_trainer_settings = cattr.structure(poca_full_config, TrainerSettings)
            poca_artifact_path = os.path.join(artifact_path, "poca")
            poca_brain_name = f"{behavior_name}_poca"
            self.poca_trainer = POCATrainer(poca_brain_name, reward_buff_cap, poca_trainer_settings, training, load, seed, poca_artifact_path)
            self.trainers.append(self.poca_trainer)
            
        # DrQv2 Trainer Setup
        if hasattr(trainer_settings, "drqv2") and trainer_settings.drqv2 is not None:
            drqv2_config = cattr.unstructure(trainer_settings.drqv2)
            drqv2_full_config = copy.deepcopy(base_settings)
            deep_update_dict(drqv2_full_config, drqv2_config)
            drqv2_full_config["trainer_type"] = "drqv2"
            drqv2_trainer_settings = cattr.structure(drqv2_full_config, TrainerSettings)
            drqv2_artifact_path = os.path.join(artifact_path, "drqv2")
            drqv2_brain_name = f"{behavior_name}_drqv2"
            self.drqv2_trainer = DrQv2Trainer(drqv2_brain_name, reward_buff_cap, drqv2_trainer_settings, training, load, seed, drqv2_artifact_path)
            self.trainers.append(self.drqv2_trainer)

        # DCAC Trainer Setup
        if hasattr(trainer_settings, "dcac") and trainer_settings.dcac is not None:
            dcac_config = cattr.unstructure(trainer_settings.dcac)
            dcac_full_config = copy.deepcopy(base_settings)
            deep_update_dict(dcac_full_config, dcac_config)
            dcac_full_config["trainer_type"] = "dcac"
            dcac_trainer_settings = cattr.structure(dcac_full_config, TrainerSettings)
            dcac_artifact_path = os.path.join(artifact_path, "dcac")
            dcac_brain_name = f"{behavior_name}_dcac"
            self.dcac_trainer = DCACTrainer(dcac_brain_name, reward_buff_cap, dcac_trainer_settings, training, load, seed, dcac_artifact_path)
            self.trainers.append(self.dcac_trainer)

        # CrossQ Trainer Setup
        if hasattr(trainer_settings, "crossq") and trainer_settings.crossq is not None:
            crossq_config = cattr.unstructure(trainer_settings.crossq)
            crossq_full_config = copy.deepcopy(base_settings)
            deep_update_dict(crossq_full_config, crossq_config)
            crossq_full_config["trainer_type"] = "crossq"
            crossq_trainer_settings = cattr.structure(crossq_full_config, TrainerSettings)
            crossq_artifact_path = os.path.join(artifact_path, "crossq")
            crossq_brain_name = f"{behavior_name}_crossq"
            self.crossq_trainer = CrossQTrainer(crossq_brain_name, reward_buff_cap, crossq_trainer_settings, training, load, seed, crossq_artifact_path)
            self.trainers.append(self.crossq_trainer)

        # PPO-ET Trainer Setup
        if hasattr(trainer_settings, "ppo_et") and trainer_settings.ppo_et is not None:
            ppo_et_config = cattr.unstructure(trainer_settings.ppo_et)
            ppo_et_full_config = copy.deepcopy(base_settings)
            deep_update_dict(ppo_et_full_config, ppo_et_config)
            ppo_et_full_config["trainer_type"] = "ppo_et"
            ppo_et_trainer_settings = cattr.structure(ppo_et_full_config, TrainerSettings)
            ppo_et_artifact_path = os.path.join(artifact_path, "ppo_et")
            ppo_et_brain_name = f"{behavior_name}_ppo_et"
            self.ppo_et_trainer = PPOETTrainer(ppo_et_brain_name, reward_buff_cap, ppo_et_trainer_settings, training, load, seed, ppo_et_artifact_path)
            self.trainers.append(self.ppo_et_trainer)

        # PPO-CE Trainer Setup
        if hasattr(trainer_settings, "ppo_ce") and trainer_settings.ppo_ce is not None:
            ppo_ce_config = cattr.unstructure(trainer_settings.ppo_ce)
            ppo_ce_full_config = copy.deepcopy(base_settings)
            deep_update_dict(ppo_ce_full_config, ppo_ce_config)
            ppo_ce_full_config["trainer_type"] = "ppo_ce"
            ppo_ce_trainer_settings = cattr.structure(ppo_ce_full_config, TrainerSettings)
            ppo_ce_artifact_path = os.path.join(artifact_path, "ppo_ce")
            ppo_ce_brain_name = f"{behavior_name}_ppo_ce"
            self.ppo_ce_trainer = PPOCETrainer(ppo_ce_brain_name, reward_buff_cap, ppo_ce_trainer_settings, training, load, seed, ppo_ce_artifact_path)
            self.trainers.append(self.ppo_ce_trainer)

        # SAC-AE Trainer Setup
        if hasattr(trainer_settings, "sac_ae") and trainer_settings.sac_ae is not None:
            sac_ae_config = cattr.unstructure(trainer_settings.sac_ae)
            sac_ae_full_config = copy.deepcopy(base_settings)
            deep_update_dict(sac_ae_full_config, sac_ae_config)
            sac_ae_full_config["trainer_type"] = "sac_ae"
            sac_ae_trainer_settings = cattr.structure(sac_ae_full_config, TrainerSettings)
            sac_ae_artifact_path = os.path.join(artifact_path, "sac_ae")
            sac_ae_brain_name = f"{behavior_name}_sac_ae"
            self.sac_ae_trainer = SACAETrainer(sac_ae_brain_name, reward_buff_cap, sac_ae_trainer_settings, training, load, seed, sac_ae_artifact_path)
            self.trainers.append(self.sac_ae_trainer)

    def _is_ready_update(self):
        return any(trainer._is_ready_update() for trainer in self.trainers)

    def _update_policy(self):
        for trainer in self.trainers:
            if trainer._is_ready_update():
                trainer._update_policy()

    def create_optimizer(self):
        # This trainer does not have its own optimizer, it uses the sub-trainers' optimizers.
        pass

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        worker_id_str = trajectory.agent_id.split('-')[0].replace("agent_", "")
        worker_id = int(worker_id_str)
        trainer_index = worker_id % len(self.trainers)
        self.trainers[trainer_index]._process_trajectory(trajectory)
        self._increment_step(len(trajectory.steps), trajectory.behavior_id)

    def create_policy(self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec) -> Policy:
        return AllPolicy(self.seed, behavior_spec, self.trainer_settings, self.trainers, parsed_behavior_id)

    def add_policy(self, parsed_behavior_id: BehaviorIdentifiers, policy: Policy) -> None:
        if not isinstance(policy, AllPolicy):
            raise TypeError("AllTrainer expects an AllPolicy.")

        self.policy = policy
        self.policies[parsed_behavior_id.behavior_id] = policy

        # Add the internal policies to the internal trainers. This will trigger their setup.
        for i, trainer in enumerate(self.trainers):
            trainer.add_policy(parsed_behavior_id, policy.policies[i])

        self._step = policy.get_current_step()

    def get_policy(self, name_behavior_id: str) -> Policy:
        return self.policy

    def save_model(self) -> None:
        for trainer in self.trainers:
            trainer.save_model()

    def end_episode(self) -> None:
        for trainer in self.trainers:
            trainer.end_episode()

    @staticmethod
    def get_trainer_name() -> str:
        return "all"
