import os
import cattr
import copy
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.sac.trainer import SACTrainer
from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.policy import Policy
from mlagents.trainers.both.policy import BothPolicy
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.settings import TrainerSettings, deep_update_dict

class BothTrainer(RLTrainer):
    def __init__(self, behavior_name, reward_buff_cap, trainer_settings, training, load, seed, artifact_path):
        super().__init__(behavior_name, trainer_settings, training, load, artifact_path, reward_buff_cap)
        self.seed = seed
        
        base_settings = trainer_settings.as_dict()
        base_settings.pop("ppo", None)
        base_settings.pop("sac", None)
        base_settings.pop("trainer_type", None)

        # PPO Trainer Setup
        ppo_config = trainer_settings.ppo
        ppo_full_config = copy.deepcopy(base_settings)
        deep_update_dict(ppo_full_config, ppo_config)
        ppo_full_config["trainer_type"] = "ppo"
        ppo_trainer_settings = cattr.structure(ppo_full_config, TrainerSettings)
        ppo_artifact_path = os.path.join(artifact_path, "ppo")
        ppo_brain_name = f"{behavior_name}_ppo"
        self.ppo_trainer = PPOTrainer(ppo_brain_name, reward_buff_cap, ppo_trainer_settings, training, load, seed, ppo_artifact_path)

        # SAC Trainer Setup
        sac_config = trainer_settings.sac
        sac_full_config = copy.deepcopy(base_settings)
        deep_update_dict(sac_full_config, sac_config)
        sac_full_config["trainer_type"] = "sac"
        sac_trainer_settings = cattr.structure(sac_full_config, TrainerSettings)
        sac_artifact_path = os.path.join(artifact_path, "sac")
        sac_brain_name = f"{behavior_name}_sac"
        self.sac_trainer = SACTrainer(sac_brain_name, reward_buff_cap, sac_trainer_settings, training, load, seed, sac_artifact_path)

    def _is_ready_update(self):
        return self.ppo_trainer._is_ready_update() or self.sac_trainer._is_ready_update()

    def _update_policy(self):
        if self.ppo_trainer._is_ready_update():
            self.ppo_trainer._update_policy()
        if self.sac_trainer._is_ready_update():
            self.sac_trainer._update_policy()

    def create_optimizer(self):
        # This trainer does not have its own optimizer, it uses the sub-trainers' optimizers.
        pass

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        worker_id_str = trajectory.agent_id.split('-')[0].replace("agent_", "")
        worker_id = int(worker_id_str)
        if worker_id % 2 == 0:
            self.ppo_trainer._process_trajectory(trajectory)
        else:
            self.sac_trainer._process_trajectory(trajectory)
        self._increment_step(len(trajectory.steps), trajectory.behavior_id)

    def create_policy(self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec) -> Policy:
        return BothPolicy(self.seed, behavior_spec, self.trainer_settings, self.ppo_trainer, self.sac_trainer, parsed_behavior_id)

    def add_policy(self, parsed_behavior_id: BehaviorIdentifiers, policy: Policy) -> None:
        if not isinstance(policy, BothPolicy):
            raise TypeError("BothTrainer expects a BothPolicy.")

        self.policy = policy
        self.policies[parsed_behavior_id.behavior_id] = policy

        # Add the internal policies to the internal trainers. This will trigger their setup.
        self.ppo_trainer.add_policy(parsed_behavior_id, policy.ppo_policy)
        self.sac_trainer.add_policy(parsed_behavior_id, policy.sac_policy)

        self._step = policy.get_current_step()

    def get_policy(self, name_behavior_id: str) -> Policy:
        return self.policy

    def save_model(self) -> None:
        self.ppo_trainer.save_model()
        self.sac_trainer.save_model()

    def end_episode(self) -> None:
        self.ppo_trainer.end_episode()
        self.sac_trainer.end_episode()

    @staticmethod
    def get_trainer_name() -> str:
        return "both"
