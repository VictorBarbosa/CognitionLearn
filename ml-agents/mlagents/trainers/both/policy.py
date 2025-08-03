from mlagents.trainers.policy import Policy
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.sac.trainer import SACTrainer
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents_envs.base_env import BehaviorSpec, DecisionSteps, ActionTuple
from mlagents.trainers.action_info import ActionInfo
import numpy as np

class BothPolicy(Policy):
    def __init__(self, seed: int, behavior_spec: BehaviorSpec, trainer_settings, ppo_trainer: PPOTrainer, sac_trainer: SACTrainer, parsed_behavior_id: BehaviorIdentifiers):
        super().__init__(seed, behavior_spec, trainer_settings.network_settings)
        self.ppo_policy = ppo_trainer.create_policy(parsed_behavior_id, behavior_spec)
        self.sac_policy = sac_trainer.create_policy(parsed_behavior_id, behavior_spec)

    def get_action(self, decision_requests: DecisionSteps, worker_id: int = 0) -> ActionInfo:
        if worker_id % 2 == 0:
            return self.ppo_policy.get_action(decision_requests, worker_id)
        else:
            return self.sac_policy.get_action(decision_requests, worker_id)

    def increment_step(self, n_steps):
        self.ppo_policy.increment_step(n_steps)
        self.sac_policy.increment_step(n_steps)

    def get_current_step(self):
        return max(self.ppo_policy.get_current_step(), self.sac_policy.get_current_step())

    def load_weights(self, values) -> None:
        self.ppo_policy.load_weights(values)
        self.sac_policy.load_weights(values)

    def get_weights(self):
        return self.ppo_policy.get_weights()

    def init_load_weights(self) -> None:
        self.ppo_policy.init_load_weights()
        self.sac_policy.init_load_weights()
