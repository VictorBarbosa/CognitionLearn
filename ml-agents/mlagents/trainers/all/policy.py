
from mlagents.trainers.policy import Policy
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.sac.trainer import SACTrainer
from mlagents.trainers.td3.trainer import TD3Trainer
from mlagents.trainers.tdsac.trainer import TDSACTrainer
from mlagents.trainers.masac.trainer import MASACTrainer
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents_envs.base_env import BehaviorSpec, DecisionSteps, ActionTuple
from mlagents.trainers.action_info import ActionInfo
import numpy as np

class AllPolicy(Policy):
    def __init__(self, seed: int, behavior_spec: BehaviorSpec, trainer_settings, trainers, parsed_behavior_id: BehaviorIdentifiers):
        super().__init__(seed, behavior_spec, trainer_settings.network_settings)
        self.policies = []
        for trainer in trainers:
            self.policies.append(trainer.create_policy(parsed_behavior_id, behavior_spec))

    def get_action(self, decision_requests: DecisionSteps, worker_id: int = 0) -> ActionInfo:
        trainer_index = worker_id % len(self.policies)
        return self.policies[trainer_index].get_action(decision_requests, worker_id)

    def increment_step(self, n_steps):
        for policy in self.policies:
            policy.increment_step(n_steps)

    def get_current_step(self):
        return max(policy.get_current_step() for policy in self.policies)

    def load_weights(self, values) -> None:
        for policy in self.policies:
            policy.load_weights(values)

    def get_weights(self):
        return self.policies[0].get_weights()

    def init_load_weights(self) -> None:
        for policy in self.policies:
            policy.init_load_weights()
