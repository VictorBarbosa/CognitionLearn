
from typing import NamedTuple, Any, Dict, List
import numpy as np
from mlagents_envs.base_env import AgentId, ActionTuple
from mlagents.trainers.torch_entities.action_log_probs import LogProbsTuple


ActionInfoOutputs = Dict[str, np.ndarray]


class ActionInfo(NamedTuple):
    """
    A NamedTuple containing actions and related quantities to the policy forward
    pass. Additionally contains the agent ids in the corresponding DecisionStep
    :param action: The action output of the policy
    :param env_action: The possibly clipped action to be executed in the environment
    :param outputs: Dict of all quantities associated with the policy forward pass
    :param agent_ids: List of int agent ids in DecisionStep
    """

    action: ActionTuple
    env_action: ActionTuple
    outputs: ActionInfoOutputs
    agent_ids: List[AgentId]

    @staticmethod
    def empty() -> "ActionInfo":
        return ActionInfo(ActionTuple(), ActionTuple(), {}, [])

    @staticmethod
    def merge(action_infos: list["ActionInfo"]) -> "ActionInfo":
        """Merge a list of ActionInfo objects into a single one."""
        non_empty_infos = [info for info in action_infos if len(info.agent_ids) > 0]
        if not non_empty_infos:
            return ActionInfo.empty()

        if len(non_empty_infos) == 1:
            return non_empty_infos[0]

        # Merge ActionTuples for 'action'
        all_continuous_actions = [info.action.continuous for info in non_empty_infos]
        all_discrete_actions = [info.action.discrete for info in non_empty_infos]
        merged_continuous_actions = np.concatenate(all_continuous_actions)
        merged_discrete_actions = np.concatenate(all_discrete_actions)
        merged_action = ActionTuple(continuous=merged_continuous_actions, discrete=merged_discrete_actions)

        # Merge ActionTuples for 'env_action'
        all_continuous_env_actions = [info.env_action.continuous for info in non_empty_infos]
        all_discrete_env_actions = [info.env_action.discrete for info in non_empty_infos]
        merged_continuous_env_actions = np.concatenate(all_continuous_env_actions)
        merged_discrete_env_actions = np.concatenate(all_discrete_env_actions)
        merged_env_action = ActionTuple(continuous=merged_continuous_env_actions, discrete=merged_discrete_env_actions)

        # Merge 'outputs' dictionary
        merged_outputs = {}
        first_outputs = non_empty_infos[0].outputs
        for key in first_outputs:
            # Skip if the output is empty for the first info object
            if not hasattr(first_outputs[key], '__len__') or len(first_outputs[key]) == 0:
                continue

            if isinstance(first_outputs[key], LogProbsTuple):
                all_cont_log_probs = [info.outputs[key].continuous for info in non_empty_infos]
                all_disc_log_probs = [info.outputs[key].discrete for info in non_empty_infos]
                merged_cont_log_probs = np.concatenate(all_cont_log_probs)
                merged_disc_log_probs = np.concatenate(all_disc_log_probs)
                merged_outputs[key] = LogProbsTuple(continuous=merged_cont_log_probs, discrete=merged_disc_log_probs)
            elif isinstance(first_outputs[key], np.ndarray):
                merged_outputs[key] = np.concatenate([info.outputs[key] for info in non_empty_infos])

        # Merge 'agent_ids'
        merged_agent_ids = np.concatenate([np.array(info.agent_ids) for info in non_empty_infos]).tolist()

        return ActionInfo(
            action=merged_action,
            env_action=merged_env_action,
            outputs=merged_outputs,
            agent_ids=merged_agent_ids,
        )
