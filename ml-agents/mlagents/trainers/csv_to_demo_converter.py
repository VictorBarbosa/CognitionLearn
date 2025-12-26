"""
Utility functions for converting CSV data to .demo files for ML-Agents.
"""

import csv
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import struct
from enum import IntEnum
from mlagents_envs.base_env import ActionSpec
from mlagents_envs.communicator_objects.demonstration_meta_pb2 import DemonstrationMetaProto
from mlagents_envs.communicator_objects.brain_parameters_pb2 import BrainParametersProto, ActionSpecProto
from mlagents_envs.communicator_objects.agent_info_action_pair_pb2 import AgentInfoActionPairProto
from mlagents_envs.communicator_objects.agent_info_pb2 import AgentInfoProto
from mlagents_envs.communicator_objects.agent_action_pb2 import AgentActionProto
from mlagents_envs.communicator_objects.observation_pb2 import ObservationProto
from google.protobuf.internal.encoder import _EncodeVarint

# Versão da API suportada para demonstrações
SUPPORTED_DEMONSTRATION_VERSION = 1
INITIAL_POS = 33  # Posição inicial para escrita de dados após metadados

# Enums for ObservationType and DimensionProperty (since they might not be available)
class ObservationType(IntEnum):
    DEFAULT = 0
    GOAL_SIGNAL = 1

class DimensionProperty(IntEnum):
    NONE = 0
    TRANSLATIONAL_EQUIVARIANCE = 1
    ROTATIONAL_EQUIVARIANCE = 2


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.
    
    :param csv_path: Path to the CSV file
    :return: DataFrame with the CSV data
    """
    try:
        # Try to load with different encodings
        try:
            data = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            data = pd.read_csv(csv_path, encoding='latin1')
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {str(e)}")


def prepare_observations(data: pd.DataFrame, observation_columns: List[str]) -> np.ndarray:
    """
    Extracts observations from the data.
    
    :param data: DataFrame with the data
    :param observation_columns: List of column names containing observations
    :return: Array of observations (samples, features)
    """
    try:
        observations = data[observation_columns].values.astype(np.float32)
        return observations
    except KeyError as e:
        missing_cols = [col for col in observation_columns if col not in data.columns]
        raise ValueError(f"Observation columns not found in CSV: {missing_cols}")


def prepare_actions(data: pd.DataFrame, action_columns: List[str]) -> np.ndarray:
    """
    Extracts actions from the data.
    
    :param data: DataFrame with the data
    :param action_columns: List of column names containing actions
    :return: Array of actions (samples, features)
    """
    try:
        actions = data[action_columns].values.astype(np.float32)
        return actions
    except KeyError as e:
        missing_cols = [col for col in action_columns if col not in data.columns]
        raise ValueError(f"Action columns not found in CSV: {missing_cols}")


def create_demo_metadata(
    demonstration_name: str,
    number_steps: int,
    number_episodes: int = 1,
    mean_reward: float = 0.0
) -> DemonstrationMetaProto:
    """
    Creates demonstration metadata.
    
    :param demonstration_name: Name of the demonstration
    :param number_steps: Number of steps in the demonstration
    :param number_episodes: Number of episodes
    :param mean_reward: Mean reward
    :return: DemonstrationMetaProto object
    """
    meta = DemonstrationMetaProto()
    meta.api_version = SUPPORTED_DEMONSTRATION_VERSION
    meta.demonstration_name = demonstration_name
    meta.number_steps = number_steps
    meta.number_episodes = number_episodes
    meta.mean_reward = mean_reward
    return meta


def create_brain_parameters(
    brain_name: str,
    action_spec: ActionSpec,
    observation_specs: List = None
) -> BrainParametersProto:
    """
    Creates brain parameters.
    
    :param brain_name: Name of the brain
    :param action_spec: Action specification
    :param observation_specs: List of observation specifications (optional)
    :return: BrainParametersProto object
    """
    brain_params = BrainParametersProto()
    brain_params.brain_name = brain_name
    brain_params.is_training = True
    
    # Create action spec proto
    action_spec_proto = ActionSpecProto()
    action_spec_proto.num_continuous_actions = action_spec.continuous_size
    action_spec_proto.num_discrete_actions = action_spec.discrete_size
    action_spec_proto.discrete_branch_sizes.extend(action_spec.discrete_branches)
    
    brain_params.action_spec.CopyFrom(action_spec_proto)
    return brain_params


def create_observation_proto(
    observation: np.ndarray,
    observation_index: int = 0
) -> ObservationProto:
    """
    Creates an observation proto from numpy array.
    
    :param observation: Observation data as numpy array
    :param observation_index: Index of the observation
    :return: ObservationProto object
    """
    obs_proto = ObservationProto()
    obs_proto.shape.extend(observation.shape if len(observation.shape) > 0 else [1])
    
    # Set observation type to default (0 = DEFAULT)
    obs_proto.observation_type = 0
    
    # Create float data
    float_data = ObservationProto.FloatData()
    flat_obs = observation.flatten() if len(observation.shape) > 0 else np.array([observation])
    float_data.data.extend(flat_obs.tolist())
    obs_proto.float_data.CopyFrom(float_data)
    
    # Set name
    obs_proto.name = f"observation_{observation_index}"
    
    return obs_proto


def create_agent_info_proto(
    observation: np.ndarray,
    reward: float = 0.0,
    done: bool = False,
    agent_id: int = 0
) -> AgentInfoProto:
    """
    Creates agent info proto.
    
    :param observation: Observation data
    :param reward: Reward value
    :param done: Done flag
    :param agent_id: Agent ID
    :return: AgentInfoProto object
    """
    agent_info = AgentInfoProto()
    agent_info.reward = reward
    agent_info.done = done
    agent_info.id = agent_id
    agent_info.max_step_reached = False
    agent_info.group_id = 0
    agent_info.group_reward = 0.0
    
    # Add observation
    obs_proto = create_observation_proto(observation)
    agent_info.observations.append(obs_proto)
    
    return agent_info


def create_agent_action_proto(
    action: np.ndarray,
    action_spec: ActionSpec
) -> AgentActionProto:
    """
    Creates agent action proto.
    
    :param action: Action data
    :param action_spec: Action specification
    :return: AgentActionProto object
    """
    action_proto = AgentActionProto()
    action_proto.value = 0.0  # Default value
    
    # Handle continuous actions
    if action_spec.continuous_size > 0:
        # Take only the continuous part of the action
        continuous_actions = action[:action_spec.continuous_size]
        action_proto.continuous_actions.extend(continuous_actions.tolist())
    
    # Handle discrete actions
    elif action_spec.discrete_size > 0:
        # For discrete actions, we need to convert float values to integers
        # Take only the discrete part of the action
        discrete_part = action[:action_spec.discrete_size]
        discrete_actions = [int(round(a)) for a in discrete_part]
        action_proto.discrete_actions.extend(discrete_actions)
    
    return action_proto


def create_agent_info_action_pair(
    observation: np.ndarray,
    action: np.ndarray,
    action_spec: ActionSpec,
    reward: float = 0.0,
    done: bool = False,
    agent_id: int = 0
) -> AgentInfoActionPairProto:
    """
    Creates an agent info-action pair.
    
    :param observation: Observation data
    :param action: Action data
    :param action_spec: Action specification
    :param reward: Reward value
    :param done: Done flag
    :param agent_id: Agent ID
    :return: AgentInfoActionPairProto object
    """
    pair = AgentInfoActionPairProto()
    
    # Create agent info
    agent_info = create_agent_info_proto(observation, reward, done, agent_id)
    pair.agent_info.CopyFrom(agent_info)
    
    # Create agent action
    agent_action = create_agent_action_proto(action, action_spec)
    pair.action_info.CopyFrom(agent_action)
    
    return pair


def write_delimited(f, message):
    """
    Writes a delimited protobuf message to file.
    
    :param f: File object
    :param message: Protobuf message
    """
    msg_string = message.SerializeToString()
    msg_size = len(msg_string)
    _EncodeVarint(f.write, msg_size)
    f.write(msg_string)


def write_demo_file(
    demo_path: str,
    meta_data_proto: DemonstrationMetaProto,
    brain_param_proto: BrainParametersProto,
    agent_info_protos: List[AgentInfoActionPairProto]
):
    """
    Writes a demonstration file.
    
    :param demo_path: Path to the output .demo file
    :param meta_data_proto: Metadata protobuf
    :param brain_param_proto: Brain parameters protobuf
    :param agent_info_protos: List of agent info-action pairs
    """
    with open(demo_path, "wb") as f:
        # Write metadata
        write_delimited(f, meta_data_proto)
        
        # Seek to initial position for brain parameters
        f.seek(INITIAL_POS)
        
        # Write brain parameters
        write_delimited(f, brain_param_proto)
        
        # Write agent info-action pairs
        for agent in agent_info_protos:
            write_delimited(f, agent)


def convert_csv_to_demo(
    csv_path: str,
    demo_path: str,
    observation_columns: List[str],
    action_columns: List[str],
    action_spec: ActionSpec,
    brain_name: str = "DemoBrain",
    demonstration_name: str = "CSV_Demonstration"
):
    """
    Converts CSV data to a .demo file.
    
    :param csv_path: Path to the input CSV file
    :param demo_path: Path to the output .demo file
    :param observation_columns: List of column names containing observations
    :param action_columns: List of column names containing actions
    :param action_spec: Action specification for the environment
    :param brain_name: Name of the brain
    :param demonstration_name: Name of the demonstration
    """
    # Load CSV data
    data = load_csv_data(csv_path)
    
    # Prepare observations and actions
    observations = prepare_observations(data, observation_columns)
    actions = prepare_actions(data, action_columns)
    
    # Validate data dimensions
    if len(observations) != len(actions):
        raise ValueError(f"Mismatch between number of observations ({len(observations)}) and actions ({len(actions)})")
    
    # Create metadata
    meta_data = create_demo_metadata(
        demonstration_name=demonstration_name,
        number_steps=len(observations),
        number_episodes=1,
        mean_reward=0.0  # Could calculate from data if provided
    )
    
    # Create brain parameters
    # For simplicity, we'll create a basic observation spec
    observation_specs = []  # We don't actually need to specify this for basic conversion
    brain_params = create_brain_parameters(brain_name, action_spec, observation_specs)
    
    # Create agent info-action pairs
    agent_pairs = []
    for i in range(len(observations)):
        obs = observations[i]
        act = actions[i]
        
        # For demonstration purposes, assuming reward=0 and done=False for all steps except last
        reward = 0.0
        done = (i == len(observations) - 1)  # Last step is done
        
        pair = create_agent_info_action_pair(
            observation=obs,
            action=act,
            action_spec=action_spec,
            reward=reward,
            done=done,
            agent_id=0  # Single agent
        )
        agent_pairs.append(pair)
    
    # Write demo file
    write_demo_file(demo_path, meta_data, brain_params, agent_pairs)
    
    print(f"Successfully converted CSV to .demo file: {demo_path}")