"""
Settings for supervised training in ML-Agents.
"""

import attr
from typing import List, Optional, Dict, Any
from mlagents.trainers.settings import (
    RunOptions,
    TrainerSettings,
    NetworkSettings,
    CheckpointSettings,
    SelfPlaySettings,
    BehavioralCloningSettings,
    RewardSignalSettings,
    RewardSignalType,
    ScheduleType,
    EncoderType,
    ConditioningType
)


@attr.s(auto_attribs=True)
class SupervisedLearningSettings:
    """
    Settings for supervised training.
    """
    # Path to the CSV file with training data
    csv_path: str = ""
    
    # Observation columns in the CSV
    observation_columns: List[str] = attr.ib(factory=list)
    
    # Action columns in the CSV
    action_columns: List[str] = attr.ib(factory=list)
    
    # Number of training epochs
    num_epoch: int = 100
    
    # Batch size
    batch_size: int = 128
    
    # Learning rate
    learning_rate: float = 3e-4
    
    # Epoch interval to save checkpoints
    checkpoint_interval: int = 1000
    
    # Fraction of data for validation
    validation_split: float = 0.2
    
    # Whether the data should be shuffled
    shuffle: bool = True
    
    # Noise level for data augmentation
    augment_noise: float = 0.01
    
    # Whether to use early stopping
    early_stopping: bool = True
    
    # Patience for early stopping
    patience: int = 10
    
    # Minimum improvement for early stopping
    min_delta: float = 0.001
    
    # Path to initialize weights (optional)
    init_path: Optional[str] = None
    
    # Dropout rate for regularization
    dropout_rate: float = 0.1
    
    # Weight decay for L2 regularization
    weight_decay: float = 1e-4
    
    # Patience for learning rate reduction
    lr_patience: int = 5
    
    # Shape of observations
    observation_shape: List[int] = attr.ib(factory=list)
    
    # Size of actions
    action_size: int = 0
    
    # Type of actions (continuous or discrete)
    action_type: str = "continuous"
    
    # Whether to export to ONNX
    export_onnx: bool = True
    
    # Whether to export to .pt
    export_pt: bool = True
    
    # Device for training (cuda or cpu)
    device: str = "cuda"
    
    # Seed for reproducibility
    seed: int = 42
    
    # Whether to show detailed information
    verbose: bool = True


# Default configuration for supervised training
DEFAULT_SUPERVISED_SETTINGS = SupervisedLearningSettings()


def create_supervised_run_options(
    behavior_name: str,
    csv_path: str,
    observation_columns: List[str],
    action_columns: List[str],
    output_path: str = "./results",
    num_epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    validation_split: float = 0.2,
    shuffle: bool = True,
    augment_noise: float = 0.01,
    early_stopping: bool = True,
    patience: int = 10,
    min_delta: float = 0.001,
    dropout_rate: float = 0.1,
    weight_decay: float = 1e-4,
    lr_patience: int = 5,
    observation_shape: List[int] = None,
    action_size: int = 0,
    action_type: str = "continuous",
    export_onnx: bool = True,
    export_pt: bool = True,
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = True
) -> RunOptions:
    """
    Creates run options for supervised training.
    
    :param behavior_name: Behavior name
    :param csv_path: Path to the CSV file
    :param observation_columns: Observation columns
    :param action_columns: Action columns
    :param output_path: Output path
    :param num_epochs: Number of epochs
    :param batch_size: Batch size
    :param learning_rate: Learning rate
    :param validation_split: Fraction for validation
    :param shuffle: Whether to shuffle data
    :param augment_noise: Augmentation noise
    :param early_stopping: Whether to use early stopping
    :param patience: Patience for early stopping
    :param min_delta: Minimum improvement
    :param dropout_rate: Dropout rate
    :param weight_decay: Weight decay
    :param lr_patience: Patience for LR
    :param observation_shape: Shape of observations
    :param action_size: Size of actions
    :param action_type: Type of actions
    :param export_onnx: Whether to export ONNX
    :param export_pt: Whether to export .pt
    :param device: Device
    :param seed: Seed
    :param verbose: Verbosity
    :return: Run options
    """
    # Define default shape of observations if not provided
    if observation_shape is None:
        observation_shape = [len(observation_columns)]
    
    # Create supervised learning settings
    supervised_settings = SupervisedLearningSettings(
        csv_path=csv_path,
        observation_columns=observation_columns,
        action_columns=action_columns,
        num_epoch=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_split=validation_split,
        shuffle=shuffle,
        augment_noise=augment_noise,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        lr_patience=lr_patience,
        observation_shape=observation_shape,
        action_size=action_size,
        action_type=action_type,
        export_onnx=export_onnx,
        export_pt=export_pt,
        device=device,
        seed=seed,
        verbose=verbose
    )
    
    # Create network settings
    network_settings = NetworkSettings(
        normalize=True,
        hidden_units=256,
        num_layers=2,
        encoder_type=EncoderType.FULLY_CONNECTED,
        conditioning_type=ConditioningType.HYPER,
        deterministic=False
    )
    
    # Create checkpoint settings
    checkpoint_settings = CheckpointSettings(
        run_id=output_path,
        checkpoint_interval=1000,
        keep_checkpoints=5,
        load=False,
        resume=False
    )
    
    # Create trainer settings
    trainer_settings = TrainerSettings(
        trainer_type="ppo",
        hyperparameters=None,
        network_settings=network_settings,
        checkpoint_interval=1000,
        max_steps=500000,
        time_horizon=1000,
        summary_freq=10000,
        threaded=False,
        self_play=None,
        behavioral_cloning=None,
        reward_signals={}
    )
    
    # Add supervised training settings to the trainer
    trainer_settings.supervised = supervised_settings
    
    # Create run options
    run_options = RunOptions(
        default_settings=None,
        behaviors={behavior_name: trainer_settings},
        env_settings=None,
        engine_settings=None,
        environment_parameters=None,
        checkpoint_settings=checkpoint_settings,
        torch_settings=None,
        debug=False
    )
    
    return run_options
