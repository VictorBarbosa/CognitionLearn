import importlib.metadata as importlib_metadata  # pylint: disable=E0611
from typing import Dict, Tuple, Any

from mlagents import plugins as mla_plugins
from mlagents.plugins import ML_AGENTS_TRAINER_TYPE
from mlagents.trainers.poca.optimizer_torch import POCASettings
from mlagents.trainers.poca.trainer import POCATrainer
from mlagents.trainers.ppo.optimizer_torch import PPOSettings
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.ppo_et.settings import PPOETSettings
from mlagents.trainers.ppo_et.trainer import PPOETTrainer
from mlagents.trainers.ppo_ce.settings import PPOCESettings
from mlagents.trainers.ppo_ce.trainer import PPOCETrainer
from mlagents.trainers.sac.optimizer_torch import SACSettings
from mlagents.trainers.sac.trainer import SACTrainer
from mlagents.trainers.sac_ae.settings import SACAESettings
from mlagents.trainers.sac_ae.trainer import SACAETrainer
from mlagents.trainers.td3.optimizer_torch import TD3Settings
from mlagents.trainers.td3.trainer import TD3Trainer
from mlagents.trainers.tdsac.optimizer_torch import TDSACSettings
from mlagents.trainers.tdsac.trainer import TDSACTrainer
from mlagents.trainers.tqc.optimizer_torch import TQCSettings
from mlagents.trainers.tqc.trainer import TQCTrainer
from mlagents.trainers.drqv2.optimizer_torch import DrQv2Settings
from mlagents.trainers.drqv2.trainer import DrQv2Trainer
from mlagents.trainers.dcac.optimizer_torch import DCACSettings
from mlagents.trainers.dcac.trainer import DCACTrainer
from mlagents.trainers.crossq.optimizer_torch import CrossQSettings
from mlagents.trainers.crossq.trainer import CrossQTrainer
from mlagents.trainers.all.trainer import AllTrainer
from mlagents.trainers.settings import TrainerSettings, HyperparamSettings
from mlagents_envs import logging_util

logger = logging_util.get_logger(__name__)


def get_default_trainer_types() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    The Trainers that mlagents-learn always uses:
    """

    mla_plugins.all_trainer_types.update(
        {
            PPOTrainer.get_trainer_name(): PPOTrainer,
            PPOETTrainer.get_trainer_name(): PPOETTrainer,
            PPOCETrainer.get_trainer_name(): PPOCETrainer,
            SACTrainer.get_trainer_name(): SACTrainer,
            SACAETrainer.get_trainer_name(): SACAETrainer,
            POCATrainer.get_trainer_name(): POCATrainer,
            TD3Trainer.get_trainer_name(): TD3Trainer,
            TDSACTrainer.get_trainer_name(): TDSACTrainer,
            TQCTrainer.get_trainer_name(): TQCTrainer,
            DrQv2Trainer.get_trainer_name(): DrQv2Trainer,
            DCACTrainer.get_trainer_name(): DCACTrainer,
            CrossQTrainer.get_trainer_name(): CrossQTrainer,
            AllTrainer.get_trainer_name(): AllTrainer,
        }
    )
    # global all_trainer_settings
    mla_plugins.all_trainer_settings.update(
        {
            PPOTrainer.get_trainer_name(): PPOSettings,
            PPOETTrainer.get_trainer_name(): PPOETSettings,
            PPOCETrainer.get_trainer_name(): PPOCESettings,
            SACTrainer.get_trainer_name(): SACSettings,
            SACAETrainer.get_trainer_name(): SACAESettings,
            POCATrainer.get_trainer_name(): POCASettings,
            TD3Trainer.get_trainer_name(): TD3Settings,
            TDSACTrainer.get_trainer_name(): TDSACSettings,
            TQCTrainer.get_trainer_name(): TQCSettings,
            DrQv2Trainer.get_trainer_name(): DrQv2Settings,
            DCACTrainer.get_trainer_name(): DCACSettings,
            CrossQTrainer.get_trainer_name(): CrossQSettings,
            AllTrainer.get_trainer_name(): HyperparamSettings, # Using HyperparamSettings as a placeholder for AllSettings
        }
    )

    return mla_plugins.all_trainer_types, mla_plugins.all_trainer_settings


def register_trainer_plugins() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Registers all Trainer plugins (including the default one),
    and evaluates them, and returns the list of all the Trainer implementations.
    """
    if ML_AGENTS_TRAINER_TYPE not in importlib_metadata.entry_points():
        logger.warning(
            f"Unable to find any entry points for {ML_AGENTS_TRAINER_TYPE}, even the default ones. "
            "Uninstalling and reinstalling ml-agents via pip should resolve. "
            "Using default plugins for now."
        )
        return get_default_trainer_types()

    entry_points = importlib_metadata.entry_points()[ML_AGENTS_TRAINER_TYPE]

    for entry_point in entry_points:

        try:
            logger.debug(f"Initializing Trainer plugins: {entry_point.name}")
            plugin_func = entry_point.load()
            plugin_trainer_types, plugin_trainer_settings = plugin_func()
            logger.debug(
                f"Found {len(plugin_trainer_types)} Trainers for plugin {entry_point.name}"
            )
            mla_plugins.all_trainer_types.update(plugin_trainer_types)
            mla_plugins.all_trainer_settings.update(plugin_trainer_settings)
        except BaseException:
            # Catch all exceptions from setting up the plugin, so that bad user code doesn't break things.
            logger.exception(
                f"Error initializing Trainer plugins for {entry_point.name}. This plugin will not be used."
            )
    return mla_plugins.all_trainer_types, mla_plugins.all_trainer_settings
