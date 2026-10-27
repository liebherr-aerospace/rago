"""Define the abstract TunableModel config space."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from pydantic.dataclasses import dataclass

from rago.model.configs.tunable_model_config import TunableModelConfig
from rago.optimization.search_space.config_space import ConfigSpace

if TYPE_CHECKING:
    import optuna


@dataclass
class TunableModelConfigSpace[ConfigT: TunableModelConfig](ConfigSpace):
    """Abstract config space that samples a :class:`TunableModelConfig` of type ``ConfigT``."""

    @abstractmethod
    def sample(self, trial: optuna.trial.BaseTrial) -> ConfigT:
        """Sample a model configuration from the search space.

        :param trial: Optuna trial used to sample hyperparameters.
        :type trial: optuna.trial.BaseTrial
        :return: A concrete model configuration.
        :rtype: ConfigT
        """
        ...
