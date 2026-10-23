"""Define the abstract TunableModel config space."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from pydantic.dataclasses import dataclass

from rago.optimization.search_space.config_space import ConfigSpace

if TYPE_CHECKING:
    import optuna

    from rago.model.configs.tunable_model_config import TunableModelConfig


@dataclass
class TunableModelConfigSpace(ConfigSpace):
    """Abstract config space that samples a :class:`TunableModelConfig`."""

    @abstractmethod
    def sample(self, trial: optuna.trial.BaseTrial) -> TunableModelConfig:
        """Sample a model configuration from the search space.

        :param trial: Optuna trial used to sample hyperparameters.
        :type trial: optuna.trial.BaseTrial
        :return: A concrete model configuration.
        :rtype: TunableModelConfig
        """
        ...
