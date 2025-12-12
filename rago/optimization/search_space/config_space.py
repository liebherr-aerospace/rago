"""Define the config space and relation space."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from pydantic.dataclasses import dataclass

from rago.optimization.search_space.base import SearchSpace
from rago.optimization.search_space.param_space import ParamSpace

if TYPE_CHECKING:
    import optuna

    from rago.model.configs.base import Config


@dataclass
class ConfigSpace(SearchSpace):
    """Define a searchSpace of configuration that contains ParamsSpaces and ConfigSpaces."""

    def __post_init__(self) -> None:
        """Update the config space field."""
        for name, param_or_config_or_el in self.__dict__.items():
            if isinstance(param_or_config_or_el, ParamSpace) and param_or_config_or_el.name is None:
                param_or_config_or_el.name = self.__class__.__name__ + "_" + name

    @abstractmethod
    def sample(self, trial: optuna.trial.BaseTrial) -> Config:
        """Sample a configuration from configuration space.

        :param trial: Trial used to sample the configuration
        :type trial: optuna.trial.BaseTrial
        :return: The sampled configuration.
        :rtype: Config
        """


class RelationSearchSpace(ConfigSpace):
    """A relation space defines a logical link between params and config values."""
