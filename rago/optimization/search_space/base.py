"""Defines a search space base abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    import optuna

    from rago.optimization.search_space.elements import Element


@dataclass
class SearchSpace(ABC):
    """Define an Abstract Search that can be sampled to obtain elements."""

    @abstractmethod
    def sample(self, trial: optuna.trial.BaseTrial) -> Element:
        """Sample from the SearchSpace.

        :param trial: Trial used to generate params.
        :type trial: optuna.trial.BaseTrial
        :return: The sample.
        :rtype: dict | str
        """
        ...

    @classmethod
    def from_json(cls, *args: Any, **kwargs: Any) -> SearchSpace:  # noqa: ANN401
        """Instantiate a searchSpace from a json file.

        :return: The instantiated searchSpace.
        :rtype: SearchSpace
        """
        return cls(*args, **kwargs)
