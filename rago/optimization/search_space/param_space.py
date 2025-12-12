"""Defines the Param space and the parma types."""

from __future__ import annotations

from abc import abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Generic, Optional, TypeVar

from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.optimization.search_space.base import SearchSpace

if TYPE_CHECKING:
    import optuna

    from rago.optimization.search_space.elements import ParamSpaceElement

categorical_type = None | bool | int | float | str

T = TypeVar("T", bound=categorical_type)


class ParamType(StrEnum):
    """The available param types."""

    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"


@dataclass
class ParamSpace(SearchSpace):
    """A space containing all the possible parameters."""

    name: Optional[str] = Field(default=None)

    @abstractmethod
    def sample(self, trial: optuna.trial.BaseTrial) -> ParamSpaceElement:
        """Sample a parameter from the parameter space using trial."""
        ...


@dataclass
class IntParamSpace(ParamSpace):
    """A space containing all the possible int parameters."""

    low: int = Field(default=0)
    high: int = Field(default=0)
    step: int = Field(default=1)
    log: bool = Field(default=False)

    def sample(self, trial: optuna.trial.BaseTrial) -> int:
        """Sample a parameter from the parameter space using trial."""
        if self.name is None:
            raise ValueError(self.name)
        return trial.suggest_int(self.name, low=self.low, high=self.high, step=self.step, log=self.log)


@dataclass
class FloatParamSpace(ParamSpace):
    """A space containing all the possible float parameters."""

    low: float = Field(default=0.0)
    high: float = Field(default=0.0)
    step: Optional[float] = Field(default=None)
    log: bool = Field(default=False)

    def sample(self, trial: optuna.trial.BaseTrial) -> float:
        """Sample a parameter from the parameter space using trial."""
        if self.name is None:
            raise ValueError(self.name)
        return trial.suggest_float(self.name, low=self.low, high=self.high, step=self.step, log=self.log)


@dataclass
class CategoricalParamSpace(ParamSpace, Generic[T]):
    """A space containing all the possible categorical parameters."""

    choices: list[T] = Field(default_factory=list)

    def sample(self, trial: optuna.trial.BaseTrial) -> T:
        """Sample a parameter from the parameter space using trial."""
        if self.name is None:
            raise ValueError(self.name)
        sampled_param = trial.suggest_categorical(self.name, choices=self.choices)
        if isinstance(sampled_param, type(self.choices[0])):
            return sampled_param
        raise ValueError(sampled_param)
