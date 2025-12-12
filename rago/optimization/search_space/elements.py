"""Defines the different type of a search space."""

from optuna.distributions import CategoricalChoiceType

from rago.model.configs.base import Config

type ParamSpaceElement = CategoricalChoiceType
type ConfigSpaceElement = Config
type Element = ConfigSpaceElement | ParamSpaceElement
