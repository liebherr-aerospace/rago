"""Defines the base Retriever Config."""

from pydantic.dataclasses import dataclass

from rago.model.configs.tunable_model_config import TunableModelConfig


@dataclass
class RetrieverConfig(TunableModelConfig):
    """Configuration parameters of the retriever."""
