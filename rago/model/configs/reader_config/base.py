"""Define the base retriever Configuration."""

from pydantic.dataclasses import dataclass

from rago.model.configs.tunable_model_config import TunableModelConfig


@dataclass
class ReaderConfig(TunableModelConfig):
    """Configuration parameters of a Reader."""
