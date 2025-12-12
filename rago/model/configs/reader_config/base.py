"""Define the base retriever Configuration."""

from pydantic.dataclasses import dataclass

from rago.model.configs.base import Config


@dataclass
class ReaderConfig(Config):
    """Configuration parameters of a Reader."""
