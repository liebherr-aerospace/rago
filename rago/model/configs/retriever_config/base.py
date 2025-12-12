"""Defines the base Retriever Config."""

from pydantic.dataclasses import dataclass

from rago.model.configs.base import Config


@dataclass
class RetrieverConfig(Config):
    """Configuration parameters of the retriever."""
