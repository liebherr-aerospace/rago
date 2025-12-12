"""Define an LLM config."""

from pydantic.dataclasses import dataclass

from rago.model.configs.base import Config


@dataclass
class LLMConfig(Config):
    """Configuration parameters of an LLM."""
