"""Define the base TunableModel configuration."""

from pydantic.dataclasses import dataclass

from rago.model.configs.base import Config


@dataclass
class TunableModelConfig(Config):
    """Base configuration for any :class:`TunableModel`."""
