"""Base Generation Model Ouptut format for Generation Tunable Models."""
from pydantic.dataclasses import dataclass

from .base import ModelOutput


@dataclass
class GenerationOutput(ModelOutput):
    """Base Generation Model Ouptut format for Generation Tunable Models."""

    answer: str
