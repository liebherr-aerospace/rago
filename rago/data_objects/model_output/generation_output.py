"""Base Generation Model Ouptut format for Generation Tunable Models."""
from pydantic.dataclasses import dataclass

from .base import ModelOutput


@dataclass
class GenerationOutput(ModelOutput):
    """Generation Model Output format for Generation Tunable Models.

    :param answer: The generated answer.
    :type answer: Optional[str]
    """

    answer: str
