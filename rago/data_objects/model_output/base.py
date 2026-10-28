"""Base Model Ouptut format for Tunable Models."""
from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.data_objects.base import DataObject


@dataclass(kw_only=True)
class ModelOutput(DataObject):
    """Model Output format for Tunable Models."""

    time: float = Field(default = 0)




