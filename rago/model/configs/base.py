"""Define the base configuration parameters class."""

from pydantic.dataclasses import dataclass

from rago.data_objects import DataObject


@dataclass
class Config(DataObject):
    """Configuration Parameters."""
