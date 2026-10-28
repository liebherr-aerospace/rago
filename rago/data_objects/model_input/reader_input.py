"""Reader Input format."""
from pydantic.dataclasses import dataclass

from rago.data_objects import RetrievedContext

from .base import ModelInput


@dataclass
class ReaderInput(ModelInput):
    """Reader Input format."""

    query: str
    retrieved_context: list[RetrievedContext]
