"""Base Retriever Model Output format for Retriever Models."""
from pydantic.dataclasses import dataclass

from rago.data_objects.retrieved_context import RetrievedContext

from .base import ModelOutput


@dataclass
class RetrievalOutput(ModelOutput):
    """Base Retriever Model Output format for Retriever Models."""

    retrieved_context: list[RetrievedContext]
