"""Base Retriever Model Output format for Retriever Models."""
from pydantic.dataclasses import dataclass

from rago.data_objects.retrieved_context import RetrievedContext

from .base import ModelOutput


@dataclass
class RetrievalOutput(ModelOutput):
    """Base Retriever Model Output format for Retriever Models.

    :param retrieved_context: Context retrieved by the model.
    :type retrieved_context: list[RetrievedContext]
    """

    retrieved_context: list[RetrievedContext]
