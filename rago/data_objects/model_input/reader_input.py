"""Reader Input format."""
from pydantic.dataclasses import dataclass

from rago.data_objects import RetrievedContext

from .base import ModelInput


@dataclass
class ReaderInput(ModelInput):
    """Reader Input format.

    :param query: The input query.
    :type query: str
    :param retrieved_context: The retrieved context the model can use to answer the query.
    :type retrieved_context: list[RetrievedContext]
    """

    query: str
    retrieved_context: list[RetrievedContext]
