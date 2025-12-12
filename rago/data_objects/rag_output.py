"""Define the rag output dataclass."""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.data_objects.base import DataObject
from rago.data_objects.retrieved_context import RetrievedContext  # noqa: TC001


@dataclass
class RAGOutput(DataObject):
    """Output of the RAG containing information relevant to a given query (e.g. retrieved documents and answer).

    :param answer: The answer of the RAG.
    :type answer: Optional[str]
    :param retrieved_documents: The documents retrieved by the RAG to generate the answer.
    :type retrieved_documents: Optional[list[Document]]
    """

    answer: Optional[str] = Field(default=None)
    retrieved_context: Optional[list[RetrievedContext]] = Field(default=None)
