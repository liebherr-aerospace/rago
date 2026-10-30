"""Define the rag output dataclass."""

from __future__ import annotations

from pydantic.dataclasses import dataclass

from .generation_output import GenerationOutput
from .retrieval_output import RetrievalOutput


@dataclass
class RAGOutput(GenerationOutput, RetrievalOutput):
    """Output of the RAG containing information relevant to a given query (e.g. retrieved documents and answer).

    :param answer: The answer of the RAG.
    :type answer: Optional[str]
    :param retrieved_documents: The documents retrieved by the RAG to generate the answer.
    :type retrieved_documents: Optional[list[Document]]
    """
