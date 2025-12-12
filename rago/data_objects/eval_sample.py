"""Define the Data type classes."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.data_objects.base import DataObject
from rago.data_objects.document import Document  # noqa: TC001


@dataclass
class EvalSample(DataObject):
    """As its name hints it corresponds to a sample from the evaluation dataset.

    An EvalSample contains the information necessary to obtain the RAG output and evaluate it.

    :param query: The query that needs to be answered by the RAG.
    It used both to generate the RAG output and to evaluate the RAG output.
    e.g To evaluate the relevance of the RAG output.
    :type query: str
    :param context: The set of texts (str) used by the dataset generator to generate the query.
    It used to evaluate the RAG output (e.g to evaluate the correctness of the answer).
    :type context: Optional[list[str]]
    :param explanations: Eventually given by the generator, it gives further guidance to the judge for evaluation.
    :type explanations: Optional[str]
    :param reference_answer: Correct answer to the query or an answer to compare to.
    :type reference_answer: Optional[str]
    :param reference_score: Score of the reference answer.
    :type reference_score: Optional[float]
    """

    query: str = Field(..., min_length=1, strict=True)
    context: Optional[list[Document]] = Field(default=None, min_length=1)
    explanations: Optional[str] = Field(default=None, min_length=1)
    reference_answer: Optional[str] = Field(default=None, min_length=1)
    reference_score: Optional[float] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
