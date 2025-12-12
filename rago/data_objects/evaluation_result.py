"""Define the evaluation result dataclass."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from pydantic import Field
from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    from rago.data_objects.eval_sample import EvalSample
    from rago.data_objects.metric import Metric
    from rago.data_objects.rag_output import RAGOutput


@dataclass
class EvaluationResult:
    """Contains all the information that characterize an evaluation (input and output).

    :param config: The RAG configuration evaluated.
    :type config: dict[str, Union[float, str]]
    :param eval_sample: The input sample to evaluate on.
    :type eval_sample: Optional[EvalSample]
    :param rag_output: The output generated with the RAG config.
    :type rag_output: Optional[RagOutput]
    :param metrics: The metrics resulting from the evaluation
    :type metrics: Optional[dict[str, Metric]]
    """

    config: dict[str, Union[float, str]] = Field(...)
    eval_sample: Optional[EvalSample] = Field(default=None)
    rag_output: Optional[RAGOutput] = Field(default=None)
    metrics: Optional[dict[str, Metric]] = Field(default=None, min_length=1)
