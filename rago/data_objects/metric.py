"""Define the metric dataclass."""

from __future__ import annotations

from pydantic.dataclasses import dataclass

from .model_output import ModelOutput


@dataclass
class Metric(ModelOutput):
    """Contains the evaluation score and explanation given to an output according to a particular metric.

    :param score:  The score given to the RAG output.
    :type score: Optional[float]
    :param explanation: The explanation of the score given to the output.
    :type explanation: Optional[str]
    """

    score: float


