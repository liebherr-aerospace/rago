"""Define the metric dataclass."""

from __future__ import annotations

from pydantic.dataclasses import dataclass

from .model_output import ModelOutput


@dataclass
class Metric(ModelOutput):
    """Contains the evaluation score and explanation given to an output according to a particular metric.

    :param score:  The score given to the model output.
    :type score: Optional[float]
    """

    score: float


