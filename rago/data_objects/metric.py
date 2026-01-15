"""Define the metric dataclass."""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic.dataclasses import dataclass
from .base import DataObject

@dataclass
class Metric(DataObject):
    """Contains the evaluation score and explanation given to an output according to a particular metric.

    :param score:  The score given to the RAG output.
    :type score: Optional[float]
    :param explanation: The explanation of the score given to the output.
    :type explanation: Optional[str]
    """

    score: float  = Field(default = 0)

    explanation: Optional[str] = Field(default=None)
