"""Contains the data generate for a candidate trial."""
from pydantic.dataclasses import dataclass

from .metric import Metric
from .model_output import ModelOutput


@dataclass
class Candidate[ModelOutputT: ModelOutput]:
    """Contains the data generate by model and evaluator for a candidate trial."""

    output: ModelOutputT
    metric: Metric
