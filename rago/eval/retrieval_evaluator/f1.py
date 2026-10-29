"""Define an evaluator that measures the recall, precision and F1 of the retrieved context."""

from __future__ import annotations

from pydantic.dataclasses import dataclass

from rago.data_objects import Metric, RetrievalOutput
from rago.eval.base import BaseIndependentEvaluator, EvaluatorReference, register_evaluator

from .precision import PrecisionEvaluator
from .relevancy import RelevancyEvaluator


@dataclass
class F1Metric(Metric):
    """Contains the evaluation score and explanation given to an output according to a particular metric."""

    score : float
    precision: float
    relevancy: float

@register_evaluator("f1")
class F1Evaluator(BaseIndependentEvaluator[RetrievalOutput]):
    """Evaluator of the RAG output's retrieved context against the eval sample's context.

    Returns three metrics:
        - ``relevancy`` (recall): proportion of expected chunks found in the retrieved context.
        - ``precision``: proportion of retrieved chunks matching an expected chunk.
        - ``context_f1``: harmonic mean of ``relevancy`` and ``precision``.

    If the eval sample has no context, all metrics are 1.
    If it has context but nothing was retrieved, all metrics are 0.
    """

    def __init__(self)-> None:
        """Instantiate the F1Evaluator with a relevancy and a precision evaluator."""
        self.precision = PrecisionEvaluator()
        self.relevancy = RelevancyEvaluator()

    def evaluate(self, candidate_output: RetrievalOutput, reference: EvaluatorReference) -> F1Metric:
        """Evaluate the retrieved context of the candidate output.

        :param candidate_output: The RAG's output to the query.
        :type candidate_output: RAGOutput
        :param eval_sample: The sample containing the expected context.
        :type eval_sample: EvalSample
        :return: The ``relevancy``, ``precision`` and ``context_f1`` metrics.
        :rtype: dict[str, Metric]
        """
        relevancy = self.relevancy.evaluate(candidate_output, reference).score
        precision = self.precision.evaluate(candidate_output, reference).score
        denominator = relevancy + precision
        f1 = 0.0 if denominator == 0 else 2 * relevancy * precision / denominator
        return F1Metric(
            score = f1,
            relevancy = relevancy,
            precision = precision,
        )

