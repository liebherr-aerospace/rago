"""Define an F1 score evaluator  of the retrieved context compared to the reference eval sample's context."""

from __future__ import annotations

from pydantic.dataclasses import dataclass

from rago.data_objects import Metric, RetrievalOutput
from rago.eval.base import BaseIndependentEvaluator, EvaluatorReference, register_evaluator

from .precision import RetrievalPrecisionEvaluator
from .relevancy import RetrievalRecallEvaluator


@dataclass
class RetrievalF1Metric(Metric):
    """Contains the evaluation F1 score given to a retrieved context as well as recall and precision.

    :param score: Harmonic mean of relevancy and precision of retrieved context.
    :type score: float
    :param precision: Proportion of retrieved text matching an expected text.
    :type precision: float
    :param recall: Proportion of expected text found in the retrieved context.
    :type recall: float.
    """

    score : float
    precision: float
    recall: float

@register_evaluator("f1")
class RetrievalF1Evaluator(BaseIndependentEvaluator[RetrievalOutput]):
    """F1 score Evaluator of the retrieval output's retrieved context compared to the reference eval sample's context.

    If the eval sample in reference has no context, all metrics are 1.
    If it has context but the retrieved context is empty, all metrics are 0.
    """

    def __init__(self)-> None:
        """Instantiate the F1Evaluator with a relevancy and a precision evaluator."""
        self.precision = RetrievalRecallEvaluator()
        self.recall = RetrievalPrecisionEvaluator()

    def evaluate(self, candidate_output: RetrievalOutput, reference: EvaluatorReference) -> RetrievalF1Metric:
        """Evaluate F1 score of the retrieved context in the candidate retrieval output.

        :param candidate_output: The retrieval output to evaluate.
        :type candidate_output: RetrievalOutput
        :param reference: The reference used by the evaluator to evaluate the candidate
        :type reference: EvaluatorReference
        :return: The F1 metric with F1 score, recall and precision.
        :rtype: RetrievalF1Metric
        """
        recall = self.recall.evaluate(candidate_output, reference).score
        precision = self.precision.evaluate(candidate_output, reference).score
        denominator = recall + precision
        f1 = 0.0 if denominator == 0 else 2 * recall * precision / denominator
        return RetrievalF1Metric(
            score = f1,
            recall = recall,
            precision = precision,
        )

