"""Define an evaluator measuring the precision of a retrieval output's retrieved context."""

from __future__ import annotations

from rago.data_objects import Metric, RetrievalOutput
from rago.eval.base import EvaluatorReference, register_evaluator

from .base import RuleBasedRetrievalEvaluator


@register_evaluator("precision")
class RetrievalPrecisionEvaluator(RuleBasedRetrievalEvaluator):
    """Evaluator of the retrieval output's retrieved context precision to the reference eval sample's context."""

    def evaluate(self, candidate_output: RetrievalOutput, reference: EvaluatorReference) -> Metric:
        """Evaluate the precision of the retrieved context in the candidate retrieval output.

        :param candidate_output: The retrieval output to evaluate.
        :type candidate_output: RetrievalOutput
        :param reference: The reference used by the evaluator to evaluate the candidate.
        :type reference: EvaluatorReference
        :return: The precision metric.
        :rtype: Metric
        """
        if not reference.eval_sample.context:
            return Metric(1.0)
        if not candidate_output.retrieved_context:
            return Metric(0.0)
        nb_correct = self.get_num_correct_retrieved_context(
            retrieved_context = candidate_output.retrieved_context,
            reference_contexts = reference.eval_sample.context,
            )
        return Metric(nb_correct / len(candidate_output.retrieved_context))
