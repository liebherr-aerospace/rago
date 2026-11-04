"""Define an evaluator measuring the precision of a retrieval output's retrieved context."""

from __future__ import annotations

from rago.data_objects import Metric, RetrievalOutput
from rago.eval.base import BaseIndependentEvaluator, EvaluatorReference, register_evaluator


@register_evaluator("precision")
class RetrievalPrecisionEvaluator(BaseIndependentEvaluator[RetrievalOutput]):
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
        expected_texts = [ctx.text for ctx in reference.eval_sample.context]
        nb_relevant = sum(
            any(expected in doc.text or doc.text in expected for expected in expected_texts)
            for doc in candidate_output.retrieved_context
        )
        return Metric(nb_relevant / len(candidate_output.retrieved_context))
