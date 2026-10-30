"""Define an evaluator measuring the precision of a retrieval output's retrieved context."""

from __future__ import annotations

from rago.data_objects import Metric, RetrievalOutput
from rago.eval.base import BaseIndependentEvaluator, EvaluatorReference, register_evaluator


@register_evaluator("relevancy")
class RetrievalRecallEvaluator(BaseIndependentEvaluator[RetrievalOutput]):
    """Evaluator of the retrieval output's retrieved context recall to the reference eval sample's context."""

    def evaluate(self, candidate_output: RetrievalOutput, reference: EvaluatorReference) -> Metric:
        """Evaluate the proportion of the expected context contained in the candidate retrieved context.

        :param candidate_output: The retrieval output to evaluate.
        :type candidate_output: RetrievalOutput
        :param reference: The reference used by the evaluator to evaluate the candidate.
        :type reference: EvaluatorReference
        :return: The recall metric..
        :rtype: Metric
        """
        if not reference.eval_sample.context:
            return Metric(1.0)
        if not candidate_output.retrieved_context:
            return Metric(0.0)
        retrieved_text = "\n".join(doc.text for doc in candidate_output.retrieved_context)
        nb_found = sum(ctx.text in retrieved_text for ctx in reference.eval_sample.context)
        return Metric(nb_found / len(reference.eval_sample.context))
