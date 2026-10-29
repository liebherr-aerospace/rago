"""Deprecated evaluator measuring only the precision of the retrieved context."""

from __future__ import annotations

import warnings

from rago.data_objects import Metric, RetrievalOutput
from rago.eval.base import BaseIndependentEvaluator, EvaluatorReference, register_evaluator


@register_evaluator("precision")
class PrecisionEvaluator(BaseIndependentEvaluator[RetrievalOutput]):
    """Deprecated: use :class:`ContextRetrievalEvaluator`, which also returns relevancy and context_f1."""

    def __init__(self) -> None:
        """Warn that this evaluator is deprecated."""
        warnings.warn(
            "PrecisionEvaluator is deprecated, use ContextRetrievalEvaluator instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def evaluate(self, candidate_output: RetrievalOutput, reference: EvaluatorReference) -> Metric:
        """Evaluate the precision of the retrieved context.

        :param candidate_output: The RAG's output to the query.
        :type candidate_output: RAGOutput
        :param eval_sample: The sample containing the expected context.
        :type eval_sample: EvalSample
        :return: The evaluation result with a ``"precision"`` key.
        :rtype: dict[str, Metric]
        """
        if not reference.eval_sample.context:
            return Metric(1.0)
        if not candidate_output.retrieved_context:
            return Metric(1.0)
        retrieved_text = "\n".join(doc.text for doc in candidate_output.retrieved_context)
        nb_found = sum(ctx.text in retrieved_text for ctx in reference.eval_sample.context)
        return Metric(nb_found / len(reference.eval_sample.context))
