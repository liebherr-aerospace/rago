"""Deprecated evaluator measuring only the relevancy (recall) of the retrieved context."""

from __future__ import annotations

import warnings

from rago.data_objects import EvalSample, Metric, RetrievalOutput
from rago.eval.base import BaseIndependentEvaluator, register_evaluator


@register_evaluator("relevancy")
class RelevancyEvaluator(BaseIndependentEvaluator[RetrievalOutput]):
    """Deprecated: use :class:`ContextRetrievalEvaluator`, which also returns precision and context_f1."""

    def __init__(self) -> None:
        """Warn that this evaluator is deprecated."""
        warnings.warn(
            "RelevancyEvaluator is deprecated, use ContextRetrievalEvaluator instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def evaluate(self, candidate_output: RetrievalOutput, eval_sample: EvalSample) -> Metric:
        """Evaluate the proportion of the expected context contained in the retrieved context.

        :param candidate_output: The RAG's output to the query.
        :type candidate_output: RAGOutput
        :param eval_sample: The sample containing the expected context.
        :type eval_sample: EvalSample
        :return: The evaluation result with a ``"relevancy"`` key.
        :rtype: dict[str, Metric]
        """
        if not eval_sample.context:
            return Metric(1.0)
        if not candidate_output.retrieved_context:
            return Metric(0.0)
        retrieved_text = "\n".join(doc.text for doc in candidate_output.retrieved_context)
        nb_found = sum(ctx.text in retrieved_text for ctx in eval_sample.context)
        return Metric(nb_found / len(eval_sample.context))
