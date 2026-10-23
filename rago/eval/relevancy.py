"""Deprecated evaluator measuring only the relevancy (recall) of the retrieved context."""

from __future__ import annotations

import warnings

from rago.data_objects import EvalSample, Metric, RAGOutput
from rago.eval.retrieval import ContextRetrievalEvaluator


class RelevancyEvaluator(ContextRetrievalEvaluator):
    """Deprecated: use :class:`ContextRetrievalEvaluator`, which also returns precision and context_f1."""

    def __init__(self) -> None:
        """Warn that this evaluator is deprecated."""
        warnings.warn(
            "RelevancyEvaluator is deprecated, use ContextRetrievalEvaluator instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def evaluate(self, candidate_output: RAGOutput, eval_sample: EvalSample) -> dict[str, Metric]:
        """Evaluate the proportion of the expected context contained in the retrieved context.

        :param candidate_output: The RAG's output to the query.
        :type candidate_output: RAGOutput
        :param eval_sample: The sample containing the expected context.
        :type eval_sample: EvalSample
        :return: The evaluation result with a ``"relevancy"`` key.
        :rtype: dict[str, Metric]
        """
        return {"relevancy": Metric(self.compute_relevancy(candidate_output, eval_sample))}
