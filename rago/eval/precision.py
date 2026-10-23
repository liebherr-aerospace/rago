"""Deprecated evaluator measuring only the precision of the retrieved context."""

from __future__ import annotations

import warnings

from rago.data_objects import EvalSample, Metric, RAGOutput
from rago.eval.retrieval import ContextRetrievalEvaluator


class PrecisionEvaluator(ContextRetrievalEvaluator):
    """Deprecated: use :class:`ContextRetrievalEvaluator`, which also returns relevancy and context_f1."""

    def __init__(self) -> None:
        """Warn that this evaluator is deprecated."""
        warnings.warn(
            "PrecisionEvaluator is deprecated, use ContextRetrievalEvaluator instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def evaluate(self, candidate_output: RAGOutput, eval_sample: EvalSample) -> dict[str, Metric]:
        """Evaluate the precision of the retrieved context.

        :param candidate_output: The RAG's output to the query.
        :type candidate_output: RAGOutput
        :param eval_sample: The sample containing the expected context.
        :type eval_sample: EvalSample
        :return: The evaluation result with a ``"precision"`` key.
        :rtype: dict[str, Metric]
        """
        return {"precision": Metric(self.compute_precision(candidate_output, eval_sample))}
