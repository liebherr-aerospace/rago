"""Define an evaluator that measures how well the retriever recalls reference context chunks."""

from __future__ import annotations

from typing import ClassVar

from rago.data_objects import EvalSample, Metric, RAGOutput
from rago.eval.base import BaseIndependentEvaluator, register_evaluator


@register_evaluator("context_recall")
class ContextRecallScore(BaseIndependentEvaluator[RAGOutput]):
    """Evaluator that measures the recall of reference context chunks among the retrieved chunks.

    For each reference chunk in the eval_sample context, it checks whether the chunk
    is present (via substring containment) in any of the retrieved chunks.

    The score is the proportion of reference chunks that were found in the retrieved context:
        context_recall = |reference chunks found in retrieved| / |reference chunks|

    This evaluator does NOT require a generated answer — only the retrieved_context
    from the RAGOutput and the reference context from the EvalSample.
    It is therefore ideal for retriever-only optimization.
    """

    metrics: ClassVar[list[str]] = ["context_recall"]

    def evaluate(self, candidate_output: RAGOutput, eval_sample: EvalSample) -> dict[str, Metric]:
        """Evaluate the recall of reference context chunks in the retrieved context.

        :param candidate_output: The RAG's output (only retrieved_context is used).
        :type candidate_output: RAGOutput
        :param eval_sample: The eval sample containing reference context chunks.
        :type eval_sample: EvalSample
        :return: Dictionary with the context_recall metric.
        :rtype: dict[str, Metric]
        """
        # If no reference context, retrieval is trivially correct
        if eval_sample.context is None or len(eval_sample.context) == 0:
            return {"context_recall": Metric(1.0)}

        # If no retrieved context but reference context exists, recall is 0
        if candidate_output.retrieved_context is None or len(candidate_output.retrieved_context) == 0:
            return {"context_recall": Metric(0.0)}

        # Concatenate all retrieved chunks into a single string for substring search
        retrieved_texts = [chunk.text for chunk in candidate_output.retrieved_context]
        retrieved_joined = "\n".join(retrieved_texts)

        # Count how many reference chunks are found in the retrieved context
        found = sum(1 for ref_doc in eval_sample.context if ref_doc.text in retrieved_joined)
        recall = found / len(eval_sample.context)

        return {"context_recall": Metric(recall)}
