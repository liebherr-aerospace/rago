"""Define an evaluator that measures the recall, precision and F1 of the retrieved context."""

from __future__ import annotations

from rago.data_objects import EvalSample, Metric, RAGOutput
from rago.eval.base import BaseIndependentEvaluator


class ContextRetrievalEvaluator(BaseIndependentEvaluator[RAGOutput]):
    """Evaluator of the RAG output's retrieved context against the eval sample's context.

    Returns three metrics:
        - ``relevancy`` (recall): proportion of expected chunks found in the retrieved context.
        - ``precision``: proportion of retrieved chunks matching an expected chunk.
        - ``context_f1``: harmonic mean of ``relevancy`` and ``precision``.

    If the eval sample has no context, all metrics are 1.
    If it has context but nothing was retrieved, all metrics are 0.
    """

    def evaluate(self, candidate_output: RAGOutput, eval_sample: EvalSample) -> dict[str, Metric]:
        """Evaluate the retrieved context of the candidate output.

        :param candidate_output: The RAG's output to the query.
        :type candidate_output: RAGOutput
        :param eval_sample: The sample containing the expected context.
        :type eval_sample: EvalSample
        :return: The ``relevancy``, ``precision`` and ``context_f1`` metrics.
        :rtype: dict[str, Metric]
        """
        relevancy = self.compute_relevancy(candidate_output, eval_sample)
        precision = self.compute_precision(candidate_output, eval_sample)
        denominator = relevancy + precision
        f1 = 0.0 if denominator == 0 else 2 * relevancy * precision / denominator
        return {
            "relevancy": Metric(relevancy),
            "precision": Metric(precision),
            "context_f1": Metric(f1),
        }

    @staticmethod
    def compute_relevancy(candidate_output: RAGOutput, eval_sample: EvalSample) -> float:
        """Compute the proportion of the expected context contained in the retrieved context.

        :param candidate_output: The RAG's output to the query.
        :type candidate_output: RAGOutput
        :param eval_sample: The sample containing the expected context.
        :type eval_sample: EvalSample
        :return: The relevancy (recall) score.
        :rtype: float
        """
        if not eval_sample.context:
            return 1.0
        if not candidate_output.retrieved_context:
            return 0.0
        retrieved_text = "\n".join(doc.text for doc in candidate_output.retrieved_context)
        nb_found = sum(ctx.text in retrieved_text for ctx in eval_sample.context)
        return nb_found / len(eval_sample.context)

    @staticmethod
    def compute_precision(candidate_output: RAGOutput, eval_sample: EvalSample) -> float:
        """Compute the proportion of retrieved chunks that match an expected chunk.

        A retrieved chunk matches if it contains, or is contained in, any expected chunk.

        :param candidate_output: The RAG's output to the query.
        :type candidate_output: RAGOutput
        :param eval_sample: The sample containing the expected context.
        :type eval_sample: EvalSample
        :return: The precision score.
        :rtype: float
        """
        if not eval_sample.context:
            return 1.0
        if not candidate_output.retrieved_context:
            return 0.0
        expected_texts = [ctx.text for ctx in eval_sample.context]
        nb_relevant = sum(
            any(expected in doc.text or doc.text in expected for expected in expected_texts)
            for doc in candidate_output.retrieved_context
        )
        return nb_relevant / len(candidate_output.retrieved_context)
