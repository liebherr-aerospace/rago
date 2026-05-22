"""Define an evaluator that measures the precision of the retrieved context.

Precision = (number of retrieved chunks that match an expected chunk)
            / (total number of retrieved chunks)

This penalises noisy retrieval: returning many irrelevant chunks lowers the
score even if the correct chunk is present.
"""

from __future__ import annotations

from rago.data_objects import EvalSample, Metric, RAGOutput
from rago.eval.base import BaseIndependentEvaluator


class PrecisionEvaluator(BaseIndependentEvaluator[RAGOutput]):
    """Evaluator that measures the precision of the RAG output's retrieved context.

    For each retrieved chunk, the evaluator checks whether it contains (or is
    contained by) any of the expected context chunks.  The precision is then:

        precision = nb_relevant_retrieved / nb_total_retrieved

    A config that returns *only* the correct chunk(s) scores 1.0.
    A config that returns the correct chunk buried among 19 irrelevant ones
    scores 1/20 = 0.05.
    """

    def evaluate(self, candidate_output: RAGOutput, eval_sample: EvalSample) -> dict[str, Metric]:
        """Evaluate the precision of the retrieved context.

        :param candidate_output: The RAG's output to the query.
        :type candidate_output: RAGOutput
        :param eval_sample: The context used to evaluate the candidate output.
        :type eval_sample: EvalSample
        :return: The evaluation result with a ``"precision"`` key.
        :rtype: dict[str, Metric]
        """
        # No expected context → nothing to judge, score = 1.
        if eval_sample.context is None or len(eval_sample.context) == 0:
            return {"precision": Metric(1.0)}

        # Expected context exists but nothing was retrieved → precision = 0.
        if candidate_output.retrieved_context is None or len(candidate_output.retrieved_context) == 0:
            return {"precision": Metric(0.0)}

        expected_texts = [ctx.text for ctx in eval_sample.context]

        nb_relevant = 0
        for retrieved_doc in candidate_output.retrieved_context:
            # A retrieved chunk is considered relevant if it contains any
            # expected text OR if any expected text contains it.
            for expected in expected_texts:
                if expected in retrieved_doc.text or retrieved_doc.text in expected:
                    nb_relevant += 1
                    break  # count each retrieved chunk at most once

        precision = nb_relevant / len(candidate_output.retrieved_context)
        return {"precision": Metric(precision)}
