"""Define an evaluator that evaluates the relevancy of the retrieved context."""

import numpy as np

from rago.data_objects import EvalSample, Metric, RAGOutput
from rago.eval.base import BaseIndependentEvaluator


class RelevancyEvaluator(BaseIndependentEvaluator[RAGOutput]):
    """Evaluator that evaluates the relevancy of the RAG output's retrieved context.

    i.e. The percentage of the eval sample's context that is in the retrieved context of the RAG output.
    """

    def evaluate(self, candidate_output: RAGOutput, eval_sample: EvalSample) -> dict[str, Metric]:
        """Evaluate the percentage of the evaluation context that is in the retrieved context in RAGOutput.

        If the query was not generated using any context (e.g general queries like: "Hello!") the relevancy score is 1.
        If the output does not contain any retrieved context and the eval_sample does the relevancy score is 0.

        :param candidate_output: The RAG's output to the query.
        :type candidate_output: RAGOutput
        :param eval_sample: The context used to evaluate the candidate answer.
        :type eval_sample: EvalSample
        :return: The evaluation result.
        :rtype: dict[str, Metric]
        """
        if eval_sample.context is not None and len(eval_sample.context) > 0:
            if candidate_output.retrieved_context is None or len(candidate_output.retrieved_context) == 0:
                relevancy = 0
            else:
                retrieved_context = "\n".join(
                    [retrieved_doc.text for retrieved_doc in candidate_output.retrieved_context],
                )
                nb_of_eval_sample_in_retrieved_context = np.sum(
                    [context.text in retrieved_context for context in eval_sample.context],
                )
                relevancy = nb_of_eval_sample_in_retrieved_context / len(eval_sample.context)
        else:
            relevancy = 1
        return {"relevancy": Metric(relevancy)}
