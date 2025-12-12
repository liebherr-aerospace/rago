"""Define a sequential evaluator that sequentially applies the evaluators in the list."""

from rago.data_objects import EvalSample, Metric, RAGOutput
from rago.eval import BaseEvaluator


class SequentialEvaluator(BaseEvaluator):
    """Sequentially applies evaluators."""

    def __init__(self, evaluators: dict[str, BaseEvaluator]) -> None:
        """Instantiate a sequential evaluator with the dictionary containing the evaluators it will apply sequentially.

        :param evaluators: The evaluators to apply sequentially.
        :type evaluators: dict[BaseEvaluator]
        """
        self.evaluators = evaluators

    def evaluate(self, candidate_ouput: RAGOutput, eval_sample: EvalSample) -> dict[str, Metric]:
        """Evaluate the candidate output on the eval_sample with all the evaluators.

        :param candidate_output: The RAG output to evaluate.
        :type candidate_output: RagOutput
        :param eval_sample: The sample to evaluate the candidate answer on.
        :type eval_sample: EvalSample
        :return: The evaluation for all metrics.
        :rtype: dict[str, Metric]
        """
        result = {
            metric_name: metric.evaluate(candidate_ouput, eval_sample)[metric_name]
            for metric_name, metric in self.evaluators.items()
        }
        return result

    def evaluate_pairwise(
        self,
        candidate_output_1: RAGOutput,
        candidate_output_2: RAGOutput,
        eval_sample: EvalSample,
    ) -> tuple[dict[str, Metric], dict[str, Metric]]:
        """Evaluate the two candidate outputs using the eval_sample information with all the evaluators.

        :param candidate_output_1: The first candidate output to evaluate.
        :type candidate_output_1: RagOutput
        :param candidate_output_2: The second candidate output to evaluate.
        :type candidate_output_2: RagOutput
        :param eval_sample: The sample to evaluate the candidate answer on.
        :type eval_sample: EvalSample
        :return: The evaluation results containing the eventual scores and explanations for both outputs.
        :rtype: dict[str, Metric]
        """
        result_1 = {}
        result_2 = {}
        for metric_name, metric in self.evaluators.items():
            result_metric_1, result_metric_2 = metric.evaluate_pairwise(
                candidate_output_1,
                candidate_output_2,
                eval_sample,
            )
            result_1[metric_name] = result_metric_1[metric_name]
            result_2[metric_name] = result_metric_2[metric_name]
        return result_1, result_2
