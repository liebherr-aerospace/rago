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

    def evaluate_n_wise(
        self,
        candidate_outputs: list[RAGOutput],
        eval_sample: EvalSample,
    ) -> list[dict[str, Metric]]:
        """Evaluate the candidate outputs using the eval_sample information with all the evaluators.

        :param candidate_output: The candidate outputs to evaluate.
        :type candidate_output_1: list[RAGOutput]
        :param eval_sample: The sample to evaluate the candidate answer on.
        :type eval_sample: EvalSample
        :return: The evaluation results containing the eventual scores and explanations for both outputs.
        :rtype:  list[dict[str, Metric]]
        """
        results: list[dict[str, Metric]] = [{} for _ in range(len(candidate_outputs))]
        for metric_name, metric in self.evaluators.items():
            results_metric = metric.evaluate_n_wise(candidate_outputs, eval_sample)
            for idx in range(len(candidate_outputs)):
                results[idx][metric_name] = results_metric[idx][metric_name]
        return results
