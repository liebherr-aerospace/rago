"""Define the BertScore Evaluator."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import evaluate
import numpy as np

from rago.data_objects import EvalSample, Metric, RAGOutput
from rago.eval.base import BaseIndependentEvaluator, register_evaluator


@register_evaluator("bert_score")
class BertScore(BaseIndependentEvaluator[RAGOutput]):
    """Define the Bert score that uses variants of the bert model to compute correctness by cosine similarity."""

    metrics: ClassVar[list[str]] = ["precision", "recall", "f1"]

    def __init__(self) -> None:
        """Instantiate a BertScore by loading it from evaluate."""
        self.bert_metrics = evaluate.load("bertscore")

    def evaluate(
        self,
        outputs: RAGOutput,
        eval_samples: EvalSample,
    ) -> dict[str, Metric]:
        """Evaluate the candidate output using the eval_sample information.

        :param candidate_output: The RAG's output to the query.
        :type candidate_output: OutputType
        :param eval_sample: The sample to evaluate the candidate output on.
        :type eval_sample: EvalSample
        :return: The evaluation result.
        :rtype: EvalOutputType
        """
        predictions = self.get_predictions(outputs)
        references = self.get_targets(eval_samples)
        results = self.bert_metrics.compute(predictions=[predictions], references=[references], lang="en")
        results.pop("hashcode")
        results = cast("dict[str, list[float]]", results)
        return self.get_evaluations(results)

    def get_predictions(self, outputs: RAGOutput) -> str:
        """Extract the rag answer from its ouput.

        :param outputs: The RAG output
        :type outputs: RAGOutput
        :return: The answer of the RAG.
        :rtype: list[str]
        """
        return outputs.answer if outputs.answer is not None else ""

    def get_targets(self, eval_samples: EvalSample) -> str:
        """Extract the reference answer from the eval sample.

        :param eval_samples: The eval sample
        :type eval_samples: EvalSample
        :raises ValueError: If the eval sample has no reference answer.
        :return: The reference answer.
        :rtype: list[str]
        """
        if eval_samples.reference_answer is None:
            raise ValueError
        return eval_samples.reference_answer

    def get_evaluations(self, results: dict[str, Any]) -> dict[str, Metric]:
        """Convert the dictionary to a dictionary of Metrics.

        :param results: The dictionary resulting from the Bert score eval.
        :type results: dict[str, Any]
        :return: The dictionary of metrics.
        :rtype: dict[str, Evaluation] | list[dict[str, Evaluation]]
        """
        scores = cast("dict[str, list[float]]", results)
        mean_scores = {
            "bert_score_" + metric_name: Metric(np.mean(metrics).item()) for metric_name, metrics in scores.items()
        }
        return mean_scores
