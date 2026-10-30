"""Define the BertScore Evaluator."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import evaluate
from pydantic.dataclasses import dataclass

from rago.data_objects import EvalSample, GenerationOutput, Metric
from rago.eval.base import BaseIndependentEvaluator, EvaluatorReference, register_evaluator


@dataclass
class BertScoreMetric(Metric):
    """Contains the main evaluation score and additional scores computed from BertScore."""

    score: float
    additional_scores: dict[str, float]

@register_evaluator("bert_score")
class BertScore(BaseIndependentEvaluator[GenerationOutput]):
    """Define the Bert score that uses variants of the bert model to compute correctness by cosine similarity."""

    metrics: ClassVar[list[str]] = ["precision", "recall", "f1"]

    def __init__(self, metric: str) -> None:
        """Instantiate a BertScore by loading it from evaluate."""
        self.bert_metrics = evaluate.load("bertscore")
        if metric not in self.metrics:
            msg = f"{metric} is not a valid BertScore metric. Choose a metric name among: {self.metrics}"
            raise ValueError(msg)
        self.metric = metric

    def evaluate(self, candidate_output: GenerationOutput, reference: EvaluatorReference) -> BertScoreMetric:
        """Evaluate the candidate generation output using the reference information.

        :param candidate_output: The candidate generation output to evaluate.
        :type candidate_output: GenerationOutput
        :param reference: The reference used by the evaluator to evaluate the candidate.
        :type reference: EvaluatorReference
        :return: The Bert score metrics.
        :rtype: EvalOutputType
        """
        references = self.get_targets(reference.eval_sample)
        results = self.bert_metrics.compute(predictions=[candidate_output.answer], references=[references], lang="en")
        if results is None:
            msg = "Error with the BertScore backend"
            raise ValueError(msg)
        results.pop("hashcode")
        results = cast("dict[str, list[float]]", results)
        return self.get_evaluations(results)


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

    def _get_mean_scores(self, scores: list[float]) -> float:
        return sum(scores)/min(len(scores), 1)

    def get_evaluations(self, results: dict[str, Any]) -> BertScoreMetric:
        """Convert the dictionary to a dictionary of Metrics.

        :param results: The dictionary resulting from the Bert score eval.
        :type results: dict[str, Any]
        :return: The dictionary of metrics.
        :rtype: dict[str, Evaluation] | list[dict[str, Evaluation]]
        """
        scores = cast("dict[str, list[float]]", results)
        if self.metric not in scores:
            msg = f"{self.metric} is not a valid BertScore metric. Choose a metric name among: {self.metrics}"
            raise ValueError(msg)

        score = self._get_mean_scores(scores.pop(self.metric))
        additional_scores ={metric_name: self._get_mean_scores(metrics) for metric_name, metrics in scores.items()}
        return BertScoreMetric(score, additional_scores)
