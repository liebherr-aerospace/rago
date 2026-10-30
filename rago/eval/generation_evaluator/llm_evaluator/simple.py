"""Define a Simple LLM Evaluator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rago.eval.base import register_evaluator

from .base import BaseLLMEvaluator, JudgeError, LLMMetric

if TYPE_CHECKING:
    from collections.abc import Sequence

@register_evaluator("simple_judge")
class SimpleLLMEvaluator(BaseLLMEvaluator):
    """A simple evaluator of generations using an LLM.

    This evaluator's answer is constrained to be in an interval (i.e. between min and max score).
    The LLM used to judge can sometimes output a score outside this interval.
    The way these judge's errors are handled depends on the policy_on_errors used.
    In case of pairwise evaluation the scores of each answers depend on their order.
    i.e. The judge LLM has a position bias.
    To solve this, answers are randomly shuffled before being presented to the judge.
    This make this bias disappear when comparing two generation model on many answers.
    """

    def parse_evaluation(self, evaluation: str) -> LLMMetric:
        """Parse the judge LLM output evaluation string and return the result.

        :param evaluation: The evaluation given by the judge to parse to get the score
        :type evaluation: str
        :raises ValueError: The evaluation can not be converted to a float.
        :return: The Judge result.
        :rtype: LLMMetric
        """
        score = self.get_score_from_string_score(evaluation)
        return LLMMetric(score)

    def get_score_from_string_score(self, score_string: str) -> float:
        """Get the float evaluation score from the string score.

        Raise an error if the score is not in the authorized interval.

        :param score_string: The string to convert to score.
        :type score_string: str
        :raises ValueError: The score_string is not a float.
        :return: The score.
        :rtype: float
        """
        score = float(score_string)
        if score > self.max_score or score < self.min_score:
            raise ValueError(score_string)
        return score

    def parse_n_wise_evaluation(self, evaluation: str, expected_number_of_score: int) -> Sequence[LLMMetric]:
        """Parse the judge LLM output evaluation string and return the results for each outputs.

        :param evaluation: The evaluation given by the judge to parse to get the score.
        :type evaluation: str
        :param expected_number_of_score: Expected number of score in the evaluation.
        :type expected_number_of_score: int
        :raises JudgeError: The Judge did not respect the guidelines.
        This can be because:
            - the judge outputs a score outside the allowed interval.
            - it does not follow evaluation template.
        :raises ValueError: Any of the scores in the evaluation can not be converted to a float.
        :return: The metric for of each generation.
        :rtype: Sequence[LLMMetric]
        """
        evaluation_scores = evaluation.strip("\n").split("\n")
        if len(evaluation_scores) != expected_number_of_score:
            raise JudgeError(evaluation)

        results = [
            LLMMetric(score=self.get_score_from_string_score(evaluation_score.strip()))
            for evaluation_score in evaluation_scores
        ]

        return results
