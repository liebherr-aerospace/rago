"""Define a Simple LLM Evaluator."""

from __future__ import annotations

from rago.data_objects import Metric
from rago.eval.llm_evaluator.base import BaseLLMEvaluator, JudgeError


class SimpleLLMEvaluator(BaseLLMEvaluator):
    """A simple evaluator of answers using an LM.

    This evaluator's answer is constrained to be in an interval (i.e. between min and max score).
    The LM used to judge can sometimes output a score outside this interval.
    The way these judge's errors are handled depends on the policy_on_errors used.
    In case of pairwise evaluation the scores of each answers depend on their order.
    i.e. The judge LM has a position bias.
    To solve this, answers are randomly shuffled before being presented to the judge.
    This make this bias disappear when comparing two RAG configs on many answers.
    """

    def parse_evaluation(self, evaluation: str) -> dict[str, Metric]:
        """Parse the judge LM output evaluation string and return the result.

        :param evaluation: The evaluation given by the judge to parse to get the score
        :type evaluation: str
        :raises ValueError: The evaluation can not be converted to a float.
        :return: The evaluation result.
        :rtype: dict[str, Metric]
        """
        score = self.get_score_from_string_score(evaluation)
        return {"correctness": Metric(score=score)}

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

    def parse_n_wise_evaluation(self, evaluation: str, expected_number_of_score: int) -> list[dict[str, Metric]]:
        """Parse the judge LM output evaluation string and return the results for each outputs.

        :param evaluation: The evaluation given by the judge to parse to get the score.
        :type evaluation: str
        :param expected_number_of_score: Expected number of score in the evaluation.
        :type expected_number_of_score: int
        :raises JudgeError: The Judge did not respect the guidelines.
        This can be because:
            - the judge outputs a score outside the allowed interval.
            - it does not follow evaluation template.
        :raises ValueError: Any of the scores in the evaluation can not be converted to a float.
        :return: The evaluation results of each answers.
        :rtype: list[dict[str, Metric]]
        """
        evaluation_scores = evaluation.strip("\n").split("\n")
        if len(evaluation_scores) != expected_number_of_score:
            raise JudgeError(evaluation)

        results = [
            {"correctness": Metric(score=self.get_score_from_string_score(evaluation_score.strip()))}
            for evaluation_score in evaluation_scores
        ]

        return results
