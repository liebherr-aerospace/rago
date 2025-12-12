"""Define LLM Evaluator that uses chain of thought (CoT)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, override

from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.data_objects import Metric
from rago.eval.llm_evaluator.base import EvalPrompts, JudgeError, PolicyOnError
from rago.eval.llm_evaluator.simple import SimpleLLMEvaluator
from rago.prompts import (
    DEFAULT_COT_SCORE_TAG,
    DEFAULT_EXPLANATION_TAG,
    DEFAULT_SCORE_1_TAG,
    DEFAULT_SCORE_2_TAG,
)

if TYPE_CHECKING:
    from rago.model.wrapper.llm_agent.base import LLMAgent


@dataclass
class EvalOutputTags:
    """Contains all the tags used in the LM evaluator's output template.

    :param score_tag: Tag used by the judge in the direct evaluation's output to indicate the score.
    :type score_tag: str
    :param explanation_tag:  Tag used by the judge in its output to indicate its reasoning.
    :type explanation_tag: str
    :param score_1_tag: Tag used by the judge in the pairwise evaluation's output to indicate the first score.
    :type score_1_tag: str
    :param score_2_tag: Tag used by the judge in the pairwise evaluation's output to indicate the second score.
    :type score_2_tag: str
    """

    score_tag: str = Field(default=DEFAULT_COT_SCORE_TAG, min_length=1)
    explanation_tag: str = Field(default=DEFAULT_EXPLANATION_TAG, min_length=1)
    score_1_tag: str = Field(default=DEFAULT_SCORE_1_TAG, min_length=1)
    score_2_tag: str = Field(default=DEFAULT_SCORE_2_TAG, min_length=1)


class CoTLLMEvaluator(SimpleLLMEvaluator):
    """An evaluator of answers using an LLM prompted with chain of thought (CoT)."""

    def __init__(
        self,
        judge: LLMAgent,
        min_score: float,
        max_score: float,
        policy_on_errors: PolicyOnError,
        eval_prompts: Optional[EvalPrompts] = None,
        eval_output_tags: Optional[EvalOutputTags] = None,
    ) -> None:
        """Create a CoT LM evaluator to evaluate RAG outputs.

        The judge can only give float score between min_score and max_score.
        :param judge: The judge language model used to evaluate answers
        :type judge: LanguageModel
        :param min_score: The minimum score that the judge can give to an answer.
        :type min_score: float
        :param max_score: The maximum score that the judge can give to an answer.
        :type max_score: float
        :param policy_on_errors: How errors from the LM judge should be handled.
        :type policy_on_errors: PolicyOnError
        :param prompts: Contains all the prompt templates used by the LM evaluator.
        :type prompts: Optional[EvalPrompts]
        :param eval_output_tags: Contains all the tags that must be used in the LM evaluator's
        :type eval_output_tags: Optional[EvalOutputTags]
        """
        eval_output_tags = EvalOutputTags() if eval_output_tags is None else eval_output_tags
        eval_prompts = EvalPrompts() if eval_prompts is None else eval_prompts
        self.score_tag = eval_output_tags.score_tag
        self.explanation_tag = eval_output_tags.explanation_tag
        self.score_1_tag = eval_output_tags.score_1_tag
        self.score_2_tag = eval_output_tags.score_2_tag
        eval_prompts.evaluation_prompt = eval_prompts.evaluation_prompt.format(
            candidate_answer="{candidate_answer}",
            query="{query}",
            source_context_prompt="{source_context_prompt}",
            min_score="{min_score}",
            max_score="{max_score}",
            score_tag=eval_output_tags.score_tag,
            explanation_tag=eval_output_tags.explanation_tag,
        )
        eval_prompts.pairwise_evaluation_prompt = eval_prompts.pairwise_evaluation_prompt.format(
            candidate_answer_1="{candidate_answer_1}",
            candidate_answer_2="{candidate_answer_2}",
            query="{query}",
            source_context_prompt="{source_context_prompt}",
            min_score="{min_score}",
            max_score="{max_score}",
            score_1_tag=eval_output_tags.score_1_tag,
            score_2_tag=eval_output_tags.score_2_tag,
            explanation_tag=eval_output_tags.explanation_tag,
        )
        super().__init__(
            judge=judge,
            min_score=min_score,
            max_score=max_score,
            policy_on_errors=policy_on_errors,
            eval_prompts=eval_prompts,
        )

    @override
    def parse_evaluation(self, evaluation: str) -> dict[str, Metric]:
        """Parse the judge LM output evaluation string and return the result.

        :param evaluation: The evaluation given by the judge to parse to get the score.
        :type evaluation: str
        :raises JudgeError: The Judge did not respect the guidelines.
        This can be because:
            - the judge outputs a score outside allowed interval.
            - it does not follow evaluation template.
        :raises ValueError: The score string in the evaluation can not be converted to a float.
        :return: The evaluation result.
        :rtype: dict[str, Metric]
        """
        splitted_eval = evaluation.split(self.score_tag)
        expected_number_of_instance_of_score_tag = 2
        if len(splitted_eval) != expected_number_of_instance_of_score_tag:
            raise JudgeError(evaluation)
        explanation = self.remove_explanation_tag(splitted_eval[0])
        score = self.get_score_from_string_score(splitted_eval[-1])
        return {"correctness": Metric(score=score, explanation=explanation)}

    def remove_explanation_tag(self, explanation_with_explanation_tag: str) -> str:
        """Return the explanation string without explanation tag.

        :param explanation_with_explanation_tag: The string containing the explanation and its tag.
        :type explanation_with_explanation_tag: str
        :return: The explanation without its tag.
        :rtype: str
        """
        explanation_splitted = explanation_with_explanation_tag.split(self.explanation_tag)
        number_of_components_in_reasoning = 2
        if len(explanation_splitted) != number_of_components_in_reasoning:
            raise JudgeError(explanation_with_explanation_tag)
        explanation = explanation_splitted[-1]
        return explanation.strip()

    @override
    def parse_pairwise_evaluation(self, evaluation: str) -> tuple[dict[str, Metric], dict[str, Metric]]:
        """Parse the judge LM output evaluation string and return the results for both outputs.

        :param evaluation: The evaluation given by the judge to parse to get the score.
        :type evaluation: str
        ::raises JudgeError: The Judge did not respect the guidelines.
        This can be because:
            - the judge outputs a score outside allowed interval.
            - it does not follow evaluation template.
        :raises ValueError: Any of the two scores in the evaluation is not a float.
        :return: The evaluation results of both answers.
        :rtype: tuple[dict[str, Metric], dict[str, Metric]]
        """
        splitted_eval = evaluation.split(self.score_1_tag)
        expected_number_of_instance_of_score_tag_1 = 1
        if len(splitted_eval) != expected_number_of_instance_of_score_tag_1 + 1:
            raise JudgeError(evaluation)
        explanation = self.remove_explanation_tag(splitted_eval[0])
        splitted_scores = splitted_eval[1].split(self.score_2_tag)
        expected_number_of_instance_of_score_tag_2 = 1
        if len(splitted_scores) != expected_number_of_instance_of_score_tag_2 + 1:
            raise JudgeError(evaluation)
        eval_result_1, eval_result_2 = tuple(
            {"correctness": Metric(score=self.get_score_from_string_score(score_str), explanation=explanation)}
            for score_str in splitted_scores
        )
        return eval_result_1, eval_result_2
