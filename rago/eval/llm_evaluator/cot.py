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
    DEFAULT_SCORE_I_TAG_TEMPLATE,
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
    score_i_tag_template: str = Field(default=DEFAULT_SCORE_I_TAG_TEMPLATE, min_length=1)


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
        self.score_i_tag_template = eval_output_tags.score_i_tag_template
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
            score_1_tag=eval_output_tags.score_i_tag_template.format(i=1),
            score_2_tag=eval_output_tags.score_i_tag_template.format(i=2),
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
    def parse_n_wise_evaluation(self, evaluation: str, expected_number_of_score: int) -> list[dict[str, Metric]]:
        """Parse the judge LM output evaluation string and return the results for each outputs.

        :param evaluation: The evaluation given by the judge to parse to get the score.
        :type evaluation: str
        :param expected_number_of_score: Expected number of score in the evaluation.
        :type expected_number_of_score: int
        ::raises JudgeError: The Judge did not respect the guidelines.
        This can be because:
            - the judge outputs a score outside allowed interval.
            - it does not follow evaluation template.
        :raises ValueError: Any of the scores in the evaluation is not a float.
        :return: The evaluation results of each answers.
        :rtype: list[dict[str, Metric]]
        """
        reasoning_with_tag, rest_to_parse = self.get_last_n_remaining_answer(
            evaluation,
            expected_number_of_score,
            evaluation,
        )
        explanation = self.remove_explanation_tag(reasoning_with_tag)
        scores = []
        for i in range(1, expected_number_of_score):
            score_i, rest_to_parse = self.get_last_n_remaining_answer(
                rest_to_parse,
                expected_number_of_score - i,
                evaluation,
            )
            scores.append(score_i)
        scores.append(rest_to_parse)
        eval_results = [
            {"correctness": Metric(score=self.get_score_from_string_score(score_str), explanation=explanation)}
            for score_str in scores
        ]
        return eval_results

    def get_last_n_remaining_answer(
        self,
        rest_to_parse: str,
        nb_answer_remaining_to_parse: int,
        full_eval: str,
    ) -> tuple[str, str]:
        """Parse the first answer in the prompt and return it with the remaining part of prompt to parse.

        :param rest_to_parse: prompt to parse
        :type rest_to_parse: str
        :param nb_answer_remaining_to_parse: number of answer that will remain to parse after parsing
        :type nb_answer_remaining_to_parse: int
        :param full_eval: Full evaluation prompt
        :type full_eval: str
        :raises JudgeError: The Judge did not respect the guidelines.
        This can be because:
            - the judge outputs a score outside allowed interval.
            - it does not follow evaluation template
        :return: the first answer parsed and the remaining prompt to parse.
        :rtype: tuple[str, str]
        """
        expected_number_of_splits = 2
        splitted = rest_to_parse.split(
            self.score_i_tag_template.format(i=nb_answer_remaining_to_parse),
            maxsplit=expected_number_of_splits - 1,
        )
        if len(splitted) != expected_number_of_splits:
            raise JudgeError(full_eval)
        return splitted[0], splitted[1]
