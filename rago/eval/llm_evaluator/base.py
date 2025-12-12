"""Define the base evaluator class that uses a judge language model to evaluate RAG outputs."""

from __future__ import annotations

from abc import abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Optional

from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.data_objects import EvalSample, Metric, PromptTemplate, RAGOutput
from rago.eval.base import BaseDependentEvaluator
from rago.eval.order import Order
from rago.model.configs.llm_config.base import LLMConfig  # noqa: TC001
from rago.model.wrapper.llm_agent import LangchainLLMAgent, LLMAgent
from rago.model.wrapper.llm_agent.llm_agent_factory import LLMAgentFactory
from rago.prompts import DEFAULT_EVAL_PROMPT, DEFAULT_PAIRWISE_EVAL_PROMPT, SOURCE_CONTEXT_PROMPT

if TYPE_CHECKING:
    from rago.model.wrapper.llm_agent.base import LLMAgent


@dataclass
class LLMEvaluatorConfig:
    """Configuration parameters of the LLM."""

    min_score: float = 0
    max_score: float = 5
    judge: Optional[LLMConfig] = None


class PolicyOnError(StrEnum):
    """Define how the error of the judge should be handled.

    MIN_SCORE: When there is a JudgeError the evaluator should return min score.
    In case of pairwise evaluation that means returning min_score for both scores.
    NAN_SCORE: When there is a JudgeError the evaluator should return nan.
    In case of pairwise evaluation that means returning nan for both scores.
    THROW_ERROR: When there is a JudgeError the evaluator should throw a JudgeError.
    """

    MIN_SCORE = "min_score"
    NAN_SCORE = "nan_score"
    THROW_ERROR = "throw_error"


class JudgeError(Exception):
    """Error in the score given by the judge LM (either the format or the value)."""

    def __init__(self, evaluation: str) -> None:
        """Judge evaluation error: Evaluation is invalid.

        :param evaluation: The invalid evaluation causing the error.
        :type evaluation: str
        """
        super().__init__(f"The Judge gave an invalid evaluation: {evaluation}")


@dataclass
class EvalPrompts:
    """Contains all the prompt templates used by the LM evaluator.

    :param evaluation_prompt: The prompt used by the judge for direct evaluation.
    :type evaluation_prompt: str
    :param pairwise_evaluation_prompt: The prompt used by the judge for pairwise evaluation.
    type pairwise_evaluation_prompt: str
    :param source_context_prompt: The prompt to add when the eval_sample contains a source context.
    :type source_context_prompt: str
    """

    evaluation_prompt: str = Field(default=DEFAULT_EVAL_PROMPT, min_length=1)
    pairwise_evaluation_prompt: str = Field(default=DEFAULT_PAIRWISE_EVAL_PROMPT, min_length=1)
    source_context_prompt: str = Field(default=SOURCE_CONTEXT_PROMPT, min_length=1)


class BaseLLMEvaluator(BaseDependentEvaluator[RAGOutput]):
    """Abstract class that defines the evaluator that evaluates RAG outputs using an LM."""

    def __init__(
        self,
        judge: Optional[LLMAgent] = None,
        min_score: float = 0,
        max_score: float = 5,
        eval_prompts: Optional[EvalPrompts] = None,
        policy_on_errors: PolicyOnError = PolicyOnError.MIN_SCORE,
    ) -> None:
        """Create an instance of the BaseLLMEvaluator class.

        :param judge: The judge used by the evaluator to evaluate answers, defaults to None.
        :type judge: Optional[LLMAgent], optional
        :param min_score: The minimum score that the judge can give to an answer, defaults to 0.
        :type min_score: float, optional
        :param max_score: The maximum score that the judge can give to an answer, defaults to 5.
        :type max_score: float, optional
        :param eval_prompts: Contains all the prompt templates used by the LM evaluator, defaults to None.
        :type eval_prompts: Optional[EvalPrompts], optional
        :param policy_on_errors: How errors from the LM judge should be handled, defaults to PolicyOnError.MIN_SCORE.
        :type policy_on_errors: PolicyOnError, optional
        """
        super().__init__()

        self.judge: LLMAgent = judge if judge is not None else LangchainLLMAgent.make_from_backend()
        eval_prompts = EvalPrompts() if eval_prompts is None else eval_prompts
        self.eval_prompt = PromptTemplate(eval_prompts.evaluation_prompt)
        self.pairwise_eval_prompt = PromptTemplate(eval_prompts.pairwise_evaluation_prompt)
        self.source_context_prompt = PromptTemplate(eval_prompts.source_context_prompt)
        self.min_score = min_score
        self.max_score = max_score
        self.policy_on_errors = policy_on_errors
        self.eval_prompt = self.eval_prompt.get_partially_formatted_prompt_template(
            min_score=str(self.min_score),
            max_score=str(self.max_score),
        )
        self.pairwise_eval_prompt = self.pairwise_eval_prompt.get_partially_formatted_prompt_template(
            min_score=str(self.min_score),
            max_score=str(self.max_score),
        )

    def evaluate(self, candidate_output: RAGOutput, eval_sample: EvalSample) -> dict[str, Metric]:
        """Evaluate the candidate output on the eval_sample.

        :param candidate_output: The RAG output to evaluate.
        :type candidate_output: RAGOutput
        :param eval_sample: The sample to evaluate the candidate answer on.
        :type eval_sample: EvalSample
        :return: The evaluation score.
        :rtype: dict[str, Metric]
        """
        eval_prompt = self.get_filled_eval_prompt(candidate_output=candidate_output, eval_sample=eval_sample)
        evaluation = self.judge.query(eval_prompt)
        scores = self.get_evaluation_result_from_evaluation(evaluation)
        return scores

    def evaluate_pairwise(
        self,
        candidate_output_1: RAGOutput,
        candidate_output_2: RAGOutput,
        eval_sample: EvalSample,
    ) -> tuple[dict[str, Metric], dict[str, Metric]]:
        """Evaluate the two candidate outputs using the eval_sample information by passing them to the judge LM.

        :param candidate_output_1: The first candidate output to evaluate.
        :type candidate_output_1: RAGOutput
        :param candidate_output_2: The second candidate output to evaluate.
        :type candidate_output_2: RAGOutput
        :param eval_sample: The sample to evaluate the candidate answer on.
        :type eval_sample: EvalSample
        :return: The evaluation results containing the eventual scores and explanations for both outputs.
        :rtype: dict[str, Metric]
        """
        order = self.shuffle_answers()
        if order == Order.REVERSED:
            candidate_output_1, candidate_output_2 = candidate_output_2, candidate_output_1
        eval_prompt = self.get_filled_pairwise_eval_prompt(
            candidate_output_1=candidate_output_1,
            candidate_output_2=candidate_output_2,
            eval_sample=eval_sample,
        )
        evaluation = self.judge.query(eval_prompt)
        score1, score2 = self.get_pairwise_evaluation_result_from_evaluation(evaluation)
        if order == Order.REVERSED:
            score1, score2 = score2, score1
        return score1, score2

    def get_filled_eval_prompt(self, candidate_output: RAGOutput, eval_sample: EvalSample) -> str:
        """Get the prompt used by the judge language model to evaluate the output filled with the necessary information.

        :param candidate_output: The RAG output evaluated.
        :type candidate_output: RAGOutput
        :param eval_sample: The sample to evaluate the candidate answer on.
        :type eval_sample: EvalSample
        :return: The prompt with the placeholders filled.
        :rtype: str
        """
        source_context_content = (
            [context.text for context in eval_sample.context] if eval_sample.context is not None else None
        )
        source_context_prompt = self.get_filled_source_context_prompt(source_context_content)
        if candidate_output.answer is None:
            raise ValueError(candidate_output.answer)
        prompt = self.eval_prompt.get_filled_prompt(
            candidate_answer=candidate_output.answer,
            query=eval_sample.query,
            source_context_prompt=source_context_prompt,
        )
        return prompt

    def get_filled_source_context_prompt(self, source_context: Optional[list[str]]) -> str:
        """Return the filled source context prompt with source context if source context is not None.

        If the source context is None return an empty string.

        :param source_context: The source context to use to fill the prompt
        :type source_contextOptional[str]
        :return: The filled source context prompt.
        """
        if source_context is not None:
            source_context_prompt = self.source_context_prompt.get_filled_prompt(
                source_context=source_context,
            )
        else:
            source_context_prompt = ""
        return source_context_prompt

    def get_filled_pairwise_eval_prompt(
        self,
        candidate_output_1: RAGOutput,
        candidate_output_2: RAGOutput,
        eval_sample: EvalSample,
    ) -> str:
        """Get the filled prompt used by the judge LM for pairwise evaluation.

        :param candidate_output_1: The first RAG output evaluated.
        :type candidate_output_1: RAGOutput
        :param candidate_output_2: The second RAG output evaluated.
        :type candidate_output_2: RAGOutput
        :param eval_sample: The sample to evaluate the candidate answer on.
        :type eval_sample: EvalSample
        :return: The prompt with the place holders filled.
        :rtype: str
        """
        source_context_content = (
            [context.text for context in eval_sample.context] if eval_sample.context is not None else None
        )
        source_context_prompt = self.get_filled_source_context_prompt(source_context_content)
        if candidate_output_1.answer is None:
            raise ValueError(candidate_output_1.answer)
        if candidate_output_2.answer is None:
            raise ValueError(candidate_output_2.answer)

        prompt = self.pairwise_eval_prompt.get_filled_prompt(
            candidate_answer_1=candidate_output_1.answer,
            candidate_answer_2=candidate_output_2.answer,
            query=eval_sample.query,
            source_context_prompt=source_context_prompt,
        )
        return prompt

    def get_evaluation_result_from_evaluation(self, evaluation: str) -> dict[str, Metric]:
        """Get the evaluation result from the judge LM evaluation output string .

        :param evaluation: The evaluation given by the judge to parse to get the score.
        :type evaluation: str
        :return: The dict containing the score.
        :rtype: dict[str, Metric]
        """
        try:
            eval_result = self.parse_evaluation(evaluation)
        except (JudgeError, ValueError) as e:
            eval_result = self.process_error(e, evaluation)
        return eval_result

    def get_pairwise_evaluation_result_from_evaluation(
        self,
        evaluation: str,
    ) -> tuple[dict[str, Metric], dict[str, Metric]]:
        """Get the evaluation results for both answers from the judge output's evaluation string.

        :param evaluation: The pairwise evaluation output string from the judge.
        :type evaluation: str
        :return: The evaluation results of both answers.
        :rtype: tuple[dict[str, Metric], dict[str, Metric]]
        """
        try:
            score_1, score_2 = self.parse_pairwise_evaluation(evaluation)
        except (JudgeError, ValueError) as e:
            score = self.process_error(e, evaluation)
            score_1 = score
            score_2 = score

        return score_1, score_2

    @abstractmethod
    def parse_pairwise_evaluation(self, evaluation: str) -> tuple[dict[str, Metric], dict[str, Metric]]:
        """Parse the judge LM output evaluation string and return the results for both outputs.

        :param evaluation: The evaluation given by the judge to parse to get the score.
        :type evaluation: str
        :raises JudgeError: The Judge did not respect the guidelines.
        This can be because:
            - the judge outputs a score outside allowed interval.
            - it does not follow evaluation template.
        :raises ValueError: Any of the two scores' string in the evaluation can not be converted to a float.
        :return: The evaluation results of both answers.
        :rtype: tuple[dict[str, Metric], dict[str, Metric]]
        """

    @abstractmethod
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

    def process_error(self, exception: Exception, evaluation: str) -> dict[str, Metric]:
        """Process the error depending on the policy on error for evaluation.

        :param exception: Error to process.
        :type exception: ValueError
        :param evaluation: The output from the judge that led to the error.
        :type evaluation: str
        :raise ValueError: Incorrect PolicyOnError.
        :return: The eventual default value to replace the score.
        :rtype: float
        """
        match self.policy_on_errors:
            case PolicyOnError.MIN_SCORE:
                score_val = self.min_score
            case PolicyOnError.NAN_SCORE:
                score_val = float("nan")
            case PolicyOnError.THROW_ERROR:
                raise JudgeError(evaluation) from exception
            case _:
                raise ValueError from exception
        return {"correctness": Metric(score=score_val, explanation=exception.args[0])}

    @classmethod
    def make(
        cls,
        config: LLMEvaluatorConfig,
    ) -> BaseLLMEvaluator:
        """Build method to create an instance of a LLM Evaluator.

        :param config: Configuration of the evaluator to build.
        :type: config: LLMEvaluatorConfig
        :return: LLM Evaluator
        :rtype: BaseLLMEvaluator
        """
        llm = LLMAgentFactory.make(config.judge) if config.judge is not None else LangchainLLMAgent.make_from_backend()
        return cls(
            judge=llm,
            min_score=config.min_score,
            max_score=config.max_score,
        )
