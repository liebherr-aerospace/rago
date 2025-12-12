"""Define the abstract evaluator classes that evaluate one or more RAG output using context information."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar, Union

from rago.data_objects import DataObject, EvalSample, Metric
from rago.eval.order import Order

if TYPE_CHECKING:
    from collections.abc import Callable

EVALUATORS_REGISTRY: dict[str, type[BaseEvaluator]] = {}

EvaluatorType = TypeVar("EvaluatorType", bound="BaseEvaluator")


def register_evaluator(name: str) -> Callable[[type[EvaluatorType]], type[EvaluatorType]]:
    """Register an evaluator.

    :param name: Evaluator name.
    :type name: str
    :return: Decorator function.
    :rtype: Callable
    """

    def decorator(cls: type[EvaluatorType]) -> type[EvaluatorType]:
        EVALUATORS_REGISTRY[name] = cls
        return cls

    return decorator


OutputType = TypeVar("OutputType", bound=DataObject)
EvalOutputType = TypeVar("EvalOutputType", bound=Union[Metric, dict[str, Metric]])


class UnRegisteredEvaluatorError(ValueError):
    """Exception raised when attempting to access a processor that is not registered.

    This error is typically thrown when a dataset name is provided for which
    no corresponding processor has been registered in the processor registry.
    """

    def __init__(self, name: str) -> None:
        """Initialize the error with the name of the missing evaluator .

        :param name: The evaluator name for which no evaluator was found in the registry.
        :type name: str
        """
        super().__init__(f"There is no evaluator registered with name: {name}.")


class BaseEvaluator[OutputType: DataObject](ABC):
    """Abstract class that defines the evaluator that evaluates RAG outputs using context information.

    The RAG outputs can contain: an answer, a set of retrieved contexts or both.
    The context information is made of:
        - the query
        - (Optional) The context, the explanation and the reference answer.

    Example Usage:
    In RAGO, this evaluator is then called by the Optimization Manager
    to assess the performance of a particular RAG config on a given question.
    """

    @classmethod
    def load(cls, name: str, **kwargs: Any) -> BaseEvaluator:  # noqa: ANN401
        """Load evaluator from the evaluator registry.

        :param name: The name of the evaluator to load.
        :type name: str
        :raises ValueError: if the evaluator is not in the registry.
        :return: The evaluator.
        :rtype: BaseEvaluator
        """
        if name in EVALUATORS_REGISTRY:
            return EVALUATORS_REGISTRY[name](**kwargs)
        raise UnRegisteredEvaluatorError(name=name)

    @classmethod
    def list_available_evaluators(cls) -> list[str]:
        """Get the available evaluators.

        :return: The available evaluators.
        :rtype: list[str]
        """
        return list(EVALUATORS_REGISTRY.keys())

    @abstractmethod
    def evaluate(self, candidate_output: OutputType, eval_sample: EvalSample) -> dict[str, Metric]:
        """Evaluate the candidate RAG output using the eval_sample information.

        :param candidate_output: The RAG's output to the query.
        :type candidate_output: OutputType
        :param eval_sample: The sample to evaluate the candidate output on.
        :type eval_sample: EvalSample
        :return: The evaluation result.
        :rtype: EvalOutputType
        """

    @abstractmethod
    def evaluate_pairwise(
        self,
        candidate_output_1: OutputType,
        candidate_output_2: OutputType,
        eval_sample: EvalSample,
    ) -> tuple[dict[str, Metric], dict[str, Metric]]:
        """Compare and evaluate two candidate outputs using the eval_sample information.

        :param candidate_output_1: The first candidate output to evaluate.
        :type candidate_output_1: OutputType
        :param candidate_output_2: The second candidate output to evaluate.
        :type candidate_output_2: OutputType
        :param eval_sample: The sample to evaluate the candidate output on.
        :type eval_sample: EvalSample
        :return: The evaluation results containing the eventual scores and explanations for both outputs.
        :rtype: tuple[EvalOutputType, EvalOutputType]
        """

    @classmethod
    def update_avg_score(cls, score: float, score_eval: float, iteration_n: int) -> float:
        """Update the average score based on the given evaluation score and iteration count.

        :param score: The current score.
        :type score: float
        :param score_eval: The new evaluation score.
        :type score_eval: float
        :param iteration_n: The iteration count.
        :type iteration_n: int
        :return: The updated average score.
        :rtype: float
        """
        score += (score_eval - score) / (iteration_n + 1.0)
        return score


class BaseIndependentEvaluator[OutputType: DataObject](BaseEvaluator[OutputType]):
    """Abstract Class for the evaluators that can not evaluate two answers at the same time."""

    def evaluate_pairwise(
        self,
        candidate_output_1: OutputType,
        candidate_output_2: OutputType,
        eval_sample: EvalSample,
    ) -> tuple[dict[str, Metric], dict[str, Metric]]:
        """Compare and evaluate two candidate outputs using the eval_sample.

        :param candidate_output_1: The first candidate output to evaluate.
        :type candidate_output_1: OutputType
        :param candidate_output_2: The second candidate output to evaluate.
        :type candidate_output_2: OutputType
        :param eval_sample: The sample to evaluate the candidate output on.
        :type eval_sample: EvalSample
        :return: The evaluation results containing the eventual scores and explanations for both outputs.
        :rtype: tuple[EvalOutputType, EvalOutputType]
        """
        results_candidate_answer_1 = self.evaluate(candidate_output_1, eval_sample)
        results_candidate_answer_2 = self.evaluate(candidate_output_2, eval_sample)
        return results_candidate_answer_1, results_candidate_answer_2


class BaseDependentEvaluator[OutputType: DataObject](BaseEvaluator[OutputType]):
    """Abstract class for evaluators that produce different results when the outputs are evaluated pairwise."""

    def shuffle_answers(self) -> Order:
        """Decide whether the answers should be shuffled before being given to the LLM.

        :return: Enum indicating whether the answer should be shuffled
        :rtype: Order
        """
        return Order(value=random.choice(list(Order.__members__.values())))
