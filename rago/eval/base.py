"""Define the abstract evaluator classes that evaluate one or more Model output using context information."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.data_objects import Candidate, EvalSample, Metric, ModelOutput

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence



EVALUATORS_REGISTRY: dict[str, type[BaseEvaluator]] = {}

@dataclass
class EvaluatorReference:
    """Contains the references information used by the Evaluator to evaluate the samples."""

    eval_sample: EvalSample

def register_evaluator[EvaluatorType: BaseEvaluator](name: str) -> Callable[[type[EvaluatorType]], type[EvaluatorType]]:
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


class BaseEvaluator[ModelOutputT: ModelOutput, ReferenceT: EvaluatorReference](ABC):
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
    def load(cls, name: str, **kwargs: Any) -> BaseEvaluator[ModelOutputT, ReferenceT]:  # noqa: ANN401
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
    def evaluate(self, candidate_output: ModelOutputT, reference: ReferenceT) -> Metric:
        """Evaluate the candidate RAG output using the eval_sample information.

        :param candidate_output: The RAG's output to the query.
        :type candidate_output: OutputType
        :param eval_sample: The sample to evaluate the candidate output on.
        :type eval_sample: EvalSample
        :return: The evaluation result.
        :rtype: EvalOutputType
        """

    @abstractmethod
    def evaluate_n_wise(
        self,
        candidate_outputs: list[ModelOutputT],
        reference: ReferenceT,
    ) -> Sequence[Metric]:
        """Compare and evaluate the list of candidate outputs using the eval_sample.

        :param candidate_outputs: The list candidate outputs to evaluate to evaluate on sample.
        :type candidate_outputs: list[OutputType]
        :param eval_sample: The sample to evaluate the candidate output on.
        :type eval_sample: EvalSample
        :return: The evaluation results containing the eventual scores and explanations for each outputs.
        :rtype: list[dict[str, Metric]]
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


class BaseIndependentEvaluator[ModelOutputT: ModelOutput](BaseEvaluator[ModelOutputT, EvaluatorReference]):
    """Abstract Class for the evaluators that can not evaluate two answers at the same time."""

    def evaluate_n_wise(
        self,
        candidate_outputs: list[ModelOutputT],
        reference: EvaluatorReference,
    ) -> Sequence[Metric]:
        """Compare and evaluate the list of candidate outputs using the eval_sample.

        :param candidate_outputs: The list candidate outputs to evaluate to evaluate on sample.
        :type candidate_outputs: list[OutputType]
        :param eval_sample: The sample to evaluate the candidate output on.
        :type eval_sample: EvalSample
        :return: The evaluation results containing the eventual scores and explanations for each outputs.
        :rtype: list[dict[str, Metric]]
        """
        results_candidate_answer = [self.evaluate(out, reference) for out in candidate_outputs]
        return results_candidate_answer

@dataclass
class DependentEvaluatorReference[ModelOutputT: ModelOutput](EvaluatorReference):
    """Contains the references information used by the Evaluator to evaluate the samples."""

    eval_sample: EvalSample
    ref_candidate: Optional[Candidate[ModelOutputT]] = Field(default = None)

class BaseDependentEvaluator[ModelOutputT: ModelOutput](
    BaseEvaluator[ModelOutputT, DependentEvaluatorReference[ModelOutputT]]):
    """Abstract class for evaluators that produce different results when the outputs are evaluated pairwise."""

    def shuffle_answers(self, num_answers: int) -> list[int]:
        """Decide whether the answers should be shuffled before being given to the LLM.

        :return: Enum indicating whether the answer should be shuffled
        :rtype: Order
        """
        return cast("list[int]", np.random.permutation(num_answers).tolist())  # noqa: NPY002
