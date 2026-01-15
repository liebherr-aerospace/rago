"""Define the optimization manager that orchestrate the Optuna optimization process.

- Call the DataSetGenerator to generate dataset used for evaluation.
- Iteratively decide the RAG configuration to evaluate.
- instantiate the corresponding RAG configuration.
- Pass the answer to the Evaluator for evaluation.
- Decide what to do next based on the result and previous result (stop or continue)
- Return best RAG configuration
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Optional, Self

import numpy as np
import optuna
from pydantic.dataclasses import dataclass

from rago.dataset.generator.simple import SeedDataType, SimpleDatasetGenerator
from rago.eval import BaseEvaluator
from rago.optimization.search_space.rag_config_space import RAGConfigSpace
from rago.prompts import PromptConfig

if TYPE_CHECKING:
    from rago.data_objects import EvalSample, RAGOutput
    from rago.dataset import RAGDataset
    from rago.dataset.generator import DatasetGeneratorConfig

from enum import StrEnum

from rago.data_objects import DataObject, Metric
from rago.model.wrapper.rag.base import RAG, RAGConfig
from rago.optimization.repository.optuna_experiments_repository import OptunaExperimentRepository


class EvalMode(StrEnum):
    """Eval mode of the current evaluation."""

    TRAIN = "train"
    TEST = "test"


@dataclass
class RAGCandidateEval(DataObject):
    """Performance and config of a rag on train and test set."""

    config: RAGConfig
    train_score: float
    mean_evals: dict[str, float]
    full_evals: dict[str, list[float]]


@dataclass
class OptimParams:
    """Parameters of the optimization."""

    direction: str = "maximize"
    path_experiments: str = "experiments"
    experiment_name: str = "experiment_001"
    log_to_file: bool = True
    n_startup_trials: int = 50
    n_iter: Optional[int] = None
    show_progress_bar: bool = True


class BaseOptunaManager[EvaluatorType: BaseEvaluator[RAGOutput]](ABC):
    """An Abstract Class that defines the optimization manager."""

    manager: optuna.Study
    config_space: RAGConfigSpace
    logger: logging.Logger

    def __init__(
        self,
        *,
        params: Optional[OptimParams] = None,
        datasets: dict[str, RAGDataset],
        optim_evaluator: EvaluatorType,
        optim_metric_name: str,
        test_evaluators: list[BaseEvaluator],
        config_space: Optional[RAGConfigSpace] = None,
        prompt_config: Optional[PromptConfig] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> None:
        """Optimization Manager initialization.

        :param params: Parameters of the optimization, defaults to None.
        :type params: Optional[OptimParams], optional
        :param dataset: Dataset on which rag configs will be evaluated.
        :type dataset: RAGDataset
        :param optim_evaluator: Evaluator used to evaluate RAG outputs.
        :type optim_evaluator: BaseLLMEvaluator
        :param optim_metric_name: Name of the metric to optimize if the evaluator returns a dict, defaults to None.
        :type optim_metric_name: Optional[str], optional
        :param test_evaluators: Evaluators used in test.
        :type test_evaluators: list[EvaluatorType]
        :param config_space: The space of RAG config to search in, defaults to None
        :type config_space: Optional[RAGConfigSpace], optional
        :param prompt_config: Configuration of the prompt used by the reader of each RAG.
        :type prompt_config: Optional[PromptConfig], optional
        :param sampler: The sampler used to suggest new rag configuration to tests, defaults to None
        :type sampler: Optional[optuna.samplers.BaseSampler], optional
        :param pruner: The pruner used to terminate early unpromising trials, defaults to None
        :type pruner: Optional[optuna.pruners.BasePruner], optional
        """
        self.params = params if params is not None else OptimParams()
        self.datasets = datasets
        self.optim_evaluator = optim_evaluator
        self.optim_metric_name = optim_metric_name
        self.test_evaluators = test_evaluators
        self.config_space = config_space if config_space is not None else RAGConfigSpace()
        self.prompt_config = prompt_config if prompt_config is not None else PromptConfig()
        self.sampler = sampler
        self.pruner = pruner
        self.initialize_experiment_repository()
        self.initialize_logger()
        self.initialize_optuna_study()

    @classmethod
    def from_seed_data(  # noqa: PLR0913
        cls,
        *,
        params: Optional[OptimParams] = None,
        seed_data: SeedDataType,
        dataset_generator_config: Optional[DatasetGeneratorConfig] = None,
        splits: tuple[float, float],
        optim_evaluator: EvaluatorType,
        optim_metric_name: str,
        test_evaluators: list[BaseEvaluator],
        config_space: Optional[RAGConfigSpace] = None,
        prompt_config: Optional[PromptConfig] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> Self:
        """Initialize the Simple Direct Optimization Manager from seed data to generate dataset.

        :param params: Parameters of the optimization, defaults to None.
        :type params: Optional[OptimParams], optional
        :param seed_data: If no dataset is provided seed data used to generate a dataset, defaults to None.
        :type seed_data: Optional[RAGDataset], optional
        :param dataset_generator_config: Generator Configuration used to generate a dataset, defaults to None.
        :type dataset_generator_config: Optional[DatasetGeneratorConfig], optional
        :param splits: split sizes of the generated dataset.
        :type splits: tuple[int, int]
        :param optim_evaluator: Evaluator used to evaluate RAG outputs.
        :type optim_evaluator: BaseLLMEvaluator
        :param optim_metric_name: Name of the metric to optimize if the evaluator returns a dict.
        :type optim_metric_name: str
        :param test_evaluators: Evaluators used in test, if None evaluator is used for tests, defaults to None.
        :type test_evaluators: Optional[list[EvaluatorType]] = None
        :param config_space: The space of RAG config to search in, defaults to None
        :type config_space: Optional[RAGConfigSpace], optional
        :param prompt_config: Configuration of the prompt used by the reader of each RAG.
        :type prompt_config: Optional[PromptConfig], optional
        :param sampler: The sampler used to suggest new rag configuration to tests, defaults to None
        :type sampler: Optional[optuna.samplers.BaseSampler], optional
        :param pruner: The pruner used to terminate early unpromising trials, defaults to None
        :type pruner: Optional[optuna.pruners.BasePruner], optional
        """
        return cls(
            params=params,
            datasets=cls.get_dataset(seed_data, dataset_generator_config).split_dataset(splits, ["train", "test"]),
            optim_evaluator=optim_evaluator,
            optim_metric_name=optim_metric_name,
            test_evaluators=test_evaluators,
            config_space=config_space,
            prompt_config=prompt_config,
            sampler=sampler,
            pruner=pruner,
        )

    @classmethod
    def get_dataset(
        cls,
        seed_data: Optional[SeedDataType] = None,
        dataset_generator_config: Optional[DatasetGeneratorConfig] = None,
    ) -> RAGDataset:
        """Get the dataset to use for optimization.

        :param dataset: The dataset to use, defaults to None
        :type dataset: Optional[RAGDataset], optional
        :param seed_data: The seed data to use to generate the dataset, defaults to None
        :type seed_data: Optional[RAGDataset], optional
        :param dataset_generator_config: The config of dataset generator, defaults to None
        :type dataset_generator_config: Optional[DatasetGeneratorConfig], optional
        :return: The dataset to use for optimization.
        :rtype: RAGDataset
        """
        dataset_generator = SimpleDatasetGenerator.make(dataset_generator_config)
        return dataset_generator.generate_dataset(seed_data)

    def initialize_logger(self) -> None:
        """Initialize the logger."""
        self.logger = logging.getLogger(__name__)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        stream_handler.setLevel(logging.INFO)
        if self.params.log_to_file:
            log_file_path = self.experiment_repo.path_experiment / f"{self.experiment_repo.experiment_name}.log"
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("[INIT] Experiment repository Path: %s", self.experiment_repo.path_experiments)

    def initialize_experiment_repository(self) -> None:
        """Initialize the Optuna Experiment Repository."""
        self.experiment_repo = OptunaExperimentRepository(
            path_experiments=self.params.path_experiments,
            experiment_name=self.params.experiment_name,
        )

    def initialize_optuna_study(self) -> None:
        """Initialize the Optuna Study."""
        storage_name = self.experiment_repo.get_storage_name()
        self.manager = optuna.create_study(
            study_name=self.experiment_repo.experiment_name,
            storage=storage_name,
            direction=self.params.direction,
            pruner=self.pruner,
            load_if_exists=True,
            sampler=self.sampler,
        )
        self.logger.info("[INIT] Optuna Study Name: %s", self.manager.study_name)
        self.logger.debug("[INIT] Optuna Study Direction: %s", self.manager.direction)
        self.logger.debug("[INIT] Optuna Study Pruner: %s", self.manager.pruner)
        self.logger.debug("[INIT] Optuna Study Sampler: %s", self.manager.sampler)

    @abstractmethod
    def optimize(self) -> None:
        """Carry Out the optimization.

        :return: The optuna study that contains the experiment.
        :rtype: optuna.Study:
        """

    @abstractmethod
    def eval_trial(
        self,
        trial: optuna.trial.BaseTrial,
        dataset: RAGDataset,
    ) -> float:
        """Evaluate a RAG config on the dataset made by dataset_generator.

        :param trial: the RAG config to test
        :type trial: optuna.trial.BaseTrial
        :return: the dictionary containing the metrics
        :rtype: float
        """

    def _should_prune(self, trial: optuna.Trial, score: float) -> bool:
        return trial.should_prune() and score < self.manager.best_trial.value

    def sample_rag(self, trial: optuna.trial.BaseTrial, dataset: RAGDataset) -> RAG:
        """Sample RAG from trial.

        :param trial: Trial to use to sample the rag
        :type trial: optuna.trial.BaseTrial
        :return: The sampled RAG
        :rtype: RAG
        """
        config = self.config_space.sample(trial)
        self.logger.debug("[PROCESS] Current RAG Configuration Candidate: %s", config)
        rag_candidate = RAG.make(
            rag_config=config,
            prompt_config=self.prompt_config,
            inputs_chunks=[doc.text for doc in dataset.corpus.values()],
        )
        return rag_candidate

    def load_results(self) -> optuna.study.Study:
        """Load the results of the optimization.

        :return: The loaded study object.
        :rtype: optuna.study.Study
        """
        return self.experiment_repo.load_results()

    def get_n_best_trials(self, number: int) -> list[optuna.trial.FrozenTrial]:
        """Get the top N best trials from the optimization.

        :param number: The number of top trials to retrieve.
        :type number: int
        :return: A list of the top N best trials.
        :rtype: list[optuna.trial.FrozenTrial]
        """
        return self.experiment_repo.get_n_best_trials(number)

    def get_n_best_rag_configs(self, number: int) -> list[RAGConfig]:
        """Get the top N best RAGConfig objects from the optimization.

        :param number: The number of top RAGConfig objects to retrieve.
        :type number: int
        :return: A list of the top N best RAGConfig objects.
        :rtype: list[RAGConfig]
        """
        best_trials = self.get_n_best_trials(number)
        return [self.config_space.sample(trial) for trial in best_trials]

    def run_experiment(self) -> tuple[optuna.Study, RAGCandidateEval]:
        """Carry out the optimization and test the result.

        :return: The optimization and best result evaluation.
        :rtype: tuple[optuna.Study, RAGCandidateResult]
        """
        self.optimize()
        best_candidate_result = self.test()
        return self.manager, best_candidate_result

    def single_eval(
        self,
        evaluator: BaseEvaluator,
        eval_sample: EvalSample,
        rag_candidate: RAG,
    ) -> dict[str, Metric]:
        """Calculate and return the current score.

        :param evaluator: Evaluator used to evaluate output on sample.
        :type evaluator: BaseEvaluator
        :param eval_sample: The dataset for the evaluation.
        :type eval_sample: EvalSample
        :param rag_candidate: The RAG candidate.
        :type rag_candidate: RAG
        :return: The score of the current evaluation.
        :rtype: float
        :raise ValueError if the given context or the evaluation score is None.
        """
        self.logger.debug("[PROCESS] Eval sample: %s", eval_sample)
        self.logger.debug("[PROCESS] Query: %s", eval_sample.query)
        candidate = rag_candidate.get_rag_output(eval_sample.query)
        evaluation = evaluator.evaluate(candidate, eval_sample)
        self.logger.debug("[PROCESS] Evaluation Results: %s", evaluation)
        return evaluation

    def test(self) -> RAGCandidateEval:
        """Eval the best candidate of the optimization on the test set and save the result.

        :return: The config and score on train and test set.
        :rtype: RAGCandidateEval
        """
        best_trial = self.manager.best_trial
        best_rag_results = self.test_trial(best_trial)
        DataObject.save_to_json(best_rag_results, f"experiments/{self.params.experiment_name}/best_rag_results.json")
        return best_rag_results

    def test_trial(self, trial: optuna.trial.FrozenTrial) -> RAGCandidateEval:
        """Eval a trial on the test set.

        :return: The config and scores on train and test set.
        :rtype: RAGCandidateEval
        """
        self.logger.info("[PROCESS] Evaluating best trial on test set...")
        test_dataset = self.datasets["test"]
        config = self.config_space.sample(trial)
        rag = RAG.make(
            rag_config=config,
            prompt_config=self.prompt_config,
            inputs_chunks=[doc.text for doc in test_dataset.corpus.values()],
        )
        test_results: dict[str, list[float]] = defaultdict(list[float])
        for evaluator in self.test_evaluators:
            for test_sample in test_dataset.samples:
                single_eval = self.single_eval(evaluator, test_sample, rag)
                for metric_name, metric_value in single_eval.items():
                    test_results[metric_name].append(metric_value.score)

        return RAGCandidateEval(
            config=config,
            train_score=trial.value,
            mean_evals={m_name: np.mean(values).item() for m_name, values in test_results.items()},
            full_evals=test_results,
        )
