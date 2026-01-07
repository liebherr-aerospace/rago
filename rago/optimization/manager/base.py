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
from typing import TYPE_CHECKING, Optional

import optuna
from pydantic.dataclasses import dataclass

from rago.dataset.generator.simple import SeedDataType, SimpleDatasetGenerator
from rago.eval import BaseEvaluator
from rago.optimization.search_space.rag_config_space import RAGConfigSpace
from rago.prompts import PromptConfig

if TYPE_CHECKING:
    from rago.data_objects import RAGOutput
    from rago.dataset import RAGDataset
    from rago.dataset.generator import DatasetGeneratorConfig
    from rago.model.wrapper.rag.base import RAGConfig

from enum import StrEnum

from rago.data_objects import DataObject
from rago.optimization.repository.optuna_experiments_repository import OptunaExperimentRepository


class EvalMode(StrEnum):
    """Eval mode of the current evaluation."""

    TRAIN = "train"
    TEST = "Test"


@dataclass
class RAGCandidateResult(DataObject):
    """Performance and config of a rag on train and test set."""

    config: RAGConfig
    train_score: float
    test_score: float


@dataclass
class OptimParams:
    """Parameters of the optimization."""

    direction: str = "maximize"
    path_experiments: str = "experiments"
    experiment_name: str = "experiment_001"
    log_to_file: bool = True
    n_startup_trials: int = 50
    n_iter: int | None = None
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
        dataset: RAGDataset,
        evaluator: EvaluatorType,
        metric_name: str,
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
        :param evaluator: Evaluator used to evaluate RAG outputs.
        :type evaluator: BaseLLMEvaluator
        :param metric_name: Name of the metric to optimize if the evaluator returns a dict, defaults to None
        :type metric_name: Optional[str], optional
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
        self.dataset = dataset
        self.evaluator = evaluator
        self.metric_name = metric_name
        self.config_space = config_space if config_space is not None else RAGConfigSpace()
        self.prompt_config = prompt_config if prompt_config is not None else PromptConfig()
        self.sampler = sampler
        self.pruner = pruner
        self.chunks = [doc.text for doc in self.dataset.corpus.values()]
        self.initialize_experiment_repository()
        self.initialize_logger()
        self.initialize_optuna_study()

    @classmethod
    def from_seed_data(
        cls,
        *,
        params: OptimParams | None = None,
        seed_data: SeedDataType,
        dataset_generator_config: DatasetGeneratorConfig | None = None,
        evaluator: EvaluatorType,
        metric_name: str,
        config_space: Optional[RAGConfigSpace] = None,
        prompt_config: Optional[PromptConfig] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> BaseOptunaManager:
        """Initialize the Simple Direct Optimization Manager from seed data to generate dataset.

        :param params: Parameters of the optimization, defaults to None.
        :type params: Optional[OptimParams], optional
        :param seed_data: If no dataset is provided seed data used to generate a dataset, defaults to None.
        :type seed_data: Optional[RAGDataset], optional
        :param dataset_generator_config: Generator Configuration used to generate a dataset, defaults to None.
        :type dataset_generator_config: Optional[DatasetGeneratorConfig], optional
        :param evaluator: Evaluator used to evaluate RAG outputs.
        :type evaluator: BaseLLMEvaluator
        :param metric_name: Name of the metric to optimize if the evaluator returns a dict, defaults to None
        :type metric_name: Optional[str], optional
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
            dataset=cls.get_dataset(seed_data, dataset_generator_config),
            evaluator=evaluator,
            metric_name=metric_name,
            config_space=config_space,
            prompt_config=prompt_config,
            sampler=sampler,
            pruner=pruner,
        )

    @classmethod
    def get_dataset(
        cls,
        seed_data: Optional[SeedDataType] = None,
        dataset_generator_config: DatasetGeneratorConfig | None = None,
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
    def objective(self, trial: optuna.trial.BaseTrial, eval_mode: EvalMode = EvalMode.TRAIN) -> float:
        """Evaluate a RAG config on the dataset made by dataset_generator.

        :param trial: the RAG config to test
        :type trial: optuna.trial.BaseTrial
        :return: the dictionary containing the metrics
        :rtype: float
        """

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

    def run_experiment(self) -> tuple[optuna.Study, RAGCandidateResult]:
        """Carry out the optimization and test the result.

        :return: The optimization and best result evaluation.
        :rtype: tuple[optuna.Study, RAGCandidateResult]
        """
        self.optimize()
        best_candidate_result = self.test()
        return self.manager, best_candidate_result

    def test(self) -> RAGCandidateResult:
        """Eval the best candidate of the optimization on the test set.

        :return: The config and score on train and test set.
        :rtype: RAGCandidateResult
        """
        self.logger.info("[PROCESS] Evaluating best trial on test set...")
        best_trial = self.manager.best_trial
        config = self.config_space.sample(best_trial)
        best_rag_test_score = self.objective(best_trial, eval_mode=EvalMode.TEST)
        best_rag_results = RAGCandidateResult(
            config=config,
            train_score=best_trial.value,
            test_score=best_rag_test_score,
        )
        DataObject.save_to_json(best_rag_results, f"experiments/{self.params.experiment_name}/best_rag_results.json")
        return best_rag_results
