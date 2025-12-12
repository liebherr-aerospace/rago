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

from rago.optimization.search_space.rag_config_space import RAGConfigSpace

if TYPE_CHECKING:
    from rago.data_objects.rag_config import RAGConfig

from rago.optimization.repository.optuna_experiments_repository import OptunaExperimentRepository


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


class BaseOptunaManager(ABC):
    """An Abstract Class that defines the optimization manager."""

    manager: optuna.Study
    config_space: RAGConfigSpace
    logger: logging.Logger

    def __init__(
        self,
        *,
        params: Optional[OptimParams] = None,
        config_space: Optional[RAGConfigSpace] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> None:
        """Optimization Manager initialization.

        :param sampler: The sampler used for the optimization algorithm.
        :type sampler: optuna.samplers.BaseSampler | None
        :param pruner: The pruner method to stop the current optimization iteration.
        :type pruner: optuna.pruners.BasePruner | None
        """
        self.params = params if params is not None else OptimParams()
        self.config_space = config_space if config_space is not None else RAGConfigSpace()
        self.sampler = sampler
        self.pruner = pruner
        self.initialize_experiment_repository()
        self.initialize_logger()
        self.initialize_optuna_study()

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
    def optimize(self) -> optuna.Study:
        """Carry Out the optimization.

        :return: The optuna study that contains the experiment.
        :rtype: optuna.Study:
        """

    @abstractmethod
    def objective(self, trial: optuna.Trial) -> float:
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
        return [self.experiment_repo.convert_trial_to_rag_config(trial) for trial in best_trials]
