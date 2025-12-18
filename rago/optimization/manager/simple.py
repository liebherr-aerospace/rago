"""Simple class for RAG Config optimization with direct evaluation method."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import optuna

    from rago.data_objects import Metric, RAGOutput
    from rago.data_objects.eval_sample import EvalSample
    from rago.dataset import RAGDataset
    from rago.dataset.generator import DatasetGeneratorConfig
    from rago.eval import BaseEvaluator
    from rago.optimization.search_space.rag_config_space import RAGConfigSpace
from rago.dataset.generator.simple import SeedDataType, SimpleDatasetGenerator
from rago.model.wrapper.rag.base import RAG
from rago.prompts import PromptConfig

from .base import BaseOptunaManager, OptimParams


class SimpleDirectOptunaManager(BaseOptunaManager):
    """A simple direct optimization manager calling Optuna to optimize parameters.

    The Optimization manager is instantiated with Optuna Study.
    """

    @classmethod
    def from_seed_data(
        cls,
        *,
        params: Optional[OptimParams] = None,
        seed_data: SeedDataType,
        dataset_generator_config: Optional[DatasetGeneratorConfig] = None,
        evaluator: BaseEvaluator[RAGOutput],
        metric_name: str,
        config_space: Optional[RAGConfigSpace] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> SimpleDirectOptunaManager:
        """Initialize the Simple Direct Optimization Manager from seed data to generate dataset.

        :param evaluator: Evaluator used to evaluate RAG outputs.
        :type evaluator: BaseEvaluator[RAGOutput, EvaluatorOutputType]
        :param params: Parameters of the optimization, defaults to None.
        :type params: Optional[OptimParams], optional
        :param seed_data: If no dataset is provided seed data used to generate a dataset, defaults to None.
        :type seed_data: Optional[RAGDataset], optional
        :param dataset_generator_config: Generator Configuration used to generate a dataset, defaults to None.
        :type dataset_generator_config: Optional[DatasetGeneratorConfig], optional
        :param metric_name: Name of the metric to optimize if the evaluator returns a dict, defaults to None
        :type metric_name: Optional[str], optional
        :param config_space: The space of RAG config to search in, defaults to None
        :type config_space: Optional[RAGConfigSpace], optional
        :param sampler: The sampler used to suggest new rag configuration to tests, defaults to None
        :type sampler: Optional[optuna.samplers.BaseSampler], optional
        :param pruner: The pruner used to terminate early unpromising trials, defaults to None
        :type pruner: Optional[optuna.pruners.BasePruner], optional
        """
        return cls(
            params=params,
            seed_data=seed_data,
            dataset_generator_config=dataset_generator_config,
            evaluator=evaluator,
            metric_name=metric_name,
            config_space=config_space,
            sampler=sampler,
            pruner=pruner,
        )

    @classmethod
    def from_dataset(
        cls,
        *,
        params: Optional[OptimParams] = None,
        dataset: RAGDataset,
        evaluator: BaseEvaluator[RAGOutput],
        metric_name: str,
        config_space: Optional[RAGConfigSpace] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> SimpleDirectOptunaManager:
        """Initialize the Simple Direct Optimization Manager from a dataset.

        :param evaluator: Evaluator used to evaluate RAG outputs.
        :type evaluator: BaseEvaluator[RAGOutput, EvaluatorOutputType]
        :param params: Parameters of the optimization, defaults to None.
        :type params: Optional[OptimParams], optional
        :param dataset: Dataset to use, defaults to None.
        :type dataset: Optional[RAGDataset], optional
        :param metric_name: Name of the metric to optimize if the evaluator returns a dict, defaults to None
        :type metric_name: Optional[str], optional
        :param config_space: The space of RAG config to search in, defaults to None
        :type config_space: Optional[RAGConfigSpace], optional
        :param sampler: The sampler used to suggest new rag configuration to tests, defaults to None
        :type sampler: Optional[optuna.samplers.BaseSampler], optional
        :param pruner: The pruner used to terminate early unpromising trials, defaults to None
        :type pruner: Optional[optuna.pruners.BasePruner], optional
        """
        return cls(
            params=params,
            dataset=dataset,
            evaluator=evaluator,
            metric_name=metric_name,
            config_space=config_space,
            sampler=sampler,
            pruner=pruner,
        )

    def __init__(
        self,
        *,
        params: Optional[OptimParams] = None,
        dataset: Optional[RAGDataset] = None,
        seed_data: Optional[SeedDataType] = None,
        dataset_generator_config: Optional[DatasetGeneratorConfig] = None,
        evaluator: BaseEvaluator[RAGOutput],
        metric_name: str,
        config_space: Optional[RAGConfigSpace] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> None:
        """Initialize the Simple Direct Optimization Manager.

        :param evaluator: Evaluator used to evaluate RAG outputs.
        :type evaluator: BaseEvaluator[RAGOutput, EvaluatorOutputType]
        :param params: Parameters of the optimization, defaults to None.
        :type params: Optional[OptimParams], optional
        :param dataset: Dataset to use, defaults to None.
        :type dataset: Optional[RAGDataset], optional
        :param seed_data: If no dataset is provided seed data used to generate a dataset, defaults to None.
        :type seed_data: Optional[RAGDataset], optional
        :param dataset_generator_config: Generator Configuration used to generate a dataset, defaults to None.
        :type dataset_generator_config: Optional[DatasetGeneratorConfig], optional
        :param metric_name: Name of the metric to optimize if the evaluator returns a dict, defaults to None
        :type metric_name: Optional[str], optional
        :param config_space: The space of RAG config to search in, defaults to None
        :type config_space: Optional[RAGConfigSpace], optional
        :param sampler: The sampler used to suggest new rag configuration to tests, defaults to None
        :type sampler: Optional[optuna.samplers.BaseSampler], optional
        :param pruner: The pruner used to terminate early unpromising trials, defaults to None
        :type pruner: Optional[optuna.pruners.BasePruner], optional
        """
        super().__init__(
            params=params,
            config_space=config_space,
            sampler=sampler,
            pruner=pruner,
        )
        self.prompt_config = PromptConfig()
        self.logger.info("[INIT] Evaluator...")
        self.evaluator = evaluator
        self.metric_name = metric_name
        self.logger.debug("[INIT] LLM Evaluator %s", self.evaluator)
        self.logger.info("[INIT] Dataset Generator...")
        self.dataset = self.get_dataset(dataset, seed_data, dataset_generator_config)
        self.chunks = [doc.text for doc in self.dataset.corpus.values()]
        self.logger.info("[INIT] Get Eval Samples %s", self.dataset)

    def sample_rag(self, trial: optuna.trial.BaseTrial) -> RAG:
        """Sample RAG from trial.

        :param trial: Trial to use to sample the rag
        :type trial: optuna.trial.BaseTrial
        :return: The sampled RAG
        :rtype: RAG
        """
        config = self.config_space.sample(trial)
        self.logger.debug("[PROCESS] Current RAG Configuration Candidate: %s", config)
        rag_candidate = RAG.make(rag_config=config, prompt_config=self.prompt_config, inputs_chunks=self.chunks)
        return rag_candidate

    def get_dataset(
        self,
        dataset: Optional[RAGDataset] = None,
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
        if dataset is not None:
            return dataset
        self.dataset_generator = SimpleDatasetGenerator.make(dataset_generator_config)
        return self.dataset_generator.generate_dataset(seed_data)

    def optimize(self) -> optuna.Study:
        """Carry Out the optimization by simply call study.optimize.

        :return: The optuna study that contains the experiment.
        :rtype: optuna.Study
        """
        self.logger.info("[PROCESS] Starting Optimization...")
        self.manager.optimize(self.objective, self.params.n_iter, show_progress_bar=self.params.show_progress_bar)
        self.logger.info("[RESULT] Best trial %s", self.manager.best_trial)
        return self.manager

    def _get_score(self, evaluation: dict[str, Metric]) -> float:
        if self.metric_name is None:
            raise ValueError
        score = evaluation[self.metric_name].score
        if score is None:
            raise ValueError
        return score

    def eval(
        self,
        eval_sample: EvalSample,
        rag_candidate: RAG,
    ) -> float:
        """Calculate and return the current score.

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
        evaluation = self.evaluator.evaluate(candidate, eval_sample)
        self.logger.debug("[PROCESS] Evaluation Results: %s", evaluation)
        return self._get_score(evaluation)

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna to optimize.

        :param trial: The RAG config to test.
        :type trial: optuna.Trial
        :return: The mean score.
        :rtype: float
        """
        rag = self.sample_rag(trial)
        self.logger.info("[PROCESS] Trial %s", trial.number)
        if len(self.manager.best_trials) > 0:
            self.logger.info(
                "[PROCESS] Best is trial %s",
                self.manager.best_trial.number,
            )
            self.logger.info("[PROCESS] Best value: %s", self.manager.best_trial.values)
        score = 0.0
        for n, test_sample in enumerate(self.dataset.samples):
            self.logger.debug("[PROCESS] Iteration: %s", n)
            score_eval = self.eval(test_sample, rag)
            trial.report(score_eval, n)
            score = self.evaluator.update_avg_score(score, score_eval, n)
            if self.pruner is not None and trial.should_prune():
                self.logger.debug("[PROCESS] Pruning... return mean score: %s", score)
                gc.collect()
                return score
            self.logger.debug("[PROCESS] Mean score for current iteration: %s", score)
        self.logger.debug("[PROCESS] Final Mean score: %s", score)
        gc.collect()

        return score
