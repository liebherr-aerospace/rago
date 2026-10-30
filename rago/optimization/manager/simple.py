"""Simple class for model config optimization with direct evaluation method."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

if TYPE_CHECKING:

    import optuna

    from rago.dataset import RAGDataset

from rago.data_objects import EvalSample, Metric, ModelInput, ModelOutput
from rago.eval import BaseEvaluator, EvaluatorReference
from rago.model.configs.tunable_model_config import TunableModelConfig

from .base import BaseOptunaManager


class SimpleDirectOptunaManager[
    ModelInputT: ModelInput,
    ModelOutputT: ModelOutput, EvaluatorReferenceT: EvaluatorReference, ConfigT: TunableModelConfig](
    BaseOptunaManager[ModelInputT, ModelOutputT, EvaluatorReferenceT, ConfigT]):
    """A simple direct optimization manager calling Optuna to optimize parameters.

    The Optimization manager is instantiated with Optuna Study.
    """

    def optimize(self) -> None:
        """Carry Out the optimization by simply call study.optimize."""
        self.logger.info("[PROCESS] Starting Optimization...")

        self.manager.optimize(
            lambda trial: self.eval_trial(trial, self.datasets["train"]),
            self.params.n_iter,
            show_progress_bar=self.params.show_progress_bar,
            catch=(Exception,),
        )
        self.logger.info("[RESULT] Best trial %s", self.manager.best_trial)

    def get_evaluator_reference[ReferenceT: EvaluatorReference](
            self, evaluator: BaseEvaluator[ModelOutputT, ReferenceT], eval_sample: EvalSample) -> ReferenceT:
        """Get the evaluator reference from eval sample and manager internal state.

        :param eval_sample: The eval_sample to use as reference.
        :type eval_sample: EvalSample.
        :return: The reference to pass to the evaluator.
        :rtype: EvaluatorReferenceT
        """
        return evaluator.get_evaluator_reference(eval_sample)

    def eval_trial(
        self,
        trial: optuna.trial.BaseTrial,
        dataset: RAGDataset,
    ) -> float:
        """Evaluate a model config on the dataset made by dataset_generator.

        :param trial: The model config to test
        :type trial: optuna.trial.BaseTrial
        :return: The dictionary containing the metrics
        :rtype: float
        """
        model = self.sample_model(trial, dataset)
        self.logger.info("[PROCESS] Trial %s", trial.number)
        if len(self.manager.best_trials) > 0:
            self.logger.info(
                "[PROCESS] Best is trial %s",
                self.manager.best_trial.number,
            )
            self.logger.info("[PROCESS] Best value: %s", self.manager.best_trial.values)
        mean_score = 0
        metrics: list[Metric] = []
        for step, (idx, test_sample) in enumerate(self.get_samples_and_ids(trial, dataset.samples)):
            self.logger.debug("[PROCESS] Iteration: %s", step)
            sample_metric = self.single_eval(self.optim_evaluator, test_sample, model)
            metrics.append(sample_metric)
            mean_score = self.optim_evaluator.update_avg_score(mean_score, sample_metric.score, step)
            self.logger.debug("[PROCESS] Mean score for current iteration: %s", mean_score)
            self.report(trial, sample_metric.score, idx, mean_score)
            if self._should_prune(trial, mean_score):
                self.logger.debug("[PROCESS] Pruning...")
                self._save_trial_metrics(trial, metrics)
                gc.collect()
                return mean_score

        gc.collect()
        self._save_trial_metrics(trial, metrics)
        return mean_score
