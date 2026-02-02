"""Simple class for RAG Config optimization with direct evaluation method."""

from __future__ import annotations

import gc
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna

    from rago.dataset import RAGDataset
from rago.data_objects import Metric
from rago.eval import BaseEvaluator

from .base import BaseOptunaManager


class SimpleDirectOptunaManager(BaseOptunaManager[BaseEvaluator]):
    """A simple direct optimization manager calling Optuna to optimize parameters.

    The Optimization manager is instantiated with Optuna Study.
    """

    def optimize(self) -> None:
        """Carry Out the optimization by simply call study.optimize.

        :return: The optuna study that contains the experiment.
        :rtype: optuna.Study
        """
        self.logger.info("[PROCESS] Starting Optimization...")

        self.manager.optimize(
            lambda trial: self.eval_trial(trial, self.datasets["train"]),
            self.params.n_iter,
            show_progress_bar=self.params.show_progress_bar,
        )
        self.logger.info("[RESULT] Best trial %s", self.manager.best_trial)

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
        rag = self.sample_rag(trial, dataset)
        self.logger.info("[PROCESS] Trial %s", trial.number)
        if len(self.manager.best_trials) > 0:
            self.logger.info(
                "[PROCESS] Best is trial %s",
                self.manager.best_trial.number,
            )
            self.logger.info("[PROCESS] Best value: %s", self.manager.best_trial.values)
        trial_eval: dict[str, Metric] = defaultdict(Metric)
        for n, test_sample in enumerate(dataset.samples):
            self.logger.debug("[PROCESS] Iteration: %s", n)
            single_eval = self.single_eval(self.optim_evaluator, test_sample, rag)
            trial_eval = {
                metric_name: Metric(
                    self.optim_evaluator.update_avg_score(trial_eval[metric_name].score, metric.score, n),
                )
                for metric_name, metric in single_eval.items()
            }
            mean_score = trial_eval[self.optim_metric_name].score
            self.logger.debug("[PROCESS] Mean score for current iteration: %s", mean_score)
            trial.report(single_eval[self.optim_metric_name].score, n)
            if self._should_prune(trial, mean_score):
                self.logger.debug("[PROCESS] Pruning...")
                self._save_trial_metrics(trial, trial_eval)
                gc.collect()
                return mean_score

        gc.collect()
        self._save_trial_metrics(trial, trial_eval)
        return mean_score
