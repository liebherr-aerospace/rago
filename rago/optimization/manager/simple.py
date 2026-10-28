"""Simple class for RAG Config optimization with direct evaluation method."""

from __future__ import annotations

import gc
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import optuna

if TYPE_CHECKING:

    from rago.dataset import RAGDataset

from rago.data_objects import Metric

from .base import BaseOptunaManager


class SimpleDirectOptunaManager(BaseOptunaManager):
    """A simple direct optimization manager calling Optuna to optimize parameters.

    The Optimization manager is instantiated with Optuna Study.
    """

    def _get_num_tested_trials(self) -> int:
        return sum(
        trial.state in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL}
        for trial in self.manager.trials
    )
    def _enqueue_stale_running_trials(self) -> None:
        for trial in self.manager.trials:
            if trial.state == optuna.trial.TrialState.RUNNING:
                self.manager.enqueue_trial(trial.params,skip_if_exists=False)

    def optimize(self) -> None:
        """Carry Out the optimization by simply call study.optimize.

        :return: The optuna study that contains the experiment.
        :rtype: optuna.Study
        """
        self.logger.info("[PROCESS] Starting Optimization...")
        self._enqueue_stale_running_trials()
        n_tested = self._get_num_tested_trials()
        n_trials = self.params.n_iter - n_tested
        if n_trials <= 0:
            self.logger.info(
                "[PROCESS] Study already has %s tested trials; target is %s.",
                n_tested,
                self.params.n_iter,
            )
            return
        self.manager.optimize(
            lambda trial: self.eval_trial(trial, self.datasets["train"]),
            n_trials,
            show_progress_bar=self.params.show_progress_bar,
            catch=(Exception,),
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
        rag = self.sample_model(trial, dataset)
        if len(self.manager.best_trials) > 0:
            self.logger.info(
                "[Trial %s] Best is trial %s",
                trial.number,
                self.manager.best_trial.number,
            )
            self.logger.info("[Trial %s] Best Value: %s", trial.number, self.manager.best_trial.values)
        trial_eval: dict[str, Metric] = defaultdict(Metric)
        for n, test_sample in enumerate(dataset.samples):
            self.logger.info("[Trial %s] Iteration %s",trial.number, n)
            start = time.time()
            single_eval = self.single_eval(self.optim_evaluator, test_sample, rag)
            trial_eval = {
                metric_name: Metric(
                    self.optim_evaluator.update_avg_score(trial_eval[metric_name].score, metric.score, n),
                )
                for metric_name, metric in single_eval.items()
            }
            end = time.time()
            mean_score = trial_eval[self.optim_metric_name].score
            self.logger.info("[Trial %s] Generation & evaluation time: %s", trial.number, end -start)
            self.logger.info("[Trial %s] Score: %s", trial.number, single_eval["correctness"].score)
            trial.report(single_eval[self.optim_metric_name].score, n)
            if self._should_prune(trial, mean_score):
                self.logger.debug("[Trial %s] Pruning...", trial.number)
                self._save_trial_metrics(trial, trial_eval)
                gc.collect()
                return mean_score

        gc.collect()
        self._save_trial_metrics(trial, trial_eval)
        return mean_score
