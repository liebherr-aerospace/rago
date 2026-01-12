"""Simple class for RAG Config optimization with direct evaluation method."""

from __future__ import annotations

import gc
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna

    from rago.data_objects.eval_sample import EvalSample
    from rago.dataset import RAGDataset
    from rago.model.wrapper.rag.base import RAG
from rago.data_objects import Metric
from rago.eval import BaseEvaluator

from .base import BaseOptunaManager, EvalMode


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
            lambda trial: self.eval_trial(trial, self.optim_evaluator, EvalMode.TRAIN),
            self.params.n_iter,
            show_progress_bar=self.params.show_progress_bar,
        )
        self.logger.info("[RESULT] Best trial %s", self.manager.best_trial)

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

    def _eval_trial(
        self,
        trial: optuna.trial.BaseTrial,
        dataset: RAGDataset,
        evaluator: BaseEvaluator,
        eval_mode: EvalMode = EvalMode.TRAIN,
    ) -> float | dict[str, Metric]:
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
            single_eval = self.single_eval(evaluator, test_sample, rag)
            trial_eval = {
                metric_name: Metric(evaluator.update_avg_score(trial_eval[metric_name].score, metric.score, n))
                for metric_name, metric in single_eval.items()
            }
            if eval_mode == EvalMode.TRAIN:
                score = single_eval[self.optim_metric_name].score
                self.logger.debug("[PROCESS] Mean score for current iteration: %s", score)
                trial.report(single_eval[self.optim_metric_name].score, n)
                if self._should_prune(trial, score):
                    score = trial_eval[self.optim_metric_name].score
                    self.logger.debug("[PROCESS] Pruning... return mean score: %s", score)
                    gc.collect()
                    return score
        gc.collect()
        return trial_eval if eval_mode == EvalMode.TEST else trial_eval[self.optim_metric_name].score
