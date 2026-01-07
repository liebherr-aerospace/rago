"""Simple class for RAG Config optimization with direct evaluation method."""

from __future__ import annotations

import gc
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    import optuna

    from rago.data_objects import Metric
    from rago.data_objects.eval_sample import EvalSample
from rago.data_objects import DataObject
from rago.eval import BaseEvaluator
from rago.model.wrapper.rag.base import RAG, RAGConfig

from .base import BaseOptunaManager


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


class SimpleDirectOptunaManager(BaseOptunaManager[BaseEvaluator]):
    """A simple direct optimization manager calling Optuna to optimize parameters.

    The Optimization manager is instantiated with Optuna Study.
    """

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

    def optimize(self) -> tuple[optuna.Study, RAGCandidateResult]:
        """Carry Out the optimization by simply call study.optimize.

        :return: The optuna study that contains the experiment.
        :rtype: optuna.Study
        """
        self.logger.info("[PROCESS] Starting Optimization...")
        self.manager.optimize(self.objective, self.params.n_iter, show_progress_bar=self.params.show_progress_bar)
        self.logger.info("[RESULT] Best trial %s", self.manager.best_trial)
        self.logger.info("[PROCESS] Evaluating best trial on test set...")
        test_result = self.test()
        return self.manager, test_result

    def _get_score(self, evaluation: dict[str, Metric]) -> float:
        if self.metric_name is None:
            raise ValueError
        score = evaluation[self.metric_name].score
        if score is None:
            raise ValueError
        return score

    def single_eval(
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

    def objective(self, trial: optuna.trial.BaseTrial, eval_mode: EvalMode = EvalMode.TRAIN) -> float:
        """Objective function for Optuna to optimize.

        :param trial: The RAG config to test.
        :type trial: optuna.Trial
        :param eval_mode: wether we are in the training or test phase, defaults to EVAL_MODE.TRAIN.
        :type eval_mode: EVAL_MODE
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
            score_eval = self.single_eval(test_sample, rag)
            if eval_mode == EvalMode.TRAIN:
                trial.report(score_eval, n)
            score = self.evaluator.update_avg_score(score, score_eval, n)
            if eval_mode == EvalMode.TRAIN and self.pruner is not None and trial.should_prune():
                self.logger.debug("[PROCESS] Pruning... return mean score: %s", score)
                gc.collect()
                return score
            self.logger.debug("[PROCESS] Mean score for current iteration: %s", score)
        self.logger.debug("[PROCESS] Final Mean score: %s", score)
        gc.collect()

        return score

    def test(self) -> RAGCandidateResult:
        """Eval the best candidate of the optimization on the test set.

        :return: The config and score on train and test set.
        :rtype: RAGCandidateResult
        """
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
