"""Simple class for RAG Config optimization with pair-wise evaluation method."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import optuna

    from rago.data_objects import Metric, RAGOutput
    from rago.data_objects.eval_sample import EvalSample
from rago.model.wrapper.rag.base import RAG
from rago.optimization.manager.simple import SimpleDirectOptunaManager


class SimplePairWiseOptunaManager(SimpleDirectOptunaManager):
    """A simple direct optimization manager calling Optuna to optimize Reader parameters.

    The Optimization manager is instantiated with Optuna Study.
    """

    def optimize(self) -> optuna.Study:
        """Carry Out the optimization by simply call study.optimize.

        :return: The optuna study that contains the experiment.
        :rtype:  optuna.Study
        """
        self.logger.info("[PROCESS] Starting Optimization...")
        self.run_initial_trials()
        self.manager.optimize(self.objective, self.params.n_iter)
        self.logger.info("[RESULT] Best trial %s", self.manager.best_trial)
        return self.manager

    def run_initial_trials(
        self,
    ) -> None:
        """Initialize the optimization by getting the first best RAG configuration."""
        self.logger.info("[INIT OPTIM] Run First Round of the Battle RAG...")
        rag_candidate_1, trial1 = self.instantiate_simple_rag()
        rag_candidate_2, trial2 = self.instantiate_simple_rag()
        answer_candidate_1: list[str] = []
        answer_candidate_2: list[str] = []
        score_candidate_1: list[float] = []
        score_candidate_2: list[float] = []
        for n, test_sample in enumerate(self.dataset.samples):
            self.logger.debug("[INIT OPTIM] Iteration %s", n)
            if test_sample.context is not None:
                self.logger.debug("[INIT OPTIM] Test Eval Sample: %s", test_sample)
                rag_output_1 = rag_candidate_1.get_rag_output(test_sample.query)
                rag_output_2 = rag_candidate_2.get_rag_output(test_sample.query)
                evaluation_1, evaluation_2 = self.evaluator.evaluate_pairwise(
                    rag_output_1,
                    rag_output_2,
                    test_sample,
                )
                answer_candidate_1, score_candidate_1 = self.update_answers_scores(
                    rag_output_1,
                    evaluation_1,
                    answer_candidate_1,
                    score_candidate_1,
                )
                answer_candidate_2, score_candidate_2 = self.update_answers_scores(
                    rag_output_2,
                    evaluation_2,
                    answer_candidate_2,
                    score_candidate_2,
                )
        list_trial = [trial1, trial2]
        list_best_answer = [answer_candidate_1, answer_candidate_2]
        list_scores = [score_candidate_1, score_candidate_2]
        best_trial, best_answer, best_scores = self.get_best_candidate(
            list_trial,
            list_best_answer,
            list_scores,
        )
        self.logger.info("[INIT OPTIM] Best Answer: %s", best_answer)
        self.logger.info("[INIT OPTIM] Best Score: %s", best_scores)
        self.logger.debug(
            "[INIT OPTIM] Eval Sample Before Update: %s",
            self.dataset.samples,
        )
        if (best_answer is not None) and (best_scores is not None):
            self.dataset.samples = self.update_eval_sample_reference(
                best_answer,
                best_scores,
            )
            self.logger.debug("[INIT OPTIM] Eval Sample Updated: %s", self.dataset.samples)
            mean_best_score = np.mean(best_scores)
            self.manager.tell(best_trial, float(mean_best_score))

    def instantiate_simple_rag(self) -> tuple[RAG, optuna.Trial]:
        """Instantiate the simple RAG.

        :return: The RAG candidate and the trial
        :rtype: tuple(RAG, optuna.Trial)
        """
        trial = self.manager.ask()
        config = self.config_space.sample(trial)
        self.logger.info("[PROCESS] Current Config: %s", config)
        rag_candidate = RAG.make(rag_config=config, prompt_config=self.prompt_config, inputs_chunks=self.chunks)
        return (rag_candidate, trial)

    def update_answers_scores(
        self,
        rag_output: RAGOutput,
        evaluation: dict[str, Metric],
        answers: list[str],
        scores: list[float],
    ) -> tuple[list[str], list[float]]:
        """Update the list of the candidate answers and scores and log it.

        :param rag_output: The candidate rag output
        :type rag_output: RagOutput
        :param evaluation: The list of the candidate evaluation
        :type evaluation: dict[str, Metric]
        :param answers: The list of the candidate answers
        :type answers: list[str]
        :param scores: The list of the candidate scores
        :type scores: list[float]
        """
        self.logger.debug("[INIT OPTIM] Candidate RAG Output: %s", rag_output)
        self.logger.debug("[INIT OPTIM] Candidate Evaluation: %s", evaluation)
        if rag_output.answer is not None:
            answers.append(rag_output.answer)
        self.logger.debug("[INIT OPTIM] Candidate All Answers: %s", answers)

        score_to_add = self._get_score(evaluation)
        if score_to_add is not None:
            scores.append(score_to_add)
        self.logger.debug("[INIT OPTIM] Candidate All Scores: %s", scores)
        return answers, scores

    def get_best_candidate(
        self,
        trial: list[optuna.Trial],
        answers: list[list[str]],
        scores: list[list[float]],
    ) -> tuple[optuna.Trial, list[str], list[float]]:
        """Calculate the average scores for each RAG output and return the best trial, answer and scores.

        :param answers: The list of the answers
        :type answers: list[str]
        :param scores: The list of lists of scores associated with each RAG output.
        :type scores: list[list[float]]
        :return: A tuple containing the trial, answers, its associated scores.
        :rtype: tuple[optuna.Trial, list[str], list[float]]
        """
        avg_scores = [np.mean(score_list) for score_list in scores]
        best_index = np.argmax(avg_scores)
        self.logger.debug("[INIT OPTIM] Best index: %s", best_index)
        return trial[best_index], answers[best_index], scores[best_index]

    def update_eval_sample_reference(
        self,
        answers: Sequence[Optional[str]],
        scores: Sequence[Optional[float]],
    ) -> list[EvalSample]:
        """Update the eval sample answer and scores references.

        :param answers: The list of answers to populate the answer reference of the eval samples.
        :type answers: list[Optional[str]],
        :param scores: The list of scores to populate the reference score of the eval samples.
        :type scores: list[Optional[float]]
        :return: The list of populated Sample instances.
        :rtype: list[Sample]
        """
        for sample, answer, score in zip(self.dataset.samples, answers, scores, strict=False):
            sample.reference_answer = answer
            sample.reference_score = score
        return self.dataset.samples

    def check_update_eval_sample_reference(
        self,
        answers: Sequence[Optional[str]],
        scores: Sequence[Optional[float]],
        score: float,
        trial: optuna.Trial,
    ) -> list[EvalSample]:
        """Check and update the eval sample answer and scores references if current candidate is better.

        :param answers: The list of answers to populate the answer reference of the eval samples.
        :type answers: list[Optional[str]],
        :param scores: The list of scores to populate the reference score of the eval samples.
        :type scores: list[Optional[float]]
        :param score: The current score
        :type score: float
        :param trial: The RAG config to test.
        :type trial: optuna.Trial
        :return: The list of populated Sample instances.
        :rtype: list[Sample]
        """
        if (score > self.manager.best_trial.values[0]) and (trial.number > 0.0):
            self.update_eval_sample_reference(answers, scores)
            self.logger.debug("[PROCESS] Eval Sample Update: %s", self.dataset.samples)
        return self.dataset.samples

    def get_current_score_answer(
        self,
        eval_sample: EvalSample,
        rag_candidate: RAG,
    ) -> tuple[str, float]:
        """Get the current score and answer.

        :param eval_sample: The dataset for the evaluation.
        :type eval_sample: EvalSample
        :param rag_candidate: The RAG candidate.
        :type rag_candidate: RAG
        :return: The score of the current evaluation.
        :rtype: tuple[str, float]
        :raise ValueError if the given context or the score or answer is None.
        """
        self.logger.debug("[PROCESS] Eval sample: %s", eval_sample)
        if eval_sample.context is not None:
            candidate = rag_candidate.get_rag_output(eval_sample.query)
            answer = candidate.answer
            evaluation = self.evaluator.evaluate(candidate, eval_sample)
            self.logger.debug("[PROCESS] Evaluation Results: %s", evaluation)
            score_eval = self._get_score(evaluation)
            if (score_eval is not None) and (answer is not None):
                return answer, score_eval
        msg_error = "None values found."
        raise ValueError(msg_error)

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna to optimize.

        :param trial: The RAG config to test.
        :type trial: optuna.Trial
        :return: The mean score.
        :rtype: float
        """
        config = self.config_space.sample(trial)
        rag_candidate = RAG.make(rag_config=config, prompt_config=self.prompt_config, inputs_chunks=self.chunks)
        self.logger.info("[PROCESS] Trial %s", trial.number)
        if trial.number > 0:
            self.logger.info(
                "[PROCESS] Best is trial %s",
                self.manager.best_trial.number,
            )
            self.logger.info("[PROCESS] Best Value: %s", self.manager.best_trial.values)
        self.logger.info("[PROCESS] Current Config: %s", config)
        score = 0.0
        score_list = []
        answer_list = []
        for n, test_sample in enumerate(self.dataset.samples):
            self.logger.debug("[PROCESS] Iteration %s", n)
            answer_eval, score_eval = self.get_current_score_answer(test_sample, rag_candidate)
            answer_list.append(answer_eval)
            score_list.append(score_eval)
            trial.report(score_eval, n)
            score = self.evaluator.update_avg_score(score, score_eval, n)
            if trial.should_prune():
                self.logger.debug("[PROCESS] Pruning... return mean score: %s", score)
                self.check_update_eval_sample_reference(answer_list, score_list, score, trial)
                return score
            self.logger.debug("[PROCESS] Mean score: %s", score)
        self.check_update_eval_sample_reference(answer_list, score_list, score, trial)
        gc.collect()
        return score
