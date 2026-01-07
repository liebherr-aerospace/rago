"""Simple class for RAG Config optimization with pair-wise evaluation method."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Optional

import numpy as np
import optuna

from rago.data_objects import RAGOutput
from rago.prompts import PromptConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    import optuna

    from rago.data_objects import Metric, RAGOutput
    from rago.data_objects.eval_sample import EvalSample
    from rago.dataset import RAGDataset
    from rago.optimization.search_space.rag_config_space import RAGConfigSpace
from rago.data_objects import PromptTemplate
from rago.eval import BaseLLMEvaluator, EvalPrompts, SimpleLLMEvaluator
from rago.model.wrapper.llm_agent import LangchainLLMAgent
from rago.model.wrapper.rag.base import RAG
from rago.optimization.manager.base import BaseOptunaManager, OptimParams
from rago.optimization.manager.simple import EvalMode
from rago.prompts import DEFAULT_EVAL_PROMPT, DEFAULT_REFERENCE_EVAL_PROMPT


class SimplePairWiseOptunaManager(BaseOptunaManager[BaseLLMEvaluator]):
    """A simple direct optimization manager calling Optuna to optimize Reader parameters.

    The Optimization manager is instantiated with Optuna Study.
    """

    def __init__(
        self,
        *,
        params: Optional[OptimParams] = None,
        dataset: RAGDataset,
        evaluator: Optional[BaseLLMEvaluator] = None,
        metric_name: str = "correctness",
        config_space: Optional[RAGConfigSpace] = None,
        prompt_config: Optional[PromptConfig] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        num_initial_rags: int = 2,
        test_prompt: str = DEFAULT_EVAL_PROMPT,
    ) -> None:
        """Initialize the Simple Direct Optimization Manager from seed data to generate dataset.

        :param params: Parameters of the optimization, defaults to None.
        :type params: Optional[OptimParams], optional
        :param dataset: Dataset on which rag configs will be evaluated.
        :type dataset: RAGDataset
        :param evaluator: Evaluator used to evaluate RAG outputs, defaults to None.
        :type evaluator: Optional[BaseLLMEvaluator], optional
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
        :param num_initial_rags: Num of RAG to test initially (must be compatible with the evaluator's pairwise prompt).
        :type num_initial_rags: int = 2
        :param test_prompt: Prompt to use for testing, defaults to DEFAULT_EVAL_PROMPT.
        :type test_prompt: str
        """
        self.num_initial_trials = num_initial_rags
        self.test_prompt = PromptTemplate(test_prompt).get_partially_formatted_prompt_template(
            min_score=str(self.evaluator.min_score),
            max_score=str(self.evaluator.max_score),
        )
        if evaluator is None:
            evaluator = SimpleLLMEvaluator(
                judge=LangchainLLMAgent.make_from_backend(),
                eval_prompts=EvalPrompts(DEFAULT_REFERENCE_EVAL_PROMPT),
            )

        self.train_prompt = self.evaluator.eval_prompt
        super().__init__(
            params=params,
            dataset=dataset,
            evaluator=evaluator,
            metric_name=metric_name,
            config_space=config_space,
            prompt_config=prompt_config,
            sampler=sampler,
            pruner=pruner,
        )

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
        rags, trials = self.instantiate_simple_rags()
        answers_candidates: list[list[str]] = [[] for _ in range(self.num_initial_trials)]
        scores: list[list[float]] = [[] for _ in range(self.num_initial_trials)]
        for n, test_sample in enumerate(self.dataset.samples):
            self.logger.debug("[INIT OPTIM] Iteration %s", n)
            if test_sample.context is not None:
                self.logger.debug("[INIT OPTIM] Test Eval Sample: %s", test_sample)
                rag_outputs = [rag_cand.get_rag_output(test_sample.query) for rag_cand in rags]
                evaluations = self.evaluator.evaluate_n_wise(
                    rag_outputs,
                    test_sample,
                )

                answers_candidates, scores = self.update_answers_scores(
                    rag_outputs,
                    evaluations,
                    answers_candidates,
                    scores,
                )
        best_trial, best_answer, best_scores = self.get_best_candidate(
            trials,
            answers_candidates,
            scores,
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

    def instantiate_simple_rags(self) -> tuple[list[RAG], list[optuna.Trial]]:
        """Instantiate the simple RAG.

        :return: The RAG candidate and the trial
        :rtype: tuple(RAG, optuna.Trial)
        """
        trials: list[optuna.Trial] = []
        rags: list[RAG] = []
        for _ in range(self.num_initial_trials):
            trials.append(self.manager.ask())
            config = self.config_space.sample(trials[-1])
            self.logger.info("[PROCESS] Current Config: %s", config)
            rags.append(RAG.make(rag_config=config, prompt_config=self.prompt_config, inputs_chunks=self.chunks))
        return (rags, trials)

    def _get_score(self, evaluation: dict[str, Metric]) -> float:
        if self.metric_name is None:
            raise ValueError
        score = evaluation[self.metric_name].score
        if score is None:
            raise ValueError
        return score

    def update_answers_scores(
        self,
        rag_output: list[RAGOutput],
        evaluation: list[dict[str, Metric]],
        answers: list[list[str]],
        scores: list[list[float]],
    ) -> tuple[list[list[str]], list[list[float]]]:
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
        for idx, (out, e) in enumerate(zip(rag_output, evaluation, strict=False)):
            self.logger.debug("[INIT OPTIM] Candidate RAG Output: %s", rag_output)
            self.logger.debug("[INIT OPTIM] Candidate Evaluation: %s", evaluation)
            if out.answer is not None:
                answers[idx].append(out.answer)
            self.logger.debug("[INIT OPTIM] Candidate All Answers: %s", answers)
            score_to_add = self._get_score(e)
            if score_to_add is not None:
                scores[idx].append(score_to_add)
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
        answers: Sequence[str | None],
        scores: Sequence[float | None],
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
        answers: Sequence[str | None],
        scores: Sequence[float | None],
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

    def objective(self, trial: optuna.Trial, eval_mode: EvalMode = EvalMode.TRAIN) -> float:
        """Objective function for Optuna to optimize.

        :param trial: The RAG config to test.
        :type trial: optuna.Trial
        :return: The mean score.
        :rtype: float
        """
        self.prepare_evaluation_prompt()
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
            if eval_mode == EvalMode.TRAIN:
                trial.report(score_eval, n)
            score = self.evaluator.update_avg_score(score, score_eval, n)
            if eval_mode == EvalMode.TRAIN and trial.should_prune():
                self.logger.debug("[PROCESS] Pruning... return mean score: %s", score)
                self.check_update_eval_sample_reference(answer_list, score_list, score, trial)
                gc.collect()
                return score
            self.logger.debug("[PROCESS] Mean score: %s", score)
        if eval_mode == EvalMode.TRAIN:
            self.check_update_eval_sample_reference(answer_list, score_list, score, trial)
        gc.collect()
        return score

    def prepare_evaluation_prompt(self, eval_mode: EvalMode = EvalMode.TRAIN) -> None:
        """Set the the eval prompt corresponding to the eval_mode.

        :param eval_mode: Mode of the evaluation, defaults to EvalMode.TRAIN
        :type eval_mode: EvalMode, optional
        """
        if eval_mode is EvalMode.TRAIN:
            self.evaluator.eval_prompt = self.train_prompt
        else:
            self.evaluator.eval_prompt = self.test_prompt
