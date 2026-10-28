"""Simple class for RAG Config optimization with pair-wise evaluation method."""

from __future__ import annotations

import gc
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import numpy as np
import optuna

from rago.data_objects import GenerationOutput, RAGOutput
from rago.eval import BaseEvaluator
from rago.model.configs.tunable_model_config import TunableModelConfig
from rago.prompts import PromptConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    import optuna

    from rago.data_objects import RAGOutput
    from rago.data_objects.eval_sample import EvalSample
    from rago.dataset import RAGDataset
    from rago.model.wrapper.tunable_model import TunableModel
    from rago.optimization.search_space.tunable_model_config_space import TunableModelConfigSpace
from rago.data_objects import Metric
from rago.eval import BaseLLMEvaluator, EvalPrompts, SimpleLLMEvaluator
from rago.model.wrapper.llm_agent import LangchainLLMAgent
from rago.optimization.manager.base import BaseOptunaManager, OptimParams
from rago.prompts import DEFAULT_REFERENCE_EVAL_PROMPT


class SimplePairWiseOptunaManager[OutputType: GenerationOutput, ConfigType: TunableModelConfig](
    BaseOptunaManager[OutputType, ConfigType]):
    """A simple direct optimization manager calling Optuna to optimize Reader parameters.

    The Optimization manager is instantiated with Optuna Study.
    """

    def __init__(
        self,
        *,
        params: Optional[OptimParams] = None,
        datasets: dict[str, RAGDataset],
        optim_evaluator: BaseLLMEvaluator,
        optim_metric_name: str,
        test_evaluators: list[BaseEvaluator],
        model_class: TunableModel[OutputType, ConfigType],
        config_space: Optional[TunableModelConfigSpace] = None,
        prompt_config: Optional[PromptConfig] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        num_initial_rags: int = 2,
    ) -> None:
        """Initialize the Simple Direct Optimization Manager from seed data to generate dataset.

        :param params: Parameters of the optimization, defaults to None.
        :type params: Optional[OptimParams], optional
        :param dataset: Dataset on which rag configs will be evaluated.
        :type dataset: RAGDataset
        :param optim_evaluator: Evaluator used to evaluate RAG outputs.
        :type optim_evaluator: BaseLLMEvaluator
        :param optim_metric_name: Name of the metric to optimize if the evaluator returns a dict.
        :type optim_metric_name: str
        :param test_evaluators: Evaluators used in test, if None evaluator is used for tests, defaults to None.
        :type test_evaluators: Optional[list[EvaluatorType]] = None
        :param config_space: The space of model config to search in, defaults to None
        :type config_space: Optional[TunableModelConfigSpace], optional
        :param prompt_config: Configuration of the prompt used by the reader of each RAG.
        :type prompt_config: Optional[PromptConfig], optional
        :param sampler: The sampler used to suggest new rag configuration to tests, defaults to None
        :type sampler: Optional[optuna.samplers.BaseSampler], optional
        :param pruner: The pruner used to terminate early unpromising trials, defaults to None
        :type pruner: Optional[optuna.pruners.BasePruner], optional
        :param num_initial_rags: Num of RAG to test initially (must be compatible with the evaluator's pairwise prompt).
        :type num_initial_rags: int = 2
        """
        self.num_initial_trials = num_initial_rags
        if optim_evaluator is None:
            optim_evaluator = SimpleLLMEvaluator(
                judge=LangchainLLMAgent.make(),
                eval_prompts=EvalPrompts(DEFAULT_REFERENCE_EVAL_PROMPT),
            )
        super().__init__(
            params=params,
            datasets=datasets,
            optim_evaluator=optim_evaluator,
            optim_metric_name=optim_metric_name,
            test_evaluators=test_evaluators,
            model_class= model_class,
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
        self.manager.optimize(
            lambda trial: self.eval_trial(trial, self.datasets["train"]),
            self.params.n_iter,
            catch=(Exception,),
        )
        self.logger.info("[RESULT] Best trial %s", self.manager.best_trial)
        return self.manager

    def run_initial_trials(
        self,
    ) -> None:
        """Initialize the optimization by getting the first best RAG configuration."""
        self.logger.info("[INIT OPTIM] Run First Round of the Battle RAG...")
        dataset = self.datasets["train"]
        models, trials = self.instantiate_models()
        answers_candidates: list[list[str]] = [[] for _ in range(self.num_initial_trials)]
        scores: list[list[float]] = [[] for _ in range(self.num_initial_trials)]
        for n, test_sample in enumerate(dataset.samples):
            self.logger.debug("[INIT OPTIM] Iteration %s", n)
            if test_sample.context is not None:
                start = time.time()
                model_outputs = [model_cand.get_output(test_sample.query) for model_cand in models]
                end = time.time()
                self.logger.debug("[INIT OPTIM] RAGs generation time (s): %s", end- start)
                start = time.time()
                evaluations = self.optim_evaluator.evaluate_n_wise(
                    model_outputs,
                    test_sample,
                )
                end = time.time()
                self.logger.debug("[INIT OPTIM] Evaluation time (s): %s", end- start)
                self.logger.debug(
                    "[INIT OPTIM] Evaluation scores: [%s, %s]",
                    evaluations[0]["correctness"].score, evaluations[1]["correctness"].score)
                answers_candidates, scores = self.update_answers_scores(
                    model_outputs,
                    evaluations,
                    answers_candidates,
                    scores,
                )

        best_trial, best_answer, best_scores = self.get_best_candidate(
            trials,
            answers_candidates,
            scores,
        )
        self.update_eval_sample_reference(
            best_answer,
            best_scores,
        )

        for trial, score_list in zip(trials, scores, strict=True):
            mean_score = np.nanmean(score_list)
            if np.isnan(mean_score):
                mean_score = 0.0
            self.manager.tell(trial, float(mean_score))

    def instantiate_models(self) -> tuple[list[TunableModel[OutputType, ConfigType]], list[optuna.Trial]]:
        """Instantiate models for the initial pairwise comparison.

        :return: The model candidates and the trials.
        :rtype: tuple[list[TunableModel], list[optuna.Trial]]
        """
        trials: list[optuna.Trial] = []
        models: list[TunableModel[OutputType, ConfigType]] = []
        inputs_chunks = [doc.text for doc in self.datasets["train"].corpus.values()]
        for _ in range(self.num_initial_trials):
            trials.append(self.manager.ask())
            config = self.config_space.sample(trials[-1])
            self.logger.info("[PROCESS] Current Config: %s", config)
            models.append(self.model_class.make(
                config,
                prompt_config=self.prompt_config,
                inputs_chunks=inputs_chunks))
        return (models, trials)

    def update_answers_scores(
        self,
        model_output: list[RAGOutput],
        evaluation: list[dict[str, Metric]],
        answers: list[list[str]],
        scores: list[list[float]],
    ) -> tuple[list[list[str]], list[list[float]]]:
        """Update the list of the candidate answers and scores and log it.

        :param model_output: The candidate rag output
        :type model_output: RagOutput
        :param evaluation: The list of the candidate evaluation
        :type evaluation: dict[str, Metric]
        :param answers: The list of the candidate answers
        :type answers: list[str]
        :param scores: The list of the candidate scores
        :type scores: list[float]
        """
        for idx, (out, e) in enumerate(zip(model_output, evaluation, strict=False)):
            if out.answer is not None:
                answers[idx].append(out.answer)
            scores[idx].append(e[self.optim_metric_name].score)
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
        avg_scores = [np.nanmean(score_list) for score_list in scores]
        best_index = np.argmax(avg_scores)
        self.logger.debug("[INIT OPTIM] Best index: %s", best_index)
        return trial[best_index], answers[best_index], scores[best_index]

    def update_eval_sample_reference(
        self,
        answers: Sequence[str | None],
        scores: Sequence[float | None],
        score: Optional[float] = None,
    ) -> None:
        """Check and update the eval sample answer and scores references if current candidate is better.

        :param answers: The list of answers to populate the answer reference of the eval samples.
        :type answers: list[Optional[str]],
        :param scores: The list of scores to populate the reference score of the eval samples.
        :type scores: list[Optional[float]]
        :param score: The current score
        :type score: float
        """
        if score is None or (score > self.manager.best_trial.values[0]):
            for idx, (answer, s) in enumerate(zip(answers, scores, strict=False)):
                self.datasets["train"].samples[idx].reference_answer = answer
                self.datasets["train"].samples[idx].reference_score = s

    def get_current_score_answer(
        self,
        evaluator: BaseEvaluator,
        eval_sample: EvalSample,
        model_candidate: TunableModel[OutputType, ConfigType],
    ) -> tuple[str, dict[str, Metric]]:
        """Get the current score and answer.

        :param eval_sample: The dataset for the evaluation.
        :type eval_sample: EvalSample
        :param rag_candidate: The model candidate.
        :type rag_candidate: TunableModel
        :return: The score of the current evaluation.
        :rtype: tuple[str, float]
        :raise ValueError if the given context or the score or answer is None.
        """
        if eval_sample.context is not None:
            candidate = model_candidate.get_output(eval_sample.query)
            evaluation = evaluator.evaluate(candidate, eval_sample)
            if candidate.answer is not None:
                return candidate.answer, evaluation
        msg_error = "None values found."
        raise ValueError(msg_error)

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
        rag_candidate = self.sample_model(trial, dataset)
        if len(self.manager.best_trials) > 0:
            self.logger.info(
                "[Trial %s] Best is trial %s",
                trial.number,
                self.manager.best_trial.number,
            )
            self.logger.info("[Trial %s] Best Value: %s", trial.number, self.manager.best_trial.values)
        trial_eval: dict[str, Metric] = defaultdict(Metric)
        score_list: list[float] = []
        answer_list = []
        for n, test_sample in enumerate(dataset.samples):
            self.logger.debug("[Trial %s] Iteration %s",trial.number, n)
            start = time.time()
            answer_eval, single_eval = self.get_current_score_answer(self.optim_evaluator, test_sample, rag_candidate)
            end = time.time()
            self.logger.info("[Trial %s] Generation & evaluation time: %s", trial.number, end -start)
            self.logger.info("[Trial %s] Score: %s", trial.number, single_eval["correctness"].score)
            trial_eval = {
                name: Metric(self.optim_evaluator.update_avg_score(trial_eval[name].score, metric.score, n))
                for name, metric in single_eval.items()
            }
            self.logger.info("[Trial %s] Avg Score: %s", trial.number, trial_eval["correctness"].score)
            single_score = single_eval[self.optim_metric_name].score
            answer_list.append(answer_eval)
            score_list.append(single_score)
            trial.report(single_score, n)
            mean_score = trial_eval[self.optim_metric_name].score

            if self._should_prune(trial, mean_score):
                self.logger.debug("[Trial %s] Pruning...")
                self._save_trial_metrics(trial, trial_eval)
                gc.collect()
                return mean_score

        self._save_trial_metrics(trial, trial_eval)
        self.update_eval_sample_reference(answer_list, score_list, mean_score)
        gc.collect()
        return mean_score
