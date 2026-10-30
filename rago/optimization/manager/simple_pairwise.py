"""Simple class for RAG Config optimization with pair-wise evaluation method."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Optional, cast

import optuna

from rago.data_objects import Candidate, EvalSample
from rago.prompts import PromptConfig

if TYPE_CHECKING:

    import optuna

    from rago.data_objects import ModelInput, ModelOutput
    from rago.data_objects.eval_sample import EvalSample
    from rago.dataset import RAGDataset
    from rago.model.wrapper.tunable_model import TunableModel
    from rago.optimization.search_space.tunable_model_config_space import TunableModelConfigSpace


from rago.eval import (
    BaseDependentEvaluator,
    BaseEvaluator,
    DependentEvaluatorReference,
    EvalPrompts,
    EvaluatorReference,
    SimpleLLMEvaluator,
)
from rago.model.configs.tunable_model_config import TunableModelConfig
from rago.model.wrapper.llm_agent import LangchainLLMAgent
from rago.optimization.manager.base import BaseOptunaManager, OptimParams
from rago.prompts import DEFAULT_REFERENCE_EVAL_PROMPT


class SimplePairWiseOptunaManager[ModelInputT: ModelInput, ModelOutputT: ModelOutput, ConfigT: TunableModelConfig](
    BaseOptunaManager[ModelInputT, ModelOutputT, DependentEvaluatorReference[ModelOutputT], ConfigT]):
    """A simple direct optimization manager calling Optuna to optimize Reader parameters.

    The Optimization manager is instantiated with Optuna Study.
    """

    def __init__(
        self,
        *,
        params: Optional[OptimParams] = None,
        datasets: dict[str, RAGDataset],
        optim_evaluator: BaseEvaluator[ModelOutputT, DependentEvaluatorReference[ModelOutputT]],
        test_evaluators: list[BaseEvaluator[ModelOutputT, EvaluatorReference]],
        model_cls: type[TunableModel[ModelInputT, ModelOutputT, ConfigT]],
        config_space: TunableModelConfigSpace[ConfigT],
        prompt_config: Optional[PromptConfig] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        num_initial_models: int = 2,
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
        :param model_cls: The model to optimize, built with ``model_cls.make(config, ctx)``.
        :type model_cls: type[TunableModel[ConfigT]]
        :param config_space: The space of model config to search in.
        :type config_space: TunableModelConfigSpace[ConfigT]
        :param prompt_config: Configuration of the prompt used by the reader, if any.
        :type prompt_config: Optional[PromptConfig], optional
        :param sampler: The sampler used to suggest new configurations to test, defaults to None
        :type sampler: Optional[optuna.samplers.BaseSampler], optional
        :param pruner: The pruner used to terminate early unpromising trials, defaults to None
        :type pruner: Optional[optuna.pruners.BasePruner], optional
        :param num_initial_models: Num of models to test initially (must be compatible with the evaluator's
            pairwise prompt).
        :type num_initial_models: int = 2
        """
        self.num_initial_trials = num_initial_models
        if optim_evaluator is None:
            optim_evaluator = SimpleLLMEvaluator(
                judge=LangchainLLMAgent.make_from_backend(),
                eval_prompts=EvalPrompts(DEFAULT_REFERENCE_EVAL_PROMPT),
            )
        super().__init__(
            params=params,
            datasets=datasets,
            optim_evaluator=optim_evaluator,
            test_evaluators=test_evaluators,
            model_cls=model_cls,
            config_space=config_space,
            prompt_config=prompt_config,
            sampler=sampler,
            pruner=pruner,
        )

    def optimize(self) -> None:
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

    def run_initial_trials(
        self,
    ) -> None:
        """Initialize the optimization by getting the first best model configuration."""
        self.logger.info("[INIT OPTIM] Run First Round of the Battle...")
        dataset = self.datasets["train"]
        models, trials = self.instantiate_initial_models()
        candidates: list[dict[int, Candidate[ModelOutputT]]] = [{} for _ in range(self.num_initial_trials)]
        for n, test_sample in enumerate(dataset.samples):
            self.logger.debug("[INIT OPTIM] Iteration %s", n)
            if test_sample.context is not None:
                self.logger.debug("[INIT OPTIM] Test Eval Sample: %s", test_sample)
                models_input = self.model_cls.get_input(test_sample)
                outputs = [model.get_output(models_input) for model in models]
                evaluations = self.optim_evaluator.evaluate_n_wise(
                    outputs,
                    DependentEvaluatorReference(test_sample),
                )
                for idx_trial, (o, e) in   enumerate(zip(outputs, evaluations,strict = True)):
                    candidates[idx_trial][n] = Candidate(o, e)
        mean_scores: list[float] = [
            sum([c.metric.score for c in trial_candidates.values()])/len(trial_candidates)
            for trial_candidates in candidates
            ]
        for trial, s in zip(trials, mean_scores, strict = True):
            self.manager.tell(trial, float(s))
        idx_best = max(enumerate(mean_scores), key=lambda x: x[1])[0]
        self.update_eval_sample_reference(
            candidates[idx_best],
            mean_scores[idx_best],
        )

    def instantiate_initial_models(
            self) -> tuple[list[TunableModel[ModelInputT, ModelOutputT, ConfigT]], list[optuna.Trial]]:
        """Instantiate models for the initial pairwise comparison.

        :return: The model candidates and the trials.
        :rtype: tuple[list[TunableModel[ConfigT]], list[optuna.Trial]]
        """
        trials: list[optuna.Trial] = []
        models: list[TunableModel[ModelInputT, ModelOutputT,ConfigT]] = []
        for _ in range(self.num_initial_trials):
            trials.append(self.manager.ask())
            config = self.config_space.sample(trials[-1])
            self.logger.info("[PROCESS] Current Config: %s", config)
            models.append(self.make_model(config, self.datasets["train"]))
        return (models, trials)

    def update_eval_sample_reference(
        self,
        candidates: dict[int, Candidate[ModelOutputT]],
        mean_score: Optional[float] = None,
    ) -> None:
        """Check and update the eval sample answer and scores references if current candidate is better.

        :param answers: The list of answers to populate the answer reference of the eval samples.
        :type answers: list[Optional[str]],
        :param scores: The list of scores to populate the reference score of the eval samples.
        :type scores: list[Optional[float]]
        :param score: The current score
        :type score: float
        """
        if mean_score is None or (mean_score > self.manager.best_trial.values[0]):
            self.best_candidate: list[Candidate[ModelOutputT]] = []
            for idx in range(len(candidates)):
                if idx not in candidates:
                    msg = "New best candidate is incomplete"
                    raise ValueError(msg)

                self.best_candidate.append(candidates[idx])

    def get_evaluator_reference[ReferenceT:EvaluatorReference](
            self,
            evaluator: BaseEvaluator[ModelOutputT, ReferenceT],
            eval_sample: EvalSample,
            ref_candidate: Optional[Candidate[ModelOutputT]] = None,
            ) -> ReferenceT:
        """Get the evaluator reference from eval sample and manager internal state.

        :param eval_sample: The eval_sample to use as reference.
        :type eval_sample: EvalSample.
        :return: The reference to pass to the evaluator.
        :rtype: EvaluatorReferenceT
        """
        if ref_candidate is None:
            return evaluator.get_evaluator_reference(eval_sample)
        if isinstance(evaluator, BaseDependentEvaluator):
            return cast("ReferenceT", evaluator.get_evaluator_reference(eval_sample, ref_candidate))
        return evaluator.get_evaluator_reference(eval_sample)

    def get_sample_output_and_eval(
        self,
        evaluator: BaseEvaluator[ModelOutputT, DependentEvaluatorReference[ModelOutputT]],
        eval_sample: EvalSample,
        ref_candidate: Candidate[ModelOutputT],
        model: TunableModel[ModelInputT, ModelOutputT, ConfigT],
    ) -> Candidate[ModelOutputT]:
        """Get the current metric and output for the sample.

        :param eval_sample: The dataset for the evaluation.
        :type eval_sample: EvalSample
        :param model: The model candidate.
        :type model: TunableModel[ConfigT]
        :return: The score of the current evaluation.
        :rtype: tuple[str, float]
        :raise ValueError if the given context or the score or answer is None.
        """
        self.logger.debug("[PROCESS] Eval sample: %s", eval_sample)
        model_input = model.get_input(eval_sample)
        candidate_output = model.get_output(model_input)
        eval_ref = self.get_evaluator_reference(evaluator, eval_sample, ref_candidate)
        evaluation = evaluator.evaluate(candidate_output, eval_ref)
        self.logger.debug("[PROCESS] Evaluation Results: %s", evaluation)
        return Candidate(candidate_output, evaluation)


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
        model = self.sample_model(trial, dataset)
        self.logger.info("[PROCESS] Trial %s", trial.number)
        if len(self.manager.best_trials) > 0:
            self.logger.info(
                "[PROCESS] Best is trial %s",
                self.manager.best_trial.number,
            )
            self.logger.info("[PROCESS] Best Value: %s", self.manager.best_trial.values)
        mean_score = 0.0
        candidates_by_position: dict[int, Candidate] = {}
        for step, (idx, test_sample) in enumerate(self.get_samples_and_ids(trial, dataset.samples)):
            self.logger.debug("[PROCESS] Iteration %s", step)
            sample_candidate = self.get_sample_output_and_eval(
                self.optim_evaluator, test_sample, self.best_candidate[idx], model)
            self.logger.info(
                "[PROCESS] Trial %s | Sample %s | Query: %s  | Scores: %s",
                trial.number,
                idx,
                test_sample.query,
                sample_candidate.metric.score,
            )
            candidates_by_position[idx] = (sample_candidate)
            mean_score = self.optim_evaluator.update_avg_score(mean_score, sample_candidate.metric.score, step)
            self.report(trial, sample_candidate.metric.score, idx, mean_score)
            self.logger.debug("[PROCESS] Mean score for current iteration: %s", mean_score)
            if self._should_prune(trial, mean_score):
                self.logger.debug("[PROCESS] Pruning...")
                self._save_trial_metrics(trial, [c.metric for c in candidates_by_position.values()])
                gc.collect()
                return mean_score

        self._save_trial_metrics(trial, [c.metric for c in candidates_by_position.values()])
        self.update_eval_sample_reference(candidates_by_position, mean_score)
        gc.collect()
        return mean_score
