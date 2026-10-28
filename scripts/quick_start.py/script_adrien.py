#!/usr/bin/env python3
"""Script to evaluate a RAG pipeline using LLM-as-a-Judge on the CRAG dataset.

This script:
1. Loads the CRAG dataset and splits it into train/test with configurable sample sizes.
2. Configures a ThinkingAwareLLMEvaluator (strips <think> tags) with qwen3:32b as the judge.
3. Uses the SimplePairWiseOptunaManager to run pairwise evaluation.
4. Logs generated responses alongside scores.

Usage examples:
    # Load existing optimization results and evaluate on test set (default: use all CRAG samples)
    python -m scripts.optim_eval_llm_judge --experiment-name llm_judge_crag_v1

    # Run new optimization then evaluate (uses all train/test samples by default)
    python -m scripts.optim_eval_llm_judge \
        --experiment-name llm_judge_crag_v1 \
        --run-optimization

    # Limit to specific sample sizes for faster testing
    python -m scripts.optim_eval_llm_judge \
        --experiment-name llm_judge_crag_small \
        --train-samples 10 --test-samples 20 \
        --run-optimization

    # Custom judge model and experiment name
    python -m scripts.optim_eval_llm_judge \
        --experiment-name llm_judge_crag_v2 \
        --judge-model mistral-small3.2:24b

    # Subset of reader models only
    python -m scripts.optim_eval_llm_judge \
        --experiment-name llm_judge_2readers \
        --reader-models qwen3:8b gemma3:12b \
        --run-optimization

    # Test with only one evaluator on the test set
    python -m scripts.optim_eval_llm_judge \
        --experiment-name llm_judge_crag_v1 \
        --evaluators similarity
"""

import argparse
import copy
import logging
import re
from typing import cast

import numpy as np
from optuna.samplers import TPESampler

from rago.data_objects import EvalSample, Metric, RAGOutput
from rago.data_objects.prompt_template import PromptTemplate
from rago.dataset import QADatasetLoader, RAGDataset
from rago.eval import BaseEvaluator, BertScore, EvalPrompts, SimilarityScore, SimpleLLMEvaluator
from rago.model.configs.llm_config.langchain import LangchainOllamaConfig
from rago.model.wrapper.llm_agent.langchain import LangchainLLMAgent
from rago.optimization.manager.base import OptimParams, RAGCandidateEval
from rago.optimization.manager.simple_pairwise import SimplePairWiseOptunaManager
from rago.optimization.search_space.llm_config_space import OllamaLLMConfigSpace
from rago.optimization.search_space.param_space import CategoricalParamSpace
from rago.optimization.search_space.rag_config_space import RAGConfigSpace
from rago.optimization.search_space.reader_config_space import LangchainReaderConfigSpace
from rago.optimization.search_space.retriever_config_space import RetrieverConfigSpace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom gamma for TPE sampler — favours exploration
# ---------------------------------------------------------------------------

def exploration_gamma(n_trials: int) -> int:
    """Custom gamma function for more exploration.

    :param n_trials: Number of trials completed so far.
    :type n_trials: int
    :return: Number of trials for l(x).
    :rtype: int
    """
    return max(int(np.ceil(0.10 * np.sqrt(n_trials))), 5)


# ---------------------------------------------------------------------------
# Thinking-tag aware evaluator
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class ThinkingAwareLLMEvaluator(SimpleLLMEvaluator):
    """SimpleLLMEvaluator that strips <think> blocks and supports reference-based evaluation.

    During optimization (eval_trial), when the eval_sample contains a reference_answer
    and reference_score from the current best RAG, this evaluator automatically switches
    to a reference prompt that asks the judge to compare the candidate answer against
    the reference answer.
    """

    def __init__(self, *args, reference_eval_prompt: str | None = None, verbose_prompts: bool = False, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize with an optional reference evaluation prompt.

        :param reference_eval_prompt: Prompt template with {reference_answer} and {reference_score}
            placeholders, used when eval_sample has a reference. If None, always uses direct eval.
        :type reference_eval_prompt: str | None
        :param verbose_prompts: If True, log the full filled prompt (useful for debugging).
        :type verbose_prompts: bool
        """
        super().__init__(*args, **kwargs)
        self._ref_eval_prompt: PromptTemplate | None = None
        self._verbose_prompts = verbose_prompts
        if reference_eval_prompt is not None:
            ref_template = PromptTemplate(reference_eval_prompt)
            self._ref_eval_prompt = ref_template.get_partially_formatted_prompt_template(
                min_score=str(self.min_score),
                max_score=str(self.max_score),
            )

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks and leading/trailing whitespace."""
        return _THINK_RE.sub("", text).strip()

    def parse_evaluation(self, evaluation: str) -> dict[str, Metric]:
        return super().parse_evaluation(self._strip_thinking(evaluation))

    def parse_n_wise_evaluation(self, evaluation: str, expected_number_of_score: int) -> list[dict[str, Metric]]:
        return super().parse_n_wise_evaluation(self._strip_thinking(evaluation), expected_number_of_score)

    def get_filled_eval_prompt(self, candidate_output: RAGOutput, eval_sample: EvalSample) -> str:
        """Build the judge prompt, switching to reference mode when a reference is available.

        When eval_sample.reference_answer and eval_sample.reference_score are set
        (i.e. during optimization after the initial pairwise round), the prompt includes
        the best RAG's answer and score so the judge can compare against them.
        Otherwise, falls back to the standard direct evaluation prompt.

        :param candidate_output: The RAG output to evaluate.
        :type candidate_output: RAGOutput
        :param eval_sample: The evaluation sample (may contain reference_answer/reference_score).
        :type eval_sample: EvalSample
        :return: The filled prompt string.
        :rtype: str
        """
        source_context_content = (
            [context.text for context in eval_sample.context] if eval_sample.context is not None else None
        )
        source_context_prompt = self.get_filled_source_context_prompt(source_context_content)
        if candidate_output.answer is None:
            raise ValueError(candidate_output.answer)

        # Use reference prompt if we have a reference answer + score and a ref prompt template
        if (
            self._ref_eval_prompt is not None
            and eval_sample.reference_answer is not None
            and eval_sample.reference_score is not None
        ):
            logger.debug(
                "> Using REFERENCE prompt | ref_answer=%s | ref_score=%.2f",
                eval_sample.reference_answer[:50] + "..." if len(eval_sample.reference_answer) > 50 else eval_sample.reference_answer,
                eval_sample.reference_score,
            )
            filled_prompt = self._ref_eval_prompt.get_filled_prompt(
                candidate_answer=candidate_output.answer,
                query=eval_sample.query,
                source_context_prompt=source_context_prompt,
                reference_answer=eval_sample.reference_answer,
                reference_score=str(eval_sample.reference_score),
            )
            if self._verbose_prompts:
                logger.info("📋 REFERENCE PROMPT:\n%s", filled_prompt)
            return filled_prompt

        # Fallback: standard direct evaluation (no reference available)
        logger.debug("> Using DIRECT prompt (no reference available)")
        filled_prompt = self.eval_prompt.get_filled_prompt(
            candidate_answer=candidate_output.answer,
            query=eval_sample.query,
            source_context_prompt=source_context_prompt,
        )
        if self._verbose_prompts:
            logger.info("> DIRECT PROMPT:\n%s", filled_prompt)
        return filled_prompt


# ---------------------------------------------------------------------------
# Custom LLM-as-a-Judge prompts (better than defaults for scoring quality)
# ---------------------------------------------------------------------------
# Placeholders required by the repo's PromptTemplate / BaseLLMEvaluator:
#   Direct eval  : {query}, {source_context_prompt}, {candidate_answer}, {min_score}, {max_score}
#   Pairwise eval: {query}, {source_context_prompt}, {candidate_answer_1}, {candidate_answer_2},
#                  {min_score}, {max_score}
#   Source context: {source_context}

JUDGE_EVAL_PROMPT = """You are an impartial expert judge. Your sole task is to assess the **correctness** of an answer to a question, using the provided source documents as ground truth.

## Question
{query}

{source_context_prompt}

## Candidate Answer
{candidate_answer}

## Scoring Instructions
Rate the correctness of the candidate answer on a scale from {min_score} to {max_score}:
  {min_score} = Completely incorrect or no answer
  1 = Mostly incorrect, only trivially related to the question
  2 = Partially correct but contains significant factual errors or omissions
  3 = Correct on the main point but with minor errors or missing details
  4 = Correct and mostly complete, only negligible imprecisions
  {max_score} = Fully correct and complete according to the source documents

## Output
Output ONLY a single integer score between {min_score} and {max_score}. No explanation, no text, just the number.
/no_think
"""

JUDGE_PAIRWISE_EVAL_PROMPT = """You are an impartial expert judge. Your sole task is to assess the **correctness** of two candidate answers to the same question, using the provided source documents as ground truth.

## Question
{query}

{source_context_prompt}

## Candidate Answer 1
{candidate_answer_1}

## Candidate Answer 2
{candidate_answer_2}

## Scoring Instructions
First, evaluate Candidate Answer 1 on correctness using this scale:
  {min_score} = Completely incorrect or no answer
  1 = Mostly incorrect, only trivially related to the question
  2 = Partially correct but contains significant factual errors or omissions
  3 = Correct on the main point but with minor errors or missing details
  4 = Correct and mostly complete, only negligible imprecisions
  {max_score} = Fully correct and complete according to the source documents

Then, evaluate Candidate Answer 2 using the same scale. Compare it to Answer 1 to ensure consistency: if Answer 2 is more correct than Answer 1 it must receive a higher or equal score, and if it is less correct it must receive a lower or equal score.

## Output
Output ONLY two integer scores, one per line, in the same order as the answers above.
Line 1 = score for Answer 1
Line 2 = score for Answer 2
No explanation, no text, just the two numbers on separate lines.
/no_think
"""

JUDGE_SOURCE_CONTEXT_PROMPT = """## Source Documents (ground truth)
Use these documents as the sole reference to judge correctness:
{source_context}
"""

JUDGE_REFERENCE_EVAL_PROMPT = """You are an impartial expert judge. Your sole task is to assess the **correctness** of a candidate answer to a question, using the provided source documents as ground truth. A reference answer with its score is provided as an anchor for calibration.

## Question
{query}

{source_context_prompt}

## Reference Answer (anchor)
{reference_answer}
**Reference score: {reference_score}**

## Candidate Answer (to evaluate)
{candidate_answer}

## Scoring Instructions
Using the reference answer and its score as a calibration anchor, rate the correctness of the **candidate answer** on a scale from {min_score} to {max_score}:
  {min_score} = Completely incorrect or no answer
  1 = Mostly incorrect, only trivially related to the question
  2 = Partially correct but contains significant factual errors or omissions
  3 = Correct on the main point but with minor errors or missing details
  4 = Correct and mostly complete, only negligible imprecisions
  {max_score} = Fully correct and complete according to the source documents

If the candidate answer is more correct than the reference, give it a higher score.
If it is less correct, give it a lower score.
If they are equally correct, give the same score.

## Output
Output ONLY a single integer score between {min_score} and {max_score}. No explanation, no text, just the number.
/no_think
"""


# ---------------------------------------------------------------------------
# CRAG Dataset Loading
# The CRAG dataset provides reference_answer for each sample.
# After pairwise initialization, reference-based evaluation uses the gold
# reference_answer (with reference_score=5.0) on the test set.
# ---------------------------------------------------------------------------


def build_crag_dataset(
    train_samples: int | None = None,
    test_samples: int | None = None,
) -> dict[str, RAGDataset]:
    """Load the CRAG dataset and split into train/test with optional sampling.

    The CRAG dataset already provides ``reference_answer`` for each sample (the gold
    answer string).  Train and test splits are deep-copied so that
    ``update_eval_sample_reference`` during pairwise optimization does not overwrite
    the gold reference answers on the test samples.

    :param train_samples: Number of training samples to keep (None = use all).
    :type train_samples: int | None
    :param test_samples: Number of test samples to keep (None = use all).
    :type test_samples: int | None
    :return: Dictionary with "train" and "test" :class:`RAGDataset` splits.
    :rtype: dict[str, RAGDataset]
    """
    logger.info("Loading CRAG dataset...")
    datasets_dict = cast(
        "RAGDataset",
        QADatasetLoader.load_dataset(RAGDataset, "crag"),
    ).split_dataset([0.05, 0.95], split_names=["train", "test"], seed=0)

    logger.info(
        "Original split sizes — train: %d, test: %d",
        len(datasets_dict["train"].samples),
        len(datasets_dict["test"].samples),
    )

    # Sub-sample to requested sizes if specified
    if train_samples is not None and train_samples > 0:
        datasets_dict["train"] = datasets_dict["train"].sample(train_samples, seed=0)
        logger.info("Sampled train to %d samples", train_samples)

    if test_samples is not None and test_samples > 0:
        datasets_dict["test"] = datasets_dict["test"].sample(test_samples, seed=0)
        logger.info("Sampled test to %d samples", test_samples)

    logger.info(
        "Final split sizes — train: %d, test: %d",
        len(datasets_dict["train"].samples),
        len(datasets_dict["test"].samples),
    )

    # Deep copy so optimization doesn't overwrite gold reference_answer on test
    datasets_dict["train"] = copy.deepcopy(datasets_dict["train"])
    datasets_dict["test"] = copy.deepcopy(datasets_dict["test"])

    # For test samples, set reference_score to max (5) for all gold reference answers.
    # This allows the LLM judge to use the reference prompt on the test set too,
    # comparing the generated answer against the gold answer anchored at 5/5.
    for sample in datasets_dict["test"].samples:
        if sample.reference_answer is not None and sample.reference_score is None:
            sample.reference_score = 5.0

    return datasets_dict


def build_llm_judge(
    model_name: str = "qwen3:32b",
    verbose_prompts: bool = False,
) -> ThinkingAwareLLMEvaluator:
    """Build a ThinkingAwareLLMEvaluator configured as an LLM judge with custom prompts.

    We use qwen3:32b for its strong instruction-following and evaluation capabilities.
    Temperature is set to 0.0 for deterministic, reproducible evaluations.
    The ThinkingAwareLLMEvaluator strips <think>...</think> blocks from judge output
    before parsing, so models like qwen3 that reason internally won't break parsing.

    :param model_name: Ollama model name for the judge LLM, defaults to "qwen3:32b".
    :type model_name: str
    :return: A configured ThinkingAwareLLMEvaluator.
    :rtype: ThinkingAwareLLMEvaluator
    """
    judge_config = LangchainOllamaConfig(
        model_name=model_name,
        temperature=0.0,       # Deterministic — critical for evaluation consistency
        top_k=1,               # Greedy decoding — only the most probable token
        top_p=1.0,             # Disabled when top_k=1 (no nucleus sampling needed)
        max_new_tokens=2048,   # Must be large enough for models that use <think> reasoning
        num_ctx=4096,          # Enough context for question + answer + source docs
        repeat_last_n=0,       # No repetition penalty (output is very short)
    )

    judge_agent = LangchainLLMAgent.make(config=judge_config)

    # Custom prompts with scoring rubric for better judge quality
    eval_prompts = EvalPrompts(
        evaluation_prompt=JUDGE_EVAL_PROMPT,
        pairwise_evaluation_prompt=JUDGE_PAIRWISE_EVAL_PROMPT,
        source_context_prompt=JUDGE_SOURCE_CONTEXT_PROMPT,
    )

    evaluator = ThinkingAwareLLMEvaluator(
        judge=judge_agent,
        min_score=0,
        max_score=5,
        eval_prompts=eval_prompts,
        reference_eval_prompt=JUDGE_REFERENCE_EVAL_PROMPT,
        verbose_prompts=verbose_prompts,
    )

    logger.info(
        "LLM Judge configured: model=%s, temperature=%.1f, top_k=%d, max_tokens=%d",
        model_name,
        judge_config.temperature,
        judge_config.top_k,
        judge_config.max_new_tokens,
    )

    return evaluator


def evaluate_with_llm_judge(
    experiment_name: str = "llm_judge_crag",
    judge_model: str = "qwen3:32b",
    reader_models: list[str] | None = None,
    evaluator_names: list[str] | None = None,
    run_optimization: bool = False,
    verbose_prompts: bool = False,
    train_samples: int | None = None,
    test_samples: int | None = None,
) -> RAGCandidateEval:
    """Run pairwise optimization + test evaluation using LLM-as-a-Judge on the CRAG dataset.

    :param experiment_name: Name for the experiment folder.
    :type experiment_name: str
    :param judge_model: Model name for the LLM judge.
    :type judge_model: str
    :param reader_models: List of reader model names to explore, defaults to standard set.
    :type reader_models: list[str] | None
    :param evaluator_names: Names of evaluators to use on test set, defaults to bertscore + similarity.
    :type evaluator_names: list[str] | None
    :param run_optimization: If True, run optimization; if False, load existing study.
    :type run_optimization: bool
    :param verbose_prompts: If True, log full prompts sent to the LLM judge.
    :type verbose_prompts: bool
    :param train_samples: Number of CRAG training samples to use (None = use all).
    :type train_samples: int | None
    :param test_samples: Number of CRAG test samples to use (None = use all).
    :type test_samples: int | None
    :return: Evaluation results.
    :rtype: RAGCandidateEval
    """
    if reader_models is None:
        reader_models = ["qwen3:8b", "gemma3:12b", "gpt-oss:20b"]
    if evaluator_names is None:
        evaluator_names = ["bertscore", "similarity"]

    # Check if experiment exists (only if not running optimization)
    from pathlib import Path
    experiment_path = Path(f"experiments/{experiment_name}")
    study_db = experiment_path / "study.db"

    if run_optimization:
        logger.info(f"Running optimization for experiment: {experiment_name}")
    else:
        # Only check for existing study if not running optimization
        if not experiment_path.exists():
            raise ValueError(
                f"Experiment '{experiment_name}' not found in experiments/. "
                "Use --run-optimization to create it.",
            )
        if not study_db.exists():
            raise ValueError(
                f"Study database not found at {study_db}. "
                "Use --run-optimization to create it.",
            )
        logger.info(f"Loading existing experiment: {experiment_name}")

    logger.info("=" * 80)
    logger.info("LLM-AS-A-JUDGE EVALUATION")
    logger.info("=" * 80)
    logger.info("Judge model: %s", judge_model)
    logger.info("Reader models: %s", reader_models)

    # Build dataset from CRAG
    datasets = build_crag_dataset(
        train_samples=train_samples,
        test_samples=test_samples,
    )
    logger.info("Dataset: %d train samples, %d test samples, %d corpus documents",
                len(datasets["train"].samples), len(datasets["test"].samples),
                len(datasets["train"].corpus))

    # Build LLM judge evaluator
    llm_judge = build_llm_judge(model_name=judge_model, verbose_prompts=verbose_prompts)

    # Build test evaluators (BertScore + SimilarityScore only — LLM judge is used
    # during optimization but not on the test set because it takes to much time)
    evaluator_map: dict[str, BaseEvaluator] = {
        "bertscore": BertScore(),
        "similarity": SimilarityScore(model_name="BAAI/bge-large-en-v1.5"),
    }
    test_evaluators = [evaluator_map[name] for name in evaluator_names]
    logger.info("Test evaluators: %s", evaluator_names)

    # RAG configuration search space
    config_space = RAGConfigSpace(
        retriever_space=RetrieverConfigSpace(
            retriever_type_name=CategoricalParamSpace(
                choices=["BM25Retriever", "HybridRetriever"],
            ),
        ),
        reader_space=LangchainReaderConfigSpace(
            OllamaLLMConfigSpace(
                model_name=CategoricalParamSpace(
                    choices=reader_models,
                ),
            ),
        ),
    )

    # Optimization parameters (small since this is a mini test)
    params = OptimParams(
        experiment_name=experiment_name,
        n_startup_trials=50,
        n_iter=500,
    )

    # TPE sampler with custom gamma for exploration
    sampler = TPESampler(
        n_startup_trials=params.n_startup_trials,
        gamma=exploration_gamma,
        multivariate=True,
        prior_weight=1.5,
        constant_liar=False,
    )

    # Create the pairwise optimization manager with LLM judge
    optimizer = SimplePairWiseOptunaManager(
        params=params,
        datasets=datasets,
        optim_evaluator=llm_judge,
        optim_metric_name="correctness",
        test_evaluators=test_evaluators,
        config_space=config_space,
        num_initial_rags=2,
        sampler=sampler,
    )

    # Run optimization if requested, otherwise load existing results
    if run_optimization:
        logger.info("Starting pairwise optimization with LLM judge...")
        optimizer.optimize()
        study = optimizer.manager
        logger.info("Optimization complete! %d trials run", len(study.trials))
        logger.info("Best trial: #%d with score %.4f", study.best_trial.number, study.best_trial.value)
    else:
        logger.info("Loading optimization results...")
        study = optimizer.load_results()
        logger.info("Found %d trials in study", len(study.trials))
        logger.info("Best trial: #%d with score %.4f", study.best_trial.number, study.best_trial.value)

    # Run test evaluation
    logger.info("Evaluating best configuration on test set...")
    best_result = optimizer.test()

    # Log the best configuration
    logger.info("=" * 80)
    logger.info("BEST RAG CONFIGURATION")
    logger.info("=" * 80)
    logger.info("Trial #%d with train score: %.4f", study.best_trial.number, best_result.train_score)
    logger.info("Full config: %s", best_result.config)
    logger.info("=" * 80)

    # Display results
    print("\n" + "=" * 80)
    print("LLM-AS-A-JUDGE EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nExperiment: {experiment_name}")
    print(f"Judge Model: {judge_model}")

    # Display best RAG configuration (simple repr to avoid attribute errors)
    print(f"\n{'Best RAG Configuration:':-^80}")
    print(f"  Trial Number: {study.best_trial.number}")
    print(f"  Train Score: {best_result.train_score:.4f}")
    print(f"\n  Full Config:\n{best_result.config}")
    print(f"{'':-^80}")

    test_sample_list = datasets["test"].samples
    print(f"\nTest Evaluations (n={len(test_sample_list)} samples):")
    print(f"Evaluators: {', '.join(evaluator_names)}")
    print("\nMetrics Statistics:")
    for metric_name, scores in best_result.full_evals.items():
        arr = np.array(scores)
        print(f"  {metric_name:30s}:")
        print(f"    Mean:  {arr.mean():7.4f}")
        print(f"    Std:   {arr.std():7.4f}")
        print(f"    Min:   {arr.min():7.4f}")
        print(f"    Max:   {arr.max():7.4f}")
    print("\nFull per-sample scores:")
    for i in range(len(test_sample_list)):
        query_preview = test_sample_list[i].query[:60]
        print(f"  Sample {i+1} [{query_preview}...]:")
        for metric_name, scores in best_result.full_evals.items():
            print(f"    {metric_name:30s} = {scores[i]:.4f}")
    print("\n" + "=" * 80)
    print(f"Results saved to: experiments/{experiment_name}/best_rag_results.json")
    print("=" * 80 + "\n")

    return best_result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG with LLM-as-a-Judge on the CRAG dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="llm_judge_crag",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="qwen3:32b",
        choices=["gemma3:12b", "gpt-oss:20b", "qwen3:32b", "deepseek-r1:32b", "mistral-small3.2:24b"],
        help="LLM model to use as judge (recommended: qwen3:32b for best instruction following)",
    )
    parser.add_argument(
        "--reader-models",
        type=str,
        nargs="+",
        default=["ministral-3:8b", "qwen3:8b", "gemma3:12b", "gpt-oss:20b"],
        help="Reader models to explore in the RAG pipeline",
    )
    parser.add_argument(
        "--evaluators",
        type=str,
        nargs="+",
        choices=["bertscore", "similarity"],
        default=["bertscore", "similarity"],
        help="Evaluators to use for testing (LLM judge is used during optimization only)",
    )
    parser.add_argument(
        "--run-optimization",
        action="store_true",
        help="Run optimization before evaluating (default: load existing study)",
    )
    parser.add_argument(
        "--verbose-prompts",
        action="store_true",
        help="Log the full prompts sent to the LLM judge (useful for debugging)",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Number of CRAG training samples to use (default: use all)",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Number of CRAG test samples to use (default: use all)",
    )

    args = parser.parse_args()

    evaluate_with_llm_judge(
        experiment_name=args.experiment_name,
        judge_model=args.judge_model,
        reader_models=args.reader_models,
        evaluator_names=args.evaluators,
        run_optimization=args.run_optimization,
        verbose_prompts=args.verbose_prompts,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
    )


if __name__ == "__main__":
    main()
