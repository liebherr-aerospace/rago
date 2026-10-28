
from pathlib import Path
from typing import cast

import optuna
import requests
from transformers import set_seed
from urllib3.exceptions import InsecureRequestWarning

from rago.dataset import RAGDataset
from rago.eval import CoTLLMEvaluator
from rago.eval.llm_evaluator import EvalPrompts, PolicyOnError
from rago.eval.llm_evaluator.cot import EvalOutputTags
from rago.model.configs.llm_config.langchain import LangchainOllamaConfig
from rago.model.wrapper.llm_agent.llm_agent_factory import LLMAgentFactory
from rago.model.wrapper.rag.base import RAG, RAGConfig
from rago.optimization.manager import OptimParams, SimplePairWiseOptunaManager
from rago.optimization.search_space.llm_config_space import OllamaLLMConfigSpace
from rago.optimization.search_space.param_space import CategoricalParamSpace, IntParamSpace
from rago.optimization.search_space.rag_config_space import RAGConfigSpace
from rago.optimization.search_space.reader_config_space import LangchainReaderConfigSpace
from rago.optimization.search_space.retriever_config_space import RetrieverConfigSpace
from rago.prompts import PromptConfig
from rago.utils import PATH_DEFAULT_DATA_DIR, PATH_PROJECT
from scripts.baselines.utils import get_judge
from scripts.eval import eval_model
from scripts.prompts import (
    DEFAULT_COT_SCORE_TAG,
    DEFAULT_EXPLANATION_TAG,
    DEFAULT_SCORE_I_TAG_TEMPLATE,
    PROMPT_CONTEXT_EVAL,
    PROMPT_CONTEXTE,
    PROMPT_DUAL_EVAL,
    PROMPT_EVAL_PARTIEL,
    PROMPT_SYSTEME,
)
from scripts.seeds import load_seeds

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


def get_evaluator(
        seed: int,
        model_name: str,
) -> CoTLLMEvaluator:
    llm_config = LangchainOllamaConfig(
        model_name=model_name,
        temperature=1.0,
        top_p=1.0,
        top_k=20,
        num_ctx = 10000,
        repeat_penalty = 1.0,
        max_new_tokens = 256,
        seed = seed,
        )
    llm_agent = LLMAgentFactory.make(llm_config)
    eval_prompts = EvalPrompts(
        evaluation_prompt=PROMPT_EVAL_PARTIEL,
        pairwise_evaluation_prompt=PROMPT_DUAL_EVAL,
        source_context_prompt=PROMPT_CONTEXT_EVAL,
    )
    tags = EvalOutputTags(
        score_tag=DEFAULT_COT_SCORE_TAG,
        explanation_tag=DEFAULT_EXPLANATION_TAG,
        score_i_tag_template=DEFAULT_SCORE_I_TAG_TEMPLATE,
    )
    return CoTLLMEvaluator(
        llm_agent,
        min_score=0,
        max_score = 1,
        policy_on_errors = PolicyOnError.MIN_SCORE,
        eval_prompts=eval_prompts,
        eval_output_tags=tags,
    )

def get_optimizer(
        seed: int,
        datasets: dict[str, RAGDataset],
        model_name: str,
        ) -> SimplePairWiseOptunaManager:
    set_seed(seed)
    sampler = optuna.samplers.TPESampler(n_startup_trials=10, seed=seed)
    params = OptimParams(
        experiment_name=str(seed),
        path_experiments="experiments/baselines/pairwise_optim",
        n_startup_trials=10,
        n_iter=1,
    )
    evaluator = get_evaluator(seed, model_name)

    config_space = RAGConfigSpace(
        RetrieverConfigSpace(
            retriever_type_name=CategoricalParamSpace(
                    choices=["BM25Retriever", "HybridRetriever", "VectorIndexRetriever"],
                )),
        reader_space=LangchainReaderConfigSpace(
            OllamaLLMConfigSpace(
                model_name=CategoricalParamSpace(choices=[model_name]),
                num_ctx = IntParamSpace(low = 10000, high = 10000),
                seed = CategoricalParamSpace(choices = [seed]),
            ),
        ),
    )
    rag_prompt = PromptConfig(
        system_message=PROMPT_SYSTEME,
        context_template=PROMPT_CONTEXTE,

    )
    return  SimplePairWiseOptunaManager(
            params=params,
            datasets=datasets,
            optim_evaluator=evaluator,
            optim_metric_name="correctness",
            test_evaluators=[evaluator],
            prompt_config=rag_prompt,
            sampler = sampler,
            model_class = RAG,
            config_space=config_space,
            num_initial_rags=2,
        )

def evaluate_direct_optim_rag(seed: int, datasets: dict[str, RAGDataset], save_path: Path, model_name: str = "qwen3.5:2b") -> RAG:
    optimizer = get_optimizer(seed, datasets, model_name)
    optimizer.optimize()
    best_trials = optimizer.get_n_best_trials(number=5)
    inputs_chunks = [doc.text for doc in datasets["test"].corpus.values()]
    for idx, trial in enumerate(best_trials):
        config: RAGConfig = optimizer.config_space.sample(trial)
        rag = RAG.make(
                rag_config=config,
                prompt_config=optimizer.prompt_config,
                inputs_chunks=inputs_chunks,
            )
        eval_file_name = f"test_trial_{idx}.json"
        eval_model(rag, get_judge(seed), datasets["test"].sample(600, seed = seed), save_path/str(seed)/eval_file_name)

def run_pairwise_optim(num_seeds: int = 5) -> None:
    seeds = load_seeds(num_seeds= num_seeds)
    path_data = Path(PATH_DEFAULT_DATA_DIR)
    path_augmented = path_data/"eval_llm_2026_with_pages/augmented"
    path_direct_optim = Path(PATH_PROJECT, "experiments", "baselines", "pairwise_optim")
    datasets = cast("dict[str, RAGDataset]", RAGDataset.load_dataset(path_augmented))
    for s in seeds:
        evaluate_direct_optim_rag(s, datasets, path_direct_optim)

if __name__ == "__main__":
    run_pairwise_optim()
