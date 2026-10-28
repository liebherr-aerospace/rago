import random
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
from rago.optimization.manager import OptimParams, SimplePairWiseOptunaManager
from rago.optimization.search_space.llm_config_space import OllamaLLMConfigSpace
from rago.optimization.search_space.param_space import CategoricalParamSpace, IntParamSpace
from rago.optimization.search_space.rag_config_space import RAGConfigSpace
from rago.optimization.search_space.reader_config_space import LangchainReaderConfigSpace
from rago.optimization.search_space.retriever_config_space import RetrieverConfigSpace
from rago.prompts import PromptConfig
from rago.utils import PATH_DEFAULT_DATA_DIR
from scripts.dataset_generation_script import augment
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

AUGMENT = False

def get_optim_dataset(
        path_augmented: Path,
) -> dict[str, RAGDataset]:
    # Load dataset and evaluator
    datasets = cast("dict[str, RAGDataset]", RAGDataset.load_dataset(path_augmented))
    datasets["train"].samples = random.sample(datasets["train"].samples, min(len(datasets["train"].samples),200) )
    return datasets

def get_evaluator(
        model_name: str,
        seed: int,

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
        path_augmented: Path,
        seed: int,
        model_name: str = "qwen3.5:2b",
        experiment_name: str = "optim_3",
) -> SimplePairWiseOptunaManager:

    set_seed(seed)
    datasets = get_optim_dataset(path_augmented)
    sampler = optuna.samplers.TPESampler(n_startup_trials=10, seed=seed)
    params = OptimParams(
        experiment_name=experiment_name,
        n_startup_trials=10,
        n_iter=100,
    )
    evaluator = get_evaluator(model_name, seed)

    config_space = RAGConfigSpace(
        RetrieverConfigSpace(
            retriever_type_name=CategoricalParamSpace(
                    choices=["BM25Retriever", "HybridRetriever", "VectorIndexRetriever"],
                )),
        reader_space=LangchainReaderConfigSpace(
            OllamaLLMConfigSpace(
                model_name=CategoricalParamSpace(choices=[model_name]),
                num_ctx = IntParamSpace(low = 10000, high = 10000),
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
            config_space=config_space,
            num_initial_rags=2,
        )
if __name__ == "__main__":
    path_data = Path(PATH_DEFAULT_DATA_DIR)
    # Configure optimization
    seed = load_seeds(num_seeds= 1)[0]
    set_seed(seed)
    path_with_pages = path_data / "eval_llm_2026_with_pages"
    path_chunks = path_with_pages/ "chunks"
    path_filtered_chunks = path_with_pages / "filtered_chunks"
    path_filtered_chunks.mkdir(exist_ok=True)
    path_augmented = path_with_pages / "augmented"
    path_augmented.mkdir(exist_ok=True)
    if AUGMENT:
        for fpath in path_chunks.iterdir():
            print(f"Augmenting {fpath.stem} split...")
            augment(
                fpath,
                path_augmented/ fpath.stem,
                path_filtered_chunks/ fpath.name,
                seed = seed,
            )

    optimizer = get_optimizer(
        path_augmented = path_augmented,
        model_name= "qwen3.5:2b",
        experiment_name= "optim_3",
        seed = seed,
    )
    optimizer.optimize()
