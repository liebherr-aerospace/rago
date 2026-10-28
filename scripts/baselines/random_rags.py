"""Evaluate Random RAG configs."""
from pathlib import Path
from typing import cast

import optuna
import requests
from urllib3.exceptions import InsecureRequestWarning

from rago.dataset import RAGDataset
from rago.model.wrapper.rag.base import RAG
from rago.optimization.search_space.llm_config_space import OllamaLLMConfigSpace
from rago.optimization.search_space.param_space import CategoricalParamSpace, IntParamSpace
from rago.optimization.search_space.rag_config_space import RAGConfigSpace
from rago.optimization.search_space.reader_config_space import LangchainReaderConfigSpace
from rago.optimization.search_space.retriever_config_space import RetrieverConfigSpace
from rago.prompts import PromptConfig
from rago.utils import PATH_DEFAULT_DATA_DIR, PATH_PROJECT
from scripts.baselines.utils import get_judge
from scripts.eval import eval_model_seeds
from scripts.prompts import (
    PROMPT_CONTEXTE,
    PROMPT_SYSTEME,
)
from scripts.seeds import load_seeds

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
def get_rag(seed: int, corpus: list[str], model_name: str = "qwen3.5:2b") -> RAG:
    """Make random rag from seed, corpus and model name."""
    study = optuna.create_study(sampler = optuna.samplers.RandomSampler(seed))
    trial = study.ask()
    config_space = RAGConfigSpace(
        RetrieverConfigSpace(
            retriever_type_name=CategoricalParamSpace(
                    choices=["BM25Retriever", "HybridRetriever", "VectorIndexRetriever"],
                )),
        reader_space=LangchainReaderConfigSpace(
            OllamaLLMConfigSpace(
                model_name=CategoricalParamSpace(choices=[model_name]),
                num_ctx = IntParamSpace(low = 10000, high = 10000),
                seed = CategoricalParamSpace(choices=[seed]),
            ),
        ),
    )
    rag_prompt = PromptConfig(
        system_message=PROMPT_SYSTEME,
        context_template=PROMPT_CONTEXTE,

    )
    config = config_space.sample(trial)
    print(config)
    return RAG.make(
        config,
        rag_prompt,
        inputs_chunks=corpus,
    )

def run_random_rags(num_seeds: int | None = 10, seeds: list[int] | None = None):
    path_data = Path(PATH_DEFAULT_DATA_DIR)
    path_augmented = path_data/"eval_llm_2026_with_pages/augmented"
    path_rag_random = Path(PATH_PROJECT, "experiments", "baselines", "random_rags")
    dataset = cast("dict[str, RAGDataset]", RAGDataset.load_dataset(path_augmented))["test"]
    corpus = [d.text for d in dataset.corpus.values()]
    eval_model_seeds(
        make_model_from_seed = lambda seed : get_rag(seed, corpus),
        make_judge=get_judge,
        dataset = dataset,
        num_samples= 600,
        save_dir = path_rag_random,
        num_seeds=num_seeds,
        seeds = seeds)

if __name__ == "__main__":
    seeds = load_seeds(num_seeds=10)[7:]
    run_random_rags(num_seeds = None, seeds = seeds)

