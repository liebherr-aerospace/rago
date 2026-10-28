from pathlib import Path
from typing import cast

import optuna
import requests
from pydantic.dataclasses import dataclass
from transformers import set_seed
from urllib3.exceptions import InsecureRequestWarning

from rago.dataset import RAGDataset
from rago.eval.llm_evaluator import PolicyOnError
from rago.model.wrapper.rag.base import RAG, RAGConfig
from rago.optimization.manager import (
    BaseOptunaManager,
    OptimParams,
    SimpleDirectOptunaManager,
    SimplePairWiseOptunaManager,
)
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
    PROMPT_CONTEXTE,
    PROMPT_DIRECT_EVAL,
    PROMPT_DUAL_EVAL,
    PROMPT_EVAL_PARTIEL,
    PROMPT_SYSTEME,
)
from scripts.seeds import load_seeds

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

def get_optimizer(
        optimizer_class: type[BaseOptunaManager],
        seed: int,
        datasets: dict[str, RAGDataset],
        model_name: str,
        evaluation_prompt: str,
        pairwise_evaluation_prompt: str,
        save_path: Path,
        ) -> BaseOptunaManager:
    """Make optimizer from inputs."""
    set_seed(seed)
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,
        seed=seed,
        multivariate=True,
        group = True,
        consider_prior=True)
    params = OptimParams(
        experiment_name=str(seed),
        path_experiments=str(save_path),
        n_startup_trials=10,
        n_iter=100,
    )
    evaluator = get_judge(
        seed = seed,
        model_name = model_name,
        policy_on_error=PolicyOnError.MIN_SCORE,
        evaluation_prompt = evaluation_prompt,
        pairwise_evaluation_prompt= pairwise_evaluation_prompt)

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
    return  optimizer_class(
            params=params,
            datasets=datasets,
            optim_evaluator=evaluator,
            optim_metric_name="correctness",
            test_evaluators=[evaluator],
            prompt_config=rag_prompt,
            sampler = sampler,
            model_class = RAG,
            config_space=config_space,
        )

def evaluate_optim_rag(
        optimizer_class: type[BaseOptunaManager],
        evaluation_prompt: str,
        pairwise_evaluation_prompt: str,
        seed: int,
        datasets: dict[str, RAGDataset],
        save_path: Path,
        model_name: str = "qwen3.5:2b",
        ) -> None:
    """Evaluate the rag resulting from optimization."""
    optimizer = get_optimizer(
        optimizer_class,
        seed,
        datasets,
        model_name,
        evaluation_prompt,
        pairwise_evaluation_prompt,
        save_path)
    optimizer.optimize()
    best_trials = optimizer.get_n_best_trials(number=5)
    inputs_chunks = [doc.text for doc in datasets["test"].corpus.values()]
    for idx, trial in enumerate(best_trials):
        config: RAGConfig = optimizer.config_space.sample(trial)
        rag = RAG.make(
                config=config,
                prompt_config=optimizer.prompt_config,
                inputs_chunks=inputs_chunks,
            )
        eval_file_name = f"test_trial_{idx}.json"
        seed_dir = save_path/str(seed)
        seed_dir.mkdir(exist_ok=True, parents = True)
        eval_model(rag, get_judge(seed), datasets["test"], seed_dir / eval_file_name)

@dataclass
class RAGOptimConfig:
    """Optimization Evaluation Config."""

    optim_class: type[BaseOptunaManager]
    save_file: str
    evaluation_prompt: str
    pairwise_evaluation_prompt: str

def run_optim(
        config: RAGOptimConfig,
        seeds: list[int],
        ) -> None:
    """Run the Optimization Evaluation over multiple seeds."""
    path_data = Path(PATH_DEFAULT_DATA_DIR)
    path_augmented = path_data/"eval_llm_2026_with_pages/augmented"
    path_optim = Path(PATH_PROJECT, "experiments", "baselines", config.save_file)
    for s in seeds:

        datasets = cast("dict[str, RAGDataset]", RAGDataset.load_dataset(path_augmented))
        datasets["train"]= datasets["train"].sample(200, seed = s)
        datasets["test"]= datasets["test"].sample(600, seed = s)
        evaluate_optim_rag(
            optimizer_class = config.optim_class,
            evaluation_prompt=config.evaluation_prompt,
            pairwise_evaluation_prompt=config.pairwise_evaluation_prompt,
            seed = s,
            datasets = datasets,
            save_path= path_optim)


OPTIM_CONFIFGS = [
    RAGOptimConfig(
        optim_class = SimpleDirectOptunaManager,
        save_file = "direct_optim",
        evaluation_prompt= PROMPT_DIRECT_EVAL,
        pairwise_evaluation_prompt= PROMPT_DUAL_EVAL,
        ),
    RAGOptimConfig(
        optim_class = SimplePairWiseOptunaManager,
        save_file = "pairwise_optim",
        evaluation_prompt= PROMPT_EVAL_PARTIEL,
        pairwise_evaluation_prompt= PROMPT_DUAL_EVAL),
]
if __name__ == "__main__":
    num_seeds = 5

    seeds = load_seeds(num_seeds= num_seeds)

    run_optim(OPTIM_CONFIFGS[0], seeds[3:])
    run_optim(OPTIM_CONFIFGS[1], seeds)

