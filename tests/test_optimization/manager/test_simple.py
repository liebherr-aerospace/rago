"""Test the Optimization Results Repository."""

import os
from pathlib import Path

import optuna
import pytest

from rago.data_objects.document import Document
from rago.data_objects.rag_config import RAGConfig
from rago.dataset import RAGDataset
from rago.dataset.generator import DatasetGeneratorConfig
from rago.eval.llm_evaluator.base import LLMEvaluatorConfig
from rago.eval.llm_evaluator.simple import SimpleLLMEvaluator
from rago.model.configs.llm_config.langchain import LangchainOllamaConfig
from rago.optimization.manager.base import OptimParams
from rago.optimization.manager.simple import SimpleDirectOptunaManager
from rago.optimization.search_space.rag_config_space import RAGConfigSpace
from rago.optimization.search_space.reader_config_space import LangchainReaderConfigSpace
from rago.optimization.search_space.retriever_config_space import RetrieverConfigSpace

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

DOCUMENTS = [
    Document(
        "Warcraft II: is a fantasy real-time strategy computer game developed by Blizzard Entertainment.",
    ),
    Document(
        "Nier: Automata a 2017 action role-playing game developed by PlatinumGames and published by Square Enix.",
    ),
]
SEED_DATASET = RAGDataset([], {doc.id: doc for doc in DOCUMENTS})


@pytest.fixture
def simple_manager(
    language_model_name: str,
) -> SimpleDirectOptunaManager:
    """Define the Base class for the simple direct optimization manager."""
    llm_config = LangchainOllamaConfig(model_name=language_model_name, temperature=0.0)
    params = OptimParams(
        n_iter=3,
        path_experiments="tests/assets/test_experiments/",
        experiment_name="test_experiment_0001",
    )
    dataset_generator_config = DatasetGeneratorConfig(
        llm=llm_config,
    )
    evaluator_config = LLMEvaluatorConfig(
        judge=llm_config,
        min_score=0,
        max_score=7,
    )
    evaluator = SimpleLLMEvaluator.make(evaluator_config)

    retriever_space = RetrieverConfigSpace()
    reader_space = LangchainReaderConfigSpace()
    rag_config_space = RAGConfigSpace(retriever_space=retriever_space, reader_space=reader_space)
    simple_direct_manager = SimpleDirectOptunaManager.from_seed_data(
        params=params,
        config_space=rag_config_space,
        optim_evaluator=evaluator,
        optim_metric_name="correctness",
        test_evaluators=[evaluator],
        seed_data=SEED_DATASET,
        splits=(0.5, 0.5),
        dataset_generator_config=dataset_generator_config,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )
    return simple_direct_manager


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test ignored in GitHub Actions")
def test_load_database_results(simple_manager: SimpleDirectOptunaManager) -> None:
    """Test if results have been generated and stored."""
    simple_manager.optimize()
    loaded_study = simple_manager.load_results()
    assert loaded_study is not None


def test_result_database_exist(
    simple_manager: SimpleDirectOptunaManager,
    expected_results_storage_db: str,
) -> None:
    """Test if the database that stores the results of the experiment exists."""
    storage_name = simple_manager.experiment_repo.get_storage_name()
    assert storage_name == expected_results_storage_db


def test_get_n_best_trials(simple_manager: SimpleDirectOptunaManager) -> None:
    """Test if we get the n best trials."""
    best_trials = simple_manager.get_n_best_trials(2)
    assert all(trial.value is not None for trial in best_trials)


def test_get_n_best_rag_configs(simple_manager: SimpleDirectOptunaManager) -> None:
    """Test if we get the n best rag configuration."""
    best_rag_configs = simple_manager.get_n_best_rag_configs(2)
    assert all(isinstance(config, RAGConfig) for config in best_rag_configs)


def test_log_file_exists_with_fixture() -> None:
    """Test if log file is present in the directory."""
    log_file_path = Path("tests/assets/test_experiments/test_experiment_0001/test_experiment_0001.log")
    assert log_file_path.exists(), f"Log file {log_file_path} does not exist."
