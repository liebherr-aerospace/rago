"""Define the pytest.fixture used by the different test."""

import os
from pathlib import Path
from typing import cast

import pytest
from langchain_core.language_models import BaseLLM
from llama_index.core.llms import LLM

from rago.data_objects import Document
from rago.dataset import QADatasetLoader, RAGDataset
from rago.model.configs.llm_config.langchain import LangchainOllamaConfig
from rago.model.configs.llm_config.llama_index import LlamaIndexOllamaConfig
from rago.model.constructors.langchain.llm_factory import LangchainLLMFactory
from rago.model.constructors.llama_index.llm_factory import LLamaIndexLLMFactory
from rago.model.wrapper.llm_agent.langchain import LangchainLLMAgent
from rago.model.wrapper.llm_agent.llama_index import LlamaIndexLLMAgent


@pytest.fixture(scope="session")
def documents() -> list[Document]:
    """Return a random fake document set to retrieve from.

    The documents were generated once and are read from a file to ensure reproducibility.
    """
    lines_number = 10
    with Path("./tests/assets/fake_documents.txt").open("r") as f:
        read_documents_text = [next(f) for _ in range(lines_number)]
    return [Document(text=text) for text in read_documents_text]


@pytest.fixture(scope="session")
def hotpot_qa_sample() -> RAGDataset:
    """Return the dataset that the simple dataset generator should generate in the deterministic case."""
    dataset = QADatasetLoader.load_from_disk(RAGDataset, "./tests/assets/hotpot_qa_test_sample.json")
    dataset = cast(RAGDataset, dataset)
    return dataset


@pytest.fixture(scope="session")
def expected_dataset() -> RAGDataset:
    """Return the dataset that the simple dataset generator should generate in the deterministic case."""
    dataset = QADatasetLoader.load_from_disk(RAGDataset, "./tests/assets/expected_dataset.json")
    dataset = cast(RAGDataset, dataset)
    return dataset


@pytest.fixture(scope="session")
def expected_results_storage_db() -> str:
    """Get the path to the optim results storage.

    :return: The path to the directory.
    :rtype: str
    """
    return "sqlite:///tests/assets/test_experiments/test_experiment_0001/study.db"


@pytest.fixture(scope="session")
def base_url() -> str:
    """Return Url of the Ollama server used for test."""
    return os.environ.get("TEST_OLLAMA_HOST", "")


@pytest.fixture(scope="session")
def language_model_name() -> str:
    """Return the Name of the language model.

    Using smollm:1.7b as it is a small LM and smaller LM seem to suffer from difficulties to finish answer
    i.e. do not generate eos token.
    """
    return "smollm:1.7b"


@pytest.fixture(scope="session")
def language_model_name_for_optim() -> str:
    """Return the Name of the language model used ofr optimization."""
    return "phi3:3.8b-mini-128k-instruct-q8_0"


@pytest.fixture(scope="session")
def temperature() -> float:
    """Return Temperature used by the language model for inference.

    The temperature is characterize the creativity of the language models.
    More Technically the embedding vectors are divided by the temperature before going in the softmax.
    Here the temperature is set to 0 so that the LM are deterministic (i.e always produce the same outputs.)
    This is very useful for the tests as it enables checking that the output is correct.
    Theoretically another solution is to use a fixed seed but this might not unsure that the results are reproducible.
    """
    return 0.0


@pytest.fixture(scope="session")
def langchain_llm(language_model_name: str, temperature: float) -> BaseLLM:
    """Return a langchain llm."""
    langchain_llm_config = LangchainOllamaConfig(model_name=language_model_name, temperature=temperature)
    return LangchainLLMFactory.make(langchain_llm_config)


@pytest.fixture(scope="session")
def langchain_llm_agent(langchain_llm: BaseLLM) -> LangchainLLMAgent:
    """Return Langchain language model agent.

    :param langchain_llm: The langchain model to use for the agent.
    :type langchain_llm: BaseLLM
    :return: The Langchain language model agent.
    """
    llm_agent = LangchainLLMAgent(langchain_llm)

    return llm_agent


@pytest.fixture(scope="session")
def llama_index_llm(language_model_name: str, temperature: float) -> LLM:
    """Return a Ollama Langchain language model.

    To ensure the returned model is deterministic the model is queried before being returned.
    As there is a discrepancy between first and subsequent queries.

    :param ollama_url: Url to the Ollama Server.
    :type ollama_url: str
    :param language_model_name: Name of the language model.
    :type language_model_name: str
    :param temperature: Temperature used by the language model for inference.
    :type temperature: float
    """
    langchain_llm_config = LlamaIndexOllamaConfig(model_name=language_model_name, temperature=temperature)
    llama_index_ollama = LLamaIndexLLMFactory.make(
        config=langchain_llm_config,
    )
    return llama_index_ollama


@pytest.fixture(scope="session")
def llama_index_llm_agent(llama_index_llm: LLM) -> LlamaIndexLLMAgent:
    """Return LlamaIndex language model agent.

    :param llama_index_llm: The LlamaIndex model to use for the agent.
    :type llama_index_llm: LLM
    :return: The LlamaIndex language model agent.
    """
    llm_agent = LlamaIndexLLMAgent(llama_index_llm)
    return llm_agent
