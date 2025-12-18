"""Define the integration tests for the reader."""

import os

import pytest
from langchain_core.language_models import BaseLLM
from llama_index.core.llms import LLM

from rago.data_objects import RetrievedContext
from rago.model.configs.reader_config.langchain import LangchainReaderConfig
from rago.model.configs.reader_config.llama_index import LLamaIndexReaderConfig
from rago.model.wrapper.reader.langchain_reader import LangchainWrapperReader
from rago.model.wrapper.reader.llama_index_reader import LlamaIndexWrapperReader

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test ignored in GitHub Actions")
def test_llama_index_response_synthesizer_works(
    llama_index_llm: LLM,
) -> None:
    """Tests that the reader using a llama_index response synthesizer works correctly."""
    nodes = [RetrievedContext("The date is 2012"), RetrievedContext("Thomas is going to be 12 in 2013")]
    config = LLamaIndexReaderConfig(type="CompactAndRefine")
    reader = LlamaIndexWrapperReader.make(config, llm=llama_index_llm)
    assert reader.get_reader_output("How old is thomas?", nodes) == "In 2012, Thomas will be 12 years old."


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test ignored in GitHub Actions")
def test_base_reader_works(langchain_llm: BaseLLM) -> None:
    """Tests that the reader using a language model works correctly."""
    reader_config = LangchainReaderConfig()
    reader = LangchainWrapperReader.make(reader_config, langchain_llm)
    nodes = [RetrievedContext("The date is 2012"), RetrievedContext("Thomas is going to be 12 in 2013")]
    assert reader.get_reader_output("How old is thomas?", nodes) == "In this scenario, we can use the given context"
