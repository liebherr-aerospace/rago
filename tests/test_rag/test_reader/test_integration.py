"""Define the integration tests for the reader."""

import os

from langchain_core.language_models import BaseLLM
from llama_index.core.llms import LLM

from rago.data_objects import GenerationOutput, ReaderInput, RetrievedContext
from rago.model.configs.reader_config.langchain import LangchainReaderConfig
from rago.model.configs.reader_config.llama_index import LLamaIndexReaderConfig
from rago.model.wrapper.reader.langchain_reader import LangchainWrapperReader
from rago.model.wrapper.reader.llama_index_reader import LlamaIndexWrapperReader

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def test_llama_index_response_synthesizer_works(
    llama_index_llm: LLM,
) -> None:
    """Tests that the reader using a llama_index response synthesizer works correctly."""
    nodes = [RetrievedContext("The date is 2012"), RetrievedContext("Thomas is going to be 12 in 2013")]
    config = LLamaIndexReaderConfig(type="CompactAndRefine")
    reader = LlamaIndexWrapperReader.make(config, llm=llama_index_llm)
    out =  reader.get_output(ReaderInput("How old is thomas?", nodes))
    assert isinstance(out, GenerationOutput)
    assert len(out.answer)>0


def test_base_reader_works(langchain_llm: BaseLLM) -> None:
    """Tests that the reader using a language model works correctly."""
    reader_config = LangchainReaderConfig()
    reader = LangchainWrapperReader.make(reader_config, llm=langchain_llm)
    nodes = [RetrievedContext("The date is 2012"), RetrievedContext("Thomas is going to be 12 in 2013")]
    out = reader.get_output(ReaderInput("How old is thomas?", nodes))
    assert isinstance(out, GenerationOutput)
    assert len(out.answer)>0
