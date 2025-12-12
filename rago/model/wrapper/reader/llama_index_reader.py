"""Defines a wrapper around a llama-index reader."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rago.model.constructors.llama_index.reader_factory import LLamaIndexReaderConfig, ReaderFactory
from rago.model.wrapper.reader.base import Reader
from rago.prompts import PromptConfig

if TYPE_CHECKING:
    from llama_index.core.llms import LLM
    from llama_index.core.response_synthesizers.base import RESPONSE_TEXT_TYPE, BaseSynthesizer

    from rago.data_objects import RetrievedContext


class LlamaIndexWrapperReader(Reader):
    """Defines a wrapper around a llama-index reader."""

    lama_index_reader: BaseSynthesizer
    prompt_config: PromptConfig

    def __init__(self, lama_index_reader: BaseSynthesizer, prompt_config: PromptConfig) -> None:
        """Initialize a wrapper around a llama index reader from the llama_index_reader and its prompt_config.

        :param lama_index_reader: the llama-index reader to wrap around.
        :type lama_index_reader: BaseSynthesizer
        :param prompt_config: The prompt configuration used by the wrapper.
        :type prompt_config: PromptConfig
        """
        self.lama_index_reader = lama_index_reader
        self.prompt_config = prompt_config

    @classmethod
    def make(
        cls,
        config: LLamaIndexReaderConfig,
        llm: Optional[LLM] = None,
        prompt_config: Optional[PromptConfig] = None,
    ) -> LlamaIndexWrapperReader:
        """Build a wrapper around a llama-index reader from the config of the llama-index reader.

        :param config: The config of the llama-index reader to wrap around.
        :type config: dict
        :param llm: The llm used by the reader.
        :type llm: LLM
        :param prompt_config: The configuration for the prompt, defaults to None
        :type prompt_config: Optional[PromptConfig ], optional
        :return: The wrapper around the prompt.
        :rtype: LlamaIndexWrapperReader
        """
        prompt_config = prompt_config if prompt_config is not None else PromptConfig()
        prompt_template_str = cls.get_prompt_template_string(prompt_config)
        context_prompt_template = prompt_config.context_template.format(source_context="{context_str}")
        prompt_template_str = prompt_template_str.format(
            context_str=context_prompt_template,
            query_str="{query_str}",
        )
        reader = ReaderFactory.make(config, prompt_template_str, llm)
        return cls(reader, prompt_config)

    def get_reader_output(self, query: str, retrieved_context: Optional[list[RetrievedContext]] = None) -> str:
        """Get the reader output answer to the input query and optional retrieved context.

        :param query: The query the reader needs to answer.
        :type query: str
        :param retrieved_context: The context retrieved to help answer the query if set, defaults to None.
        :type retrieved_context: Optional[list[str]], optional
        :return: The reader output answer.
        :rtype: str
        """
        context = [context.text for context in retrieved_context] if retrieved_context is not None else []
        reader_output = self.lama_index_reader.get_response(query, context)
        return self.parse_reader_output(reader_output)

    def parse_reader_output(self, reader_output: RESPONSE_TEXT_TYPE) -> str:
        """Parse the output of the reader to obtain a string answer.

        :param reader_output: The raw reader output.
        :type reader_output: RESPONSE_TEXT_TYPE
        :return: The string answer.
        :rtype: str
        """
        match reader_output:
            case str():
                return reader_output
            case _:
                answer = "Invalid answer."
                return answer
