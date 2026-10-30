"""Defines a wrapper around a llama-index reader."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rago.data_objects import GenerationOutput, ReaderInput
from rago.model.configs.reader_config.llama_index import LLamaIndexReaderConfig
from rago.model.constructors.llama_index.reader_factory import ReaderFactory
from rago.model.wrapper.reader.base import Reader
from rago.model.wrapper.tunable_model import BuildContext

if TYPE_CHECKING:
    from llama_index.core.llms import LLM
    from llama_index.core.response_synthesizers.base import BaseSynthesizer
    from llama_index.core.types import RESPONSE_TEXT_TYPE

    from rago.prompts import PromptConfig


class LlamaIndexWrapperReader(Reader[LLamaIndexReaderConfig]):
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
        ctx: Optional[BuildContext] = None,
        *,
        llm: Optional[LLM] = None,
    ) -> LlamaIndexWrapperReader:
        """Build a wrapper around a llama-index reader from the config of the llama-index reader.

        :param config: The config of the llama-index reader to wrap around.
        :type config: LLamaIndexReaderConfig
        :param ctx: Runtime resources (prompt config), defaults to ``BuildContext()``.
        :type ctx: Optional[BuildContext]
        :param llm: The llm used by the reader, defaults to None.
        :type llm: Optional[LLM]
        :return: The wrapper around the prompt.
        :rtype: LlamaIndexWrapperReader
        """
        prompt_config = (ctx or BuildContext()).prompt_config
        prompt_template_str = cls.get_prompt_template_string(prompt_config)
        context_prompt_template = prompt_config.context_template.format(source_context="{context_str}")
        prompt_template_str = prompt_template_str.format(
            context_str=context_prompt_template,
            query_str="{query_str}",
        )
        reader = ReaderFactory.make(config, prompt_template_str, llm)
        return cls(reader, prompt_config)

    def _get_output(self, model_input: ReaderInput) -> GenerationOutput:
        """Get the reader output answer to the input query and optional retrieved context.

        :param query: The query and context the reader needs to answer.
        :type ReaderInput: ReaderInput
        :return: The reader output Generation.
        :rtype: GenerationOutput
        """
        context = [context.text for context in model_input.retrieved_context]
        reader_output = self.lama_index_reader.get_response(model_input.query, context)
        return self.parse_reader_output(reader_output)

    def parse_reader_output(self, reader_output: RESPONSE_TEXT_TYPE) -> GenerationOutput:
        """Parse the output of the reader to obtain a string answer.

        :param reader_output: The raw reader output.
        :type reader_output: RESPONSE_TEXT_TYPE
        :return: The string answer.
        :rtype: str
        """
        match reader_output:
            case str():
                return GenerationOutput(reader_output)
            case _:
                answer = "Invalid answer."
                return GenerationOutput(answer)
