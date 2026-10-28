"""Define a LlamaIndex reader wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rago.data_objects import GenerationOutput, ReaderInput
from rago.model.configs.reader_config.langchain import LangchainReaderConfig
from rago.model.constructors.langchain.llm_factory import LangchainLLMFactory
from rago.model.wrapper.reader.base import Reader
from rago.model.wrapper.tunable_model import BuildContext

if TYPE_CHECKING:
    from langchain_core.language_models import BaseLLM

    from rago.prompts import PromptConfig


class LangchainWrapperReader(Reader[LangchainReaderConfig]):
    """Wrapper to around a Langchain reader."""

    def __init__(self, langchain_reader: BaseLLM, prompt_config: PromptConfig, prompt_template: str) -> None:
        """Instantiate a Langchain LLM wrapper.

        :param langchain_reader: THe langchain, reader to wrap.
        :type langchain_reader: Runnable
        """
        self.langchain_reader = langchain_reader
        self.prompt_config = prompt_config
        self.prompt_template = prompt_template

    @classmethod
    def make(
        cls,
        config: LangchainReaderConfig,
        ctx: Optional[BuildContext] = None,
        *,
        llm: Optional[BaseLLM] = None,
    ) -> LangchainWrapperReader:
        """Build a wrapper around a Langchain reader.

        :param config: The config of the langchain reader to wrap around.
        :type config: LangchainReaderConfig
        :param ctx: Runtime resources (prompt config), defaults to ``BuildContext()``.
        :type ctx: Optional[BuildContext]
        :param llm: The llm used by the reader, built from ``config.llm`` if None.
        :type llm: Optional[BaseLLM]
        :return: The wrapper around the reader.
        :rtype: LangchainWrapperReader
        """
        if llm is None:
            if config.llm is None:
                raise ValueError(config.llm)
            llm = LangchainLLMFactory.make(config.llm)

        prompt_config = (ctx or BuildContext()).prompt_config
        prompt_template_str = cls.get_prompt_template_string(prompt_config)

        return LangchainWrapperReader(
            langchain_reader=llm,
            prompt_config=prompt_config,
            prompt_template=prompt_template_str,
        )

    def _get_output(self, model_input: ReaderInput) -> GenerationOutput:
        """Get the reader output answer to the input query and optional retrieved context.

        :param query: The query the reader needs to answer.
        :type query: str
        :param retrieved_context: The context retrieved to help answer the query if set, defaults to None.
        :type retrieved_context: Optional[list[str]], optional
        :return: The reader output answer.
        :rtype: str
        """
        context_prompt = self.prompt_config.context_template.format(
                source_context="\n".join([context.text for context in model_input.retrieved_context]),
            )

        prompt = self.prompt_template.format(query_str=model_input.query, context_str=context_prompt)
        string_answer = self.langchain_reader.invoke(prompt)

        return GenerationOutput(string_answer)
