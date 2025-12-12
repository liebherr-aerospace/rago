"""Define a LlamaIndex reader wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rago.model.constructors.langchain.llm_factory import LangchainLLMFactory
from rago.model.wrapper.reader.base import Reader
from rago.prompts import PromptConfig

if TYPE_CHECKING:
    from langchain_core.language_models import BaseLLM

    from rago.data_objects import RetrievedContext
    from rago.model.configs.reader_config.langchain import LangchainReaderConfig


class LangchainWrapperReader(Reader):
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
        llm: Optional[BaseLLM] = None,
        prompt_config: Optional[PromptConfig] = None,
    ) -> LangchainWrapperReader:
        """Build a wrapper around a Langchain reader.

        :param config: The config of the langchain reader to wrap around.
        :type config: dict
        :param llm: The llm used by the reader.
        :type llm: LLM
        :return: The wrapper around the reader.
        :rtype: LangchainWrapperReader
        """
        if llm is None:
            if config.llm is None:
                raise ValueError(config.llm)
            llm = LangchainLLMFactory.make(config.llm)

        prompt_config = PromptConfig() if prompt_config is None else prompt_config
        prompt_template_str = cls.get_prompt_template_string(prompt_config)

        return LangchainWrapperReader(
            langchain_reader=llm,
            prompt_config=prompt_config,
            prompt_template=prompt_template_str,
        )

    def get_reader_output(self, query: str, retrieved_context: Optional[list[RetrievedContext]] = None) -> str:
        """Get the reader output answer to the input query and optional retrieved context.

        :param query: The query the reader needs to answer.
        :type query: str
        :param retrieved_context: The context retrieved to help answer the query if set, defaults to None.
        :type retrieved_context: Optional[list[str]], optional
        :return: The reader output answer.
        :rtype: str
        """
        context_prompt = (
            ""
            if retrieved_context is None
            else self.prompt_config.context_template.format(
                source_context="\n".join([context.text for context in retrieved_context]),
            )
        )

        prompt = self.prompt_template.format(query_str=query, context_str=context_prompt)
        string_answer = self.langchain_reader.invoke(prompt)

        return string_answer
