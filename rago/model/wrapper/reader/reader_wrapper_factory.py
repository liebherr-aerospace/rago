"""Define a factory of reader wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rago.model.configs.reader_config.langchain import LangchainReaderConfig
from rago.model.wrapper.reader.langchain_reader import LangchainWrapperReader
from rago.model.wrapper.reader.llama_index_reader import LLamaIndexReaderConfig, LlamaIndexWrapperReader

if TYPE_CHECKING:
    from rago.model.configs.reader_config.base import ReaderConfig
    from rago.model.wrapper.reader.base import Reader
    from rago.prompts import PromptConfig


class ReaderWrapperFactory:
    """Factory of reader wrappers."""

    @staticmethod
    def make(
        config: ReaderConfig,
        prompt_config: Optional[PromptConfig] = None,
    ) -> Reader:
        """Make a reader wrapper.

        :param config: The config of the reader wrapper to make.
        :type config: ReaderConfig
        :param prompt_config: The configuration of the prompt used by the reader wrapper.
        :type prompt_config: PromptConfig
        :return: The created reader wrapper.
        :rtype: Reader
        """
        match config:
            case LLamaIndexReaderConfig():
                return LlamaIndexWrapperReader.make(config, prompt_config=prompt_config)
            case LangchainReaderConfig():
                return LangchainWrapperReader.make(config, prompt_config=prompt_config)
            case _:
                raise TypeError(config)
