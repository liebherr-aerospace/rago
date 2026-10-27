"""Define a factory of reader wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rago.model.configs.reader_config.langchain import LangchainReaderConfig
from rago.model.configs.reader_config.llama_index import LLamaIndexReaderConfig
from rago.model.wrapper.reader.langchain_reader import LangchainWrapperReader
from rago.model.wrapper.reader.llama_index_reader import LlamaIndexWrapperReader

if TYPE_CHECKING:
    from rago.model.configs.reader_config.base import ReaderConfig
    from rago.model.wrapper.reader.base import Reader
    from rago.model.wrapper.tunable_model import BuildContext


class ReaderWrapperFactory:
    """Factory of reader wrappers."""

    @staticmethod
    def make(config: ReaderConfig, ctx: Optional[BuildContext] = None) -> Reader:
        """Make a reader wrapper.

        :param config: The config of the reader wrapper to make.
        :type config: ReaderConfig
        :param ctx: Runtime resources (prompt config), defaults to ``BuildContext()``.
        :type ctx: Optional[BuildContext]
        :raises TypeError: If the config type is not supported.
        :return: The created reader wrapper.
        :rtype: Reader
        """
        match config:
            case LLamaIndexReaderConfig():
                return LlamaIndexWrapperReader.make(config, ctx)
            case LangchainReaderConfig():
                return LangchainWrapperReader.make(config, ctx)
            case _:
                raise TypeError(config)
