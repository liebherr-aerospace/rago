"""Define a retriever wrapper factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rago.model.configs.retriever_config.langchain import HybridLangchainRetrieverConfig, LangchainRetrieverConfig
from rago.model.configs.retriever_config.llama_index import LlamaIndexRetrieverConfig
from rago.model.configs.retriever_config.qdrant import QdrantRetrieverConfig
from rago.model.wrapper.retriever.hybrid_langchain_retriever import HybridLangchainRetrieverWrapper
from rago.model.wrapper.retriever.langchain_retriever import LangchainRetrieverWrapper
from rago.model.wrapper.retriever.llama_index_retriever import LlamaIndexRetrieverWrapper
from rago.model.wrapper.retriever.qdrant_retriever import QdrantRetrieverWrapper

if TYPE_CHECKING:
    from rago.model.configs.retriever_config.base import RetrieverConfig
    from rago.model.wrapper.retriever.base import Retriever
    from rago.model.wrapper.tunable_model import BuildContext


class RetrieverWrapperFactory:
    """Factory of retriever wrappers."""

    @staticmethod
    def make(config: RetrieverConfig, ctx: Optional[BuildContext] = None) -> Retriever:
        """Make a retriever wrapper.

        :param config: The config of the retriever wrapper to make.
        :type config: RetrieverConfig
        :param ctx: Runtime resources (input chunks, not needed for Qdrant), defaults to None.
        :type ctx: Optional[BuildContext]
        :raises TypeError: If the config type is not supported.
        :return: The created retriever wrapper.
        :rtype: Retriever
        """
        match config:
            # Hybrid first: it is a subclass of LangchainRetrieverConfig.
            case HybridLangchainRetrieverConfig():
                return HybridLangchainRetrieverWrapper.make(config, ctx)
            case LangchainRetrieverConfig():
                return LangchainRetrieverWrapper.make(config, ctx)
            case LlamaIndexRetrieverConfig():
                return LlamaIndexRetrieverWrapper.make(config, ctx)
            case QdrantRetrieverConfig():
                return QdrantRetrieverWrapper.make(config, ctx)
            case _:
                raise TypeError(config)
