"""Define a retriever wrapper factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rago.model.configs.retriever_config.langchain import LangchainRetrieverConfig
from rago.model.configs.retriever_config.llama_index import LlamaIndexRetrieverConfig
from rago.model.configs.retriever_config.qdrant import QdrantRetrieverConfig
from rago.model.wrapper.retriever.hybrid_langchain_retriever import HybridLangchainRetrieverWrapper
from rago.model.wrapper.retriever.langchain_retriever import LangchainRetrieverWrapper
from rago.model.wrapper.retriever.llama_index_retriever import LlamaIndexRetrieverWrapper
from rago.model.wrapper.retriever.qdrant_retriever import QdrantRetrieverWrapper

if TYPE_CHECKING:
    from collections.abc import Callable

    from rago.model.configs.retriever_config.base import RetrieverConfig
    from rago.model.wrapper.retriever.base import Retriever


def _make_qdrant(config: QdrantRetrieverConfig, _input_chunks: list[str]) -> QdrantRetrieverWrapper:
    return QdrantRetrieverWrapper.make(config)


def _make_langchain(config: LangchainRetrieverConfig, input_chunks: list[str]) -> Retriever:
    if config.type == "HybridRetriever":
        return HybridLangchainRetrieverWrapper.make(config, input_chunks)
    return LangchainRetrieverWrapper.make(config, input_chunks)


_REGISTRY: dict[type[RetrieverConfig], Callable[..., Retriever]] = {
    QdrantRetrieverConfig: _make_qdrant,
    LlamaIndexRetrieverConfig: LlamaIndexRetrieverWrapper.make,
    LangchainRetrieverConfig: _make_langchain,
}


class RetrieverWrapperFactory:
    """Factory of retriever wrappers."""

    @staticmethod
    def make(config: RetrieverConfig, input_chunks: list[str]) -> Retriever:
        """Make a retriever wrapper.

        :param config: The config of the retriever wrapper to make.
        :type config: RetrieverConfig
        :param input_chunks: The input chunks used by the retriever wrapper.
        :type input_chunks: list[str]
        :return: The created retriever wrapper.
        :rtype: Retriever
        """
        factory = _REGISTRY.get(type(config))
        if factory is None:
            raise TypeError(config)
        return factory(config, input_chunks)
