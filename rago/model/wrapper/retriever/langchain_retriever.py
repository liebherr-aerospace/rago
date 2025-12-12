"""Define a retriever using langchain retrievers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from langchain_core.documents import Document

from rago.data_objects import RetrievedContext
from rago.model.constructors.langchain.retriever_factory.base_factory import RetrieverFactory
from rago.model.wrapper.context_post_processor.llama_index_post_processor import LLamaIndexContextPostProcessorWrapper
from rago.model.wrapper.retriever.base import Retriever

if TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever

    from rago.model.configs.retriever_config.langchain import LangchainRetrieverConfig


class LangchainRetrieverWrapper(Retriever):
    """A retriever that retrieves information relevant to answer a query from a database (non-parametric knowledge)."""

    def __init__(
        self,
        langchain_retriever: BaseRetriever,
        nodes_post_processors: Optional[LLamaIndexContextPostProcessorWrapper] = None,
    ) -> None:
        """Instantiate a retriever with a Langchain retriever and optionally post-processors.

        :param langchain_retriever: The Langchain retriever to use.
        :type langchain_retriever: BaseRetriever
        :param nodes_post_processors: The context post-processors nodes to use, defaults to None
        :type nodes_post_processors: Optional[LLamaIndexContextPostProcessorWrapper], optional
        """
        self.langchain_retriever = langchain_retriever
        self.nodes_post_processors = nodes_post_processors

    @classmethod
    def make(
        cls,
        config: LangchainRetrieverConfig,
        inputs_chunks: list[str],
    ) -> LangchainRetrieverWrapper:
        """Generate a retriever from a documents set, an encoder, a llm and the config of the retriever to build.

        :param config: The config of the retriever to generate.
        :type config: dict
        :param inputs_chunks: the document that will be used to generate the non parametric knowledge.
        :type inputs_chunks: list[str]
        :return: The retriever corresponding to the config dict.
        :rtype: LangchainRetrieverWrapper
        """
        langchain_inputs_chunks = [Document(chunk) for chunk in inputs_chunks]

        retriever = RetrieverFactory.make(config, langchain_inputs_chunks)
        if config.node_post_processor_config is not None:
            nodes_post_processors = LLamaIndexContextPostProcessorWrapper.make(
                config.node_post_processor_config,
            )
        else:
            nodes_post_processors = None

        return cls(retriever, nodes_post_processors)

    def get_retriever_output(self, query: str) -> list[RetrievedContext]:
        """Get the texts relevant to the query from the retriever database.

        :param query: The query.
        :type query: str
        :return: The texts relevant to the query from the retriever database.
        :rtype: list[str]
        """
        nodes_with_scores = self.langchain_retriever.invoke(query)
        retrieved_context = [
            RetrievedContext.from_langchain_node(node_with_score) for node_with_score in nodes_with_scores
        ]
        if self.nodes_post_processors is not None:
            retrieved_context = self.nodes_post_processors.post_process_context(query, retrieved_context)
        return retrieved_context
