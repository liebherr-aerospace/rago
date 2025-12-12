"""Define a retriever using llama_index retrievers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from llama_index.core.schema import TextNode

from rago.data_objects import RetrievedContext
from rago.model.constructors.llama_index.retriever_factory import RetrieverFactory
from rago.model.wrapper.context_post_processor.llama_index_post_processor import LLamaIndexContextPostProcessorWrapper
from rago.model.wrapper.retriever.base import Retriever

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.llms import LLM
    from llama_index.core.retrievers import BaseRetriever

    from rago.model.configs.retriever_config.llama_index import LlamaIndexRetrieverConfig


class LlamaIndexRetrieverWrapper(Retriever):
    """A retriever using llama_index retrievers."""

    llama_index_retriever: BaseRetriever
    nodes_post_processors: Optional[LLamaIndexContextPostProcessorWrapper] = None

    def __init__(
        self,
        llama_index_retriever: BaseRetriever,
        nodes_post_processors: Optional[LLamaIndexContextPostProcessorWrapper] = None,
    ) -> None:
        """Instantiate a retriever with a llama_index retriever and optionally post-processors.

        :param llama_index_retriever: The llama_index retriever to use.
        :type llama_index_retriever: BaseRetriever
        :param nodes_post_processors: The context post-processors nodes to use, defaults to None
        :type nodes_post_processors: Optional[LLamaIndexContextPostProcessorWrapper], optional
        """
        self.llama_index_retriever = llama_index_retriever
        self.nodes_post_processors = nodes_post_processors

    @classmethod
    def make(
        cls,
        config: LlamaIndexRetrieverConfig,
        inputs_chunks: list[str],
        encoder: Optional[BaseEmbedding] = None,
        llm: Optional[LLM] = None,
    ) -> LlamaIndexRetrieverWrapper:
        """Generate a retriever from a documents set, an encoder, a llm and the config of the retriever to build.

        :param config: The config of the retriever to generate.
        :type config: dict
        :param inputs_chunks: the document that will be used to generate the non parametric knowledge.
        :type inputs_chunks: list[str]
        :param encoder: The encoder used by the retriever, defaults to None
        :type encoder: Optional[BaseEmbedding], optional
        :param llm: The llm used by the retriever, defaults to None
        :type llm: Optional[LLM], optional
        :return: The retriever corresponding to the config dict.
        :rtype: LlamaIndexRetrieverWrapper
        """
        langchain_inputs_chunks = [TextNode(text=chunk) for chunk in inputs_chunks]
        retriever = RetrieverFactory.make(config, langchain_inputs_chunks, encoder, llm)
        nodes_post_processors = (
            LLamaIndexContextPostProcessorWrapper.make(
                nodes_post_processors_configs=config.node_post_processor_config,
                encoder=encoder,
                llm=llm,
            )
            if config.node_post_processor_config is not None
            else None
        )
        return cls(retriever, nodes_post_processors)

    def get_retriever_output(self, query: str) -> list[RetrievedContext]:
        """Get the texts relevant to the query from the retriever database.

        :param query: The query.
        :type query: str
        :return: The texts relevant to the query from the retriever database.
        :rtype: list[str]
        """
        nodes_with_scores = self.llama_index_retriever.retrieve(query)
        retrieved_context = [
            RetrievedContext.from_llama_index_node_with_score(node_with_score) for node_with_score in nodes_with_scores
        ]

        if self.nodes_post_processors is not None:
            retrieved_context = self.nodes_post_processors.post_process_context(query, retrieved_context)
        return retrieved_context
