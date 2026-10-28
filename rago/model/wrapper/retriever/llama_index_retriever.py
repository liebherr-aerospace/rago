"""Define a retriever using llama_index retrievers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from llama_index.core.schema import TextNode

from rago.data_objects import ModelInput, RetrievalOutput, RetrievedContext
from rago.model.configs.retriever_config.llama_index import LlamaIndexRetrieverConfig
from rago.model.constructors.llama_index.retriever_factory import RetrieverFactory
from rago.model.wrapper.context_post_processor.llama_index_post_processor import LLamaIndexContextPostProcessorWrapper
from rago.model.wrapper.retriever.base import Retriever

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.llms import LLM
    from llama_index.core.retrievers import BaseRetriever

    from rago.model.wrapper.tunable_model import BuildContext


class LlamaIndexRetrieverWrapper(Retriever[LlamaIndexRetrieverConfig]):
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
        ctx: Optional[BuildContext] = None,
        *,
        encoder: Optional[BaseEmbedding] = None,
        llm: Optional[LLM] = None,
    ) -> LlamaIndexRetrieverWrapper:
        """Generate a retriever from the document chunks of the context, an encoder, a llm and the retriever config.

        :param config: The config of the retriever to generate.
        :type config: LlamaIndexRetrieverConfig
        :param ctx: Runtime resources, ``ctx.inputs_chunks`` is required.
        :type ctx: Optional[BuildContext]
        :param encoder: The encoder used by the retriever, defaults to None
        :type encoder: Optional[BaseEmbedding], optional
        :param llm: The llm used by the retriever, defaults to None
        :type llm: Optional[LLM], optional
        :raises ValueError: If no input chunks are provided.
        :return: The retriever corresponding to the config.
        :rtype: LlamaIndexRetrieverWrapper
        """
        if ctx is None or ctx.inputs_chunks is None:
            error_msg = f"{cls.__name__} requires ctx.inputs_chunks."
            raise ValueError(error_msg)
        langchain_inputs_chunks = [TextNode(text=chunk) for chunk in ctx.inputs_chunks]
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

    def _get_output(self, model_input: ModelInput) -> RetrievalOutput:
        """Get the texts relevant to the query from the retriever database.

        :param query: The query.
        :type query: str
        :return: The texts relevant to the query from the retriever database.
        :rtype: list[str]
        """
        nodes_with_scores = self.llama_index_retriever.retrieve(model_input.query)
        retrieved_context = [
            RetrievedContext.from_llama_index_node_with_score(node_with_score) for node_with_score in nodes_with_scores
        ]

        if self.nodes_post_processors is not None:
            retrieved_context = self.nodes_post_processors.post_process_context(model_input.query, retrieved_context)
        return RetrievalOutput(retrieved_context)
