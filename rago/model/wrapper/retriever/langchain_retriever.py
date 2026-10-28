"""Define a retriever using langchain retrievers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from langchain_core.documents import Document

from rago.data_objects import ModelInput, RetrievalOutput, RetrievedContext
from rago.model.configs.retriever_config.langchain import LangchainRetrieverConfig
from rago.model.constructors.langchain.retriever_factory.base_factory import RetrieverFactory
from rago.model.wrapper.context_post_processor.llama_index_post_processor import LLamaIndexContextPostProcessorWrapper
from rago.model.wrapper.retriever.base import Retriever

if TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever

    from rago.model.wrapper.tunable_model import BuildContext


class LangchainRetrieverWrapper(Retriever[LangchainRetrieverConfig]):
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
        ctx: Optional[BuildContext] = None,
    ) -> LangchainRetrieverWrapper:
        """Generate a retriever from the document chunks of the context and the config of the retriever to build.

        :param config: The config of the retriever to generate.
        :type config: LangchainRetrieverConfig
        :param ctx: Runtime resources, ``ctx.inputs_chunks`` is required.
        :type ctx: Optional[BuildContext]
        :raises ValueError: If no input chunks are provided.
        :return: The retriever corresponding to the config.
        :rtype: LangchainRetrieverWrapper
        """
        if ctx is None or ctx.inputs_chunks is None:
            error_msg = f"{cls.__name__} requires ctx.inputs_chunks."
            raise ValueError(error_msg)
        langchain_inputs_chunks = [Document(chunk) for chunk in ctx.inputs_chunks]

        retriever = RetrieverFactory.make(config, langchain_inputs_chunks)
        if config.node_post_processor_config is not None:
            nodes_post_processors = LLamaIndexContextPostProcessorWrapper.make(
                config.node_post_processor_config,
            )
        else:
            nodes_post_processors = None

        return cls(retriever, nodes_post_processors)

    def _get_output(self, model_input: ModelInput) -> RetrievalOutput:
        """Get the texts relevant to the query from the retriever database.

        :param query: The query.
        :type query: str
        :return: The texts relevant to the query from the retriever database.
        :rtype: list[str]
        """
        nodes_with_scores = self.langchain_retriever.invoke(model_input.query)
        retrieved_context = [
            RetrievedContext.from_langchain_node_with_score(node_with_score) for node_with_score in nodes_with_scores
        ]
        if self.nodes_post_processors is not None:
            retrieved_context = self.nodes_post_processors.post_process_context(model_input.query, retrieved_context)
        return RetrievalOutput(retrieved_context)
