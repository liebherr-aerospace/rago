"""Define a retriever builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from llama_index.core.prompts import PromptTemplate, PromptType
from llama_index.core.retrievers import (
    BaseRetriever,
    RouterRetriever,
    TreeAllLeafRetriever,
    TreeSelectLeafEmbeddingRetriever,
    TreeSelectLeafRetriever,
    VectorIndexRetriever,
)
from llama_index.core.selectors import BaseSelector, EmbeddingSingleSelector, LLMMultiSelector
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from rago.model.constructors.llama_index import IndexFactory
from rago.model.constructors.llama_index.encoder_factory import EncoderFactory
from rago.model.constructors.llama_index.llm_factory import LLamaIndexLLMFactory

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.llms import LLM
    from llama_index.core.schema import TextNode

    from rago.model.configs.retriever_config.llama_index import LlamaIndexRetrieverConfig, SelectorConfig


class RetrieverFactory:
    """A retriever builder."""

    @staticmethod
    def make(
        config: LlamaIndexRetrieverConfig,
        input_chunks: list[TextNode],
        encoder: Optional[BaseEmbedding] = None,
        llm: Optional[LLM] = None,
    ) -> BaseRetriever:
        """Build a retriever.

        :param input_chunks: _description_
        :type input_chunks: list[TextNode]
        :param config: _description_
        :type config: dict
        :param encoder: _description_, defaults to None
        :type encoder: Optional[BaseEmbedding], optional
        :param llm: _description_, defaults to None
        :type llm: Optional[LLM], optional
        :return: _description_
        :rtype: BaseRetriever
        """
        match config.type:
            case "VectorIndexRetriever":
                if encoder is None:
                    if config.encoder is None:
                        raise ValueError(config.encoder)
                    encoder = EncoderFactory.make(config.encoder)
                if config.similarity_top_k is None:
                    raise ValueError(config.similarity_top_k)
                if config.vector_store_query_mode is None:
                    raise ValueError(config.vector_store_query_mode)
                return VectorIndexRetriever(
                    index=IndexFactory.make_vector_store(
                        inputs_chunks=input_chunks,
                        encoder=encoder,
                    ),
                    similarity_top_k=config.similarity_top_k,
                    vector_store_query_mode=VectorStoreQueryMode(config.vector_store_query_mode),
                )
            case "TreeAllLeafRetriever":
                return RetrieverFactory.make_all_leaf_retriever(
                    config=config,
                    input_chunks=input_chunks,
                    llm=llm,
                )
            case "TreeSelectLeafEmbeddingRetriever":
                return RetrieverFactory.make_tree_select_leaf_embedding_retriever(
                    config=config,
                    input_chunks=input_chunks,
                    encoder=encoder,
                    llm=llm,
                )
            case "TreeSelectLeafRetriever":
                return RetrieverFactory.make_tree_select_leaf_retriever(
                    config=config,
                    input_chunks=input_chunks,
                    llm=llm,
                )
            case "RouterRetriever":
                return RetrieverFactory.make_router_retriever(
                    config=config,
                    input_chunks=input_chunks,
                    encoder=encoder,
                    llm=llm,
                )
            case _:
                raise ValueError(config.type)

    @staticmethod
    def make_all_leaf_retriever(
        config: LlamaIndexRetrieverConfig,
        input_chunks: list[TextNode],
        llm: Optional[LLM] = None,
    ) -> TreeAllLeafRetriever:
        """Make a tree all leaf retriever.

        :param config: The config of th retriever to create.
        :type config: LangchainRetrieverConfig
        :param input_chunks: The chunks used by the retriever.
        :type input_chunks: list[TextNode]
        :raises ValueError: If the llm config is None
        :return: The created retriever.
        :rtype: TreeAllLeafRetriever
        """
        if llm is None:
            if config.llm is None:
                raise ValueError(config.llm)
            llm = LLamaIndexLLMFactory.make(config.llm)
        return TreeAllLeafRetriever(
            index=IndexFactory.make_tree_index(
                inputs_chunks=input_chunks,
                config=config.index,
                llm=llm,
            ),
        )

    @staticmethod
    def make_tree_select_leaf_embedding_retriever(
        config: LlamaIndexRetrieverConfig,
        input_chunks: list[TextNode],
        encoder: Optional[BaseEmbedding] = None,
        llm: Optional[LLM] = None,
    ) -> TreeSelectLeafEmbeddingRetriever:
        """Make a tree select leaf embedding retriever.

        :param config: The config of th retriever to create.
        :type config: LangchainRetrieverConfig
        :param input_chunks: The chunks used by the retriever.
        :type input_chunks: list[TextNode]
        :param encoder: The encoder used by the retriever
        :type encoder: BaseEmbedding
        :raises ValueError: If the llm config is None
        :return: The created retriever.
        :rtype: TreeAllLeafRetriever
        """
        if encoder is None:
            if config.encoder is None:
                raise ValueError(config.encoder)
            encoder = EncoderFactory.make(config.encoder)
        if llm is None:
            if config.llm is None:
                raise ValueError(config.llm)
            llm = LLamaIndexLLMFactory.make(config.llm)
        return TreeSelectLeafEmbeddingRetriever(
            index=IndexFactory.make_tree_index(config=config.index, inputs_chunks=input_chunks, llm=llm),
            embed_model=encoder,
        )

    @staticmethod
    def make_router_retriever(
        config: LlamaIndexRetrieverConfig,
        input_chunks: list[TextNode],
        encoder: Optional[BaseEmbedding] = None,
        llm: Optional[LLM] = None,
    ) -> RouterRetriever:
        """Make a tree select leaf embedding retriever.

        :param config: The config of th retriever to create.
        :type config: LangchainRetrieverConfig
        :param input_chunks: The chunks used by the retriever.
        :type input_chunks: list[TextNode]
        :param encoder: The encoder used by the retriever
        :type encoder: BaseEmbedding
        :raises ValueError: If the llm config is None
        :return: The created retriever.
        :rtype: RouterRetriever
        """
        if encoder is None:
            if config.encoder is None:
                raise ValueError(config.encoder)
            encoder = EncoderFactory.make(config.encoder)
        if llm is None:
            if config.llm is None:
                raise ValueError(config.llm)
            llm = LLamaIndexLLMFactory.make(config.llm)
        if config.selector is None:
            raise ValueError(config.selector)
        if config.retriever_tools is None:
            raise ValueError(config.retriever_tools)
        selector = RetrieverFactory.make_selector(
            config.selector,
            encoder=encoder,
            llm=llm,
        )
        retriever_tools = [
            RetrieverFactory.make_retriever_tool(
                retriever_config=retriever_tool,
                inputs_chunks=input_chunks,
                encoder=encoder,
                llm=llm,
            )
            for retriever_tool in config.retriever_tools
        ]
        return RouterRetriever(
            selector=selector,
            retriever_tools=retriever_tools,
        )

    @staticmethod
    def make_tree_select_leaf_retriever(
        config: LlamaIndexRetrieverConfig,
        input_chunks: list[TextNode],
        llm: Optional[LLM] = None,
    ) -> TreeSelectLeafRetriever:
        """Make a tree select leaf retriever.

        :param config: The config of th retriever to create.
        :type config: LangchainRetrieverConfig
        :param input_chunks: The chunks used by the retriever.
        :type input_chunks: list[TextNode]
        :param encoder: The encoder used by the retriever
        :type encoder: BaseEmbedding
        :raises ValueError: If the llm config is None
        :return: The created retriever.
        :rtype: TreeSelectLeafRetriever
        """
        if llm is None:
            if config.llm is None:
                raise ValueError(config.llm)
            llm = LLamaIndexLLMFactory.make(config.llm)
        if config.child_branch_factor is None:
            raise ValueError(config.child_branch_factor)
        index = IndexFactory.make_tree_index(config.index, input_chunks, llm=llm)
        return TreeSelectLeafRetriever(
            index=index,
            child_branch_factor=config.child_branch_factor,
        )

    @staticmethod
    def make_selector(selector_config: SelectorConfig, encoder: BaseEmbedding, llm: LLM) -> BaseSelector:
        """Build a selector.

        :param selector_config: Configuration parameters of the selector to use.
        :type selector_config: SelectorConfig
        :param encoder: Encoder used by the built selector.
        :type encoder: BaseEmbedding
        :param llm: LLM used by the built selector.
        :type llm: LLM
        :return: The built selector.
        :rtype: BaseSelector
        """
        match selector_config.selector_type:
            case "LLMMultiSelector":
                prompt_str = (
                    "Some choices are given below. It is provided in a numbered list "
                    "(1 to {num_choices}), "
                    "where each item in the list corresponds to a summary.\n"
                    "---------------------\n"
                    "{context_list}"
                    "\n---------------------\n"
                    "Using only the choices above and not prior knowledge, return "
                    "the choice that is most relevant to the question: '{query_str}'\n"
                )
                prompt = PromptTemplate(
                    template=prompt_str,
                    prompt_type=PromptType.MULTI_SELECT,
                )
                return LLMMultiSelector(
                    llm=llm,
                    prompt=prompt,
                    max_outputs=selector_config.max_outputs,
                )
            case "EmbeddingSingleSelector":
                return EmbeddingSingleSelector(
                    embed_model=encoder,
                )
            case _:
                raise ValueError(selector_config.selector_type)

    @staticmethod
    def make_retriever_tool(
        retriever_config: LlamaIndexRetrieverConfig,
        inputs_chunks: list[TextNode],
        encoder: BaseEmbedding,
        llm: LLM,
    ) -> RetrieverTool:
        """Build a retriever tool.

        :param config: The config of th retriever to create.
        :type config: LangchainRetrieverConfig
        :param input_chunks: The chunks used by the retriever.
        :type input_chunks: list[TextNode]
        :param encoder: The encoder used by the retriever
        :type encoder: BaseEmbedding
        :raises ValueError: If the llm config is None
        :return: The built retriever tool.
        :rtype: RetrieverTool
        """
        retriever = RetrieverFactory.make(retriever_config, inputs_chunks, encoder, llm)
        return RetrieverTool(retriever=retriever, metadata=ToolMetadata(""))
