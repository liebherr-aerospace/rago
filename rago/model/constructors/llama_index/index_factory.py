"""Define a Factory of LlamaIndex Indices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from llama_index.core.indices import KnowledgeGraphIndex, TreeIndex, VectorStoreIndex

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.indices.base import BaseIndex
    from llama_index.core.llms import LLM
    from llama_index.core.schema import TextNode

    from rago.model.configs.index_config.llama_index import IndexConfig


class IndexFactory:
    """A LlamaIndex Index factory."""

    @staticmethod
    def make(
        config: IndexConfig,
        input_chunks: list[TextNode],
        encoder: Optional[BaseEmbedding] = None,
        llm: Optional[LLM] = None,
    ) -> BaseIndex:
        """Make an index from its config.

        :param config: The configuration of the index to build.
        :type config: IndexConfig
        :param input_chunks: The input chunks used to build the index store.
        :type input_chunks: list[TextNode]
        :param encoder: The encoder used by the index, defaults to None
        :type encoder: Optional[BaseEmbedding], optional
        :param llm: The llm used by the index, defaults to None
        :type llm: Optional[LLM], optional
        :raises ValueError: The index type is VectorIndex but the encoder is not set.
        :raises ValueError: The index type is TreeIndex but the llm is not set.
        :raises ValueError: The index type is not None.
        :return: the built index.
        :rtype: BaseIndex
        """
        match config.type:
            case "VectorStoreIndex":
                if encoder is None:
                    raise ValueError(encoder)
                return IndexFactory.make_vector_store(input_chunks, encoder=encoder)
            case "TreeIndex":
                if llm is None:
                    raise ValueError(llm)
                return IndexFactory.make_tree_index(config, input_chunks, llm=llm)
            case _:
                raise ValueError(config.type)

    @staticmethod
    def make_vector_store(inputs_chunks: list[TextNode], encoder: BaseEmbedding) -> VectorStoreIndex:
        """Build a vector store from inputs chunks and encoder.

        :param inputs_chunks: The inputs chunks used by the index to build.
        :type inputs_chunks: list[TextNode]
        :param encoder: The encoder used by the index to build.
        :type encoder: BaseEmbedding
        :return: The built vector index.
        :rtype: VectorStoreIndex
        """
        return VectorStoreIndex(
            nodes=inputs_chunks,
            embed_model=encoder,
        )

    @staticmethod
    def make_tree_index(config: IndexConfig, inputs_chunks: list[TextNode], llm: LLM) -> TreeIndex:
        """Build a tree index.

        :param config: The configuration of the index to build.
        :type config: IndexConfig
        :param inputs_chunks: Chunks used to build tree.
        :type inputs_chunks: list[TextNode]
        :param llm: LLM used to build the tree index.
        :type llm: LLM
        :return: The built tree index.
        :rtype: TreeIndex
        """
        if config.num_children is None:
            raise ValueError(config.num_children)
        return TreeIndex(
            nodes=inputs_chunks,
            llm=llm,
            num_children=config.num_children,
        )

    @staticmethod
    def make_knowledge_graph_index(
        inputs_chunks: list[TextNode],
        max_triplets_per_chunk: int,
        encoder: BaseEmbedding,
        llm: LLM,
    ) -> KnowledgeGraphIndex:
        """Build a knowledge graph.

        :param inputs_chunks: Chunks used to build knowledge graph.
        :type inputs_chunks: list[TextNode]
        :param max_triplets_per_chunk: max graph triplets to create from each chunks
        :type max_triplets_per_chunk: int
        :param encoder: The encoder used by the index to build.
        :type encoder: str
        :param llm: The LLM used by the index to build.
        :type llm: str
        :return: The built graph index.
        :rtype: KnowledgeGraphIndex
        """
        return KnowledgeGraphIndex(
            nodes=inputs_chunks,
            embed_model=encoder,
            llm=llm,
            max_triplets_per_chunk=max_triplets_per_chunk,
        )
