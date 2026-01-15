"""Define a factory to make retriever with langchain."""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Optional

import chromadb
from langchain_chroma import Chroma

from rago.model.constructors.langchain.encoder_factory import EncoderFactory
from rago.model.constructors.langchain.retriever_factory.bm25_factory import OpenSearchBM25Retriever

if TYPE_CHECKING:
    from langchain.embeddings.base import Embeddings
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

    from rago.model.configs.retriever_config.langchain import LangchainRetrieverConfig

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "")
OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "")


class RetrieverFactory:
    """A Langchain retriever factory to make retriever with langchain."""

    @staticmethod
    def make(
        config: LangchainRetrieverConfig,
        input_chunks: list[Document],
        encoder: Optional[Embeddings] = None,
    ) -> BaseRetriever:
        """Make a langchain retriever from its config.

        :param config: The config of the retriever to create.
        :type config: dict
        :param input_chunks: The inputs chunks the retriever is to retrieve.
        :type input_chunks: list[Document]
        :param encoder: The encoder used by the retriever to create, defaults to None.
        :type encoder: Optional[Embeddings], optional
        :raises ValueError: The encoder is Not but the config does not contains the encoder config to create one.
        :raises ValueError: The retriever type in the config is unknown.
        :return: The created retriever.
        :rtype: BaseRetriever
        """
        match config.type:
            case "VectorIndexRetriever":
                encoder = RetrieverFactory.get_encoder(config, encoder)
                return RetrieverFactory.make_vector_store_retriever(config, input_chunks=input_chunks, encoder=encoder)
            case "BM25Retriever":
                return RetrieverFactory.make_bm25_retriever(config, input_chunks=input_chunks)
            case _:
                raise ValueError(config.type)

    @staticmethod
    def get_encoder(config: LangchainRetrieverConfig, encoder: Optional[Embeddings]) -> Embeddings:
        """Check and get encoder.

        :param config: The config of the retriever to create.
        :type config: dict
        :param encoder: The encoder used by the retriever to create, defaults to None.
        :type encoder: Optional[Embeddings], optional
        :return the encoder.
        :rtype: LangchainEncoderConfig
        :raise a value error is the encoder is None in the configuration?
        """
        if encoder is not None:
            return encoder
        if config.encoder is None:
            error_msg = f"No encoder provided in the config {config}"
            raise ValueError(error_msg)
        return EncoderFactory.make(config.encoder)

    @staticmethod
    def make_vector_store_retriever(
        config: LangchainRetrieverConfig,
        input_chunks: list[Document],
        encoder: Embeddings,
    ) -> VectorStoreRetriever:
        """Make a vector store retriever.

        :param config: The config of the vector store retriever to create.
        :type config: dict
        :param input_chunks: The chunks retrieved by the vector store retriever to create.
        :type input_chunks: list[Document]
        :param encoder: The encoder used by the retriever to convert queries and chunks to embeddings.
        :type encoder: Embeddings
        :return: The created vector store retriever.
        :rtype: VectorStoreRetriever
        """
        vectorstore_from_client = RetrieverFactory.make_chroma(
            collection_name=str(uuid.uuid4()),
            encoder=encoder,
            similarity_function=config.similarity_function,
        )
        batch_size = 5000
        for i in range(0, len(input_chunks), batch_size):
            batch = input_chunks[i:i + batch_size]
            vectorstore_from_client.add_documents(batch)
        return vectorstore_from_client.as_retriever(search_type=config.search_type, search_kwargs=config.search_kwargs)

    @staticmethod
    def make_chroma(
        collection_name: str,
        similarity_function: Optional[str],
        encoder: Embeddings,
    ) -> VectorStore:
        """Create a chroma index.

        :param collection_name: The name of the collection in the chroma index.
        :type collection_name: str
        :param similarity_function: Used by the chroma Index to determine similarity between a query and its elements.
        :type similarity_function: str
        :param encoder: Used by the chroma to encode input chunks and query into embeddings, defaults to None.s
        :type encoder: Embeddings, optional
        :return: The create chroma index.
        :rtype: VectorStore
        """
        ephemeral_client = chromadb.EphemeralClient()
        return Chroma(
            client=ephemeral_client,
            collection_name=collection_name,
            embedding_function=encoder,
            collection_metadata={"hnsw:space": similarity_function},
        )

    @staticmethod
    def make_bm25_retriever(
        config: LangchainRetrieverConfig,
        input_chunks: list[Document],
    ) -> BaseRetriever:
        """Create a BM25 retriever using OpenSearch.

        :param config: The config of the vector store retriever to create.
        :type config: dict
        :param input_chunks: The chunks retrieved by the vector store retriever to create.
        :type input_chunks: list[Document]
        :return: The created BaseRetriever.
        :rtype: BaseRetriever
        """
        search_kwargs = config.search_kwargs or {}
        retriever = OpenSearchBM25Retriever.create(
            opensearch_url=OPENSEARCH_URL,
            index_name=OPENSEARCH_INDEX_NAME,
            k1=search_kwargs["k1"],
            b=search_kwargs["b"],
            similarity=search_kwargs["similarity"],
        )
        retriever.add_texts(input_chunks)
        return retriever
