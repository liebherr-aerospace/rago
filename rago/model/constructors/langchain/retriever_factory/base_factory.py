"""Define a factory to make retriever with langchain."""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from typing import TYPE_CHECKING, ClassVar, Optional

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

logger = logging.getLogger(__name__)


def _corpus_hash(input_chunks: list[Document]) -> str:
    """Compute a lightweight hash of a corpus to use as a cache key component.

    :param input_chunks: The corpus documents.
    :type input_chunks: list[Document]
    :return: A hex digest uniquely identifying the corpus content.
    :rtype: str
    """
    h = hashlib.sha256()
    for doc in input_chunks:
        h.update(doc.page_content.encode("utf-8"))
    return h.hexdigest()[:16]


class RetrieverFactory:
    """A Langchain retriever factory to make retriever with langchain.

    The factory caches vector stores and BM25 indexes so that the same corpus
    is indexed only once per (encoder, similarity_function) or (k1, b, similarity)
    combination, even across many Optuna trials.
    """

    # Cache: (encoder_model_name, similarity_function, corpus_hash) → indexed Chroma VectorStore
    _vectorstore_cache: ClassVar[dict[tuple[str, Optional[str], str], VectorStore]] = {}

    # Cache: (k1, b, similarity, corpus_hash) → ready-to-query OpenSearchBM25Retriever
    _bm25_cache: ClassVar[dict[tuple[float, float, str, str], OpenSearchBM25Retriever]] = {}

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
    def _get_encoder_name(config: LangchainRetrieverConfig, encoder: Embeddings) -> str:
        """Extract a stable encoder name for cache keying.

        :param config: The retriever config (may contain encoder config with model_name).
        :type config: LangchainRetrieverConfig
        :param encoder: The encoder instance.
        :type encoder: Embeddings
        :return: A string identifying the encoder.
        :rtype: str
        """
        if config.encoder is not None and hasattr(config.encoder, "model_name"):
            return config.encoder.model_name
        if hasattr(encoder, "model_name"):
            return encoder.model_name
        # Fallback: use object id (no caching benefit, but safe)
        return str(id(encoder))

    @staticmethod
    def make_vector_store_retriever(
        config: LangchainRetrieverConfig,
        input_chunks: list[Document],
        encoder: Embeddings,
    ) -> VectorStoreRetriever:
        """Make a vector store retriever, reusing a cached Chroma store when possible.

        The Chroma vector store (with all documents already embedded and indexed)
        is cached by ``(encoder_model_name, similarity_function, corpus_hash)``.
        Only the lightweight retriever wrapper (with trial-specific ``search_type``
        and ``search_kwargs``) is created each time.

        :param config: The config of the vector store retriever to create.
        :type config: dict
        :param input_chunks: The chunks retrieved by the vector store retriever to create.
        :type input_chunks: list[Document]
        :param encoder: The encoder used by the retriever to convert queries and chunks to embeddings.
        :type encoder: Embeddings
        :return: The created vector store retriever.
        :rtype: VectorStoreRetriever
        """
        encoder_name = RetrieverFactory._get_encoder_name(config, encoder)
        c_hash = _corpus_hash(input_chunks)
        cache_key = (encoder_name, config.similarity_function, c_hash)

        if cache_key in RetrieverFactory._vectorstore_cache:
            logger.debug(
                "[CACHE HIT] Reusing Chroma vectorstore (encoder=%s, sim=%s)",
                encoder_name,
                config.similarity_function,
            )
            vectorstore = RetrieverFactory._vectorstore_cache[cache_key]
        else:
            logger.info(
                "[CACHE MISS] Building Chroma vectorstore (encoder=%s, sim=%s, %d chunks)",
                encoder_name,
                config.similarity_function,
                len(input_chunks),
            )
            vectorstore = RetrieverFactory.make_chroma(
                collection_name=str(uuid.uuid4()),
                encoder=encoder,
                similarity_function=config.similarity_function,
            )
            batch_size = 5000
            for i in range(0, len(input_chunks), batch_size):
                batch = input_chunks[i : i + batch_size]
                vectorstore.add_documents(batch)
            RetrieverFactory._vectorstore_cache[cache_key] = vectorstore

        # The retriever is cheap to create — only search_kwargs change between trials
        return vectorstore.as_retriever(search_type=config.search_type, search_kwargs=config.search_kwargs)

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
        """Create a BM25 retriever using OpenSearch, reusing a cached index when possible.

        The OpenSearch index is cached by ``(k1, b, similarity, corpus_hash)``
        so that re-indexing the same corpus with the same BM25 parameters is
        skipped on subsequent trials.

        :param config: The config of the vector store retriever to create.
        :type config: dict
        :param input_chunks: The chunks retrieved by the vector store retriever to create.
        :type input_chunks: list[Document]
        :return: The created BaseRetriever.
        :rtype: BaseRetriever
        """
        search_kwargs = config.search_kwargs or {}
        k1 = search_kwargs["k1"]
        b = search_kwargs["b"]
        similarity = search_kwargs["similarity"]
        c_hash = _corpus_hash(input_chunks)
        cache_key = (k1, b, similarity, c_hash)

        if cache_key in RetrieverFactory._bm25_cache:
            logger.debug("[CACHE HIT] Reusing BM25 index (k1=%.2f, b=%.2f)", k1, b)
            return RetrieverFactory._bm25_cache[cache_key]

        logger.info("[CACHE MISS] Building BM25 index (k1=%.2f, b=%.2f, %d chunks)", k1, b, len(input_chunks))
        retriever = OpenSearchBM25Retriever.create(
            opensearch_url=OPENSEARCH_URL,
            index_name=OPENSEARCH_INDEX_NAME,
            k1=k1,
            b=b,
            similarity=similarity,
        )
        retriever.add_texts(input_chunks)
        RetrieverFactory._bm25_cache[cache_key] = retriever
        return retriever

    @staticmethod
    def clear_cache() -> None:
        """Clear all cached vector stores and BM25 indexes."""
        RetrieverFactory._vectorstore_cache.clear()
        RetrieverFactory._bm25_cache.clear()
        logger.info("[CACHE] Langchain retriever factory cache cleared")
