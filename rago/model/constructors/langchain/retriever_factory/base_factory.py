"""Define a factory to make retriever with langchain."""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional

import chromadb
from langchain_chroma import Chroma

from rago.model.constructors.langchain.encoder_factory import EncoderFactory
from rago.model.constructors.langchain.retriever_factory.bm25_factory import OpenSearchBM25Retriever
from rago.utils import PATH_PROJECT

if TYPE_CHECKING:
    from langchain.embeddings.base import Embeddings
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.vectorstores import VectorStoreRetriever

    from rago.model.configs.retriever_config.langchain import LangchainRetrieverConfig

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "")
OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "")

CHROMA_CACHE_DIR = Path(
    os.getenv("RAGO_CHROMA_CACHE_DIR", str(Path(PATH_PROJECT) / ".cache" / "rago" / "chroma")),
)

logger = logging.getLogger(__name__)


def _safe_dir_name(*parts: Optional[str]) -> str:
    """Build a filesystem-safe directory name from arbitrary key parts.

    :param parts: Strings (or None) to combine into a directory name.
    :return: A sanitised string usable as a directory name.
    :rtype: str
    """
    joined = "_".join(str(p) if p is not None else "none" for p in parts)
    return joined.replace("/", "--").replace("\\", "--").replace(":", "-")


class RetrieverFactory:
    """A Langchain retriever factory to make retriever with langchain.

    The factory caches vector stores and BM25 indexes so that the same corpus
    is indexed only once per (encoder, similarity_function) or (k1, b, similarity)
    combination, even across many Optuna trials.
    """

    _corpus_id_cache: ClassVar[dict[int, str]] = {}

    _chroma_client: ClassVar[Optional[chromadb.ClientAPI]] = None

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
        return str(id(encoder))

    @staticmethod
    def _get_corpus_id(input_chunks: list[Document]) -> str:
        """Return a stable UUID for a given corpus list object (cheap, no hashing).

        Within the same process the same *list object* always receives the same
        UUID.  A new UUID is generated the first time a list is seen.

        :param input_chunks: The corpus documents.
        :type input_chunks: list[Document]
        :return: A hex string identifying the corpus.
        :rtype: str
        """
        obj_id = id(input_chunks)
        if obj_id not in RetrieverFactory._corpus_id_cache:
            RetrieverFactory._corpus_id_cache[obj_id] = uuid.uuid4().hex[:16]
        return RetrieverFactory._corpus_id_cache[obj_id]

    @staticmethod
    def _get_chroma_client() -> chromadb.ClientAPI:
        """Return (and lazily create) the shared persistent Chroma client.

        A single directory (``CHROMA_CACHE_DIR``) is used for all collections.

        :return: A persistent Chroma client.
        :rtype: chromadb.ClientAPI
        """
        if RetrieverFactory._chroma_client is None:
            CHROMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            RetrieverFactory._chroma_client = chromadb.PersistentClient(path=str(CHROMA_CACHE_DIR))
        return RetrieverFactory._chroma_client

    @staticmethod
    def make_vector_store_retriever(
        config: LangchainRetrieverConfig,
        input_chunks: list[Document],
        encoder: Embeddings,
    ) -> VectorStoreRetriever:
        """Make a vector store retriever, reusing a cached Chroma collection when possible.

        Each unique ``(encoder, similarity_function, corpus)`` combination is
        stored as a *collection* inside a single shared Chroma persistent
        directory (``CHROMA_CACHE_DIR``).  If the collection already contains
        documents it is reused directly; otherwise the corpus is embedded and
        indexed.

        Only the lightweight retriever wrapper (with trial-specific
        ``search_type`` and ``search_kwargs``) is created each time.

        :param config: The config of the vector store retriever to create.
        :type config: LangchainRetrieverConfig
        :param input_chunks: The chunks retrieved by the vector store retriever to create.
        :type input_chunks: list[Document]
        :param encoder: The encoder used by the retriever to convert queries and chunks to embeddings.
        :type encoder: Embeddings
        :return: The created vector store retriever.
        :rtype: VectorStoreRetriever
        """
        encoder_name = RetrieverFactory._get_encoder_name(config, encoder)
        corpus_id = RetrieverFactory._get_corpus_id(input_chunks)
        collection_name = _safe_dir_name(encoder_name, config.similarity_function, corpus_id)

        client = RetrieverFactory._get_chroma_client()

        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": config.similarity_function} if config.similarity_function else None,
        )
        needs_indexing = collection.count() == 0

        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=encoder,
            collection_metadata={"hnsw:space": config.similarity_function} if config.similarity_function else None,
        )

        if needs_indexing:
            logger.info(
                "[CACHE MISS] Building Chroma collection %s (encoder=%s, sim=%s, %d chunks)",
                collection_name,
                encoder_name,
                config.similarity_function,
                len(input_chunks),
            )
            batch_size = 5000
            for i in range(0, len(input_chunks), batch_size):
                batch = input_chunks[i : i + batch_size]
                vectorstore.add_documents(batch)
        else:
            logger.debug(
                "[CACHE HIT] Reusing Chroma collection %s (encoder=%s, sim=%s)",
                collection_name,
                encoder_name,
                config.similarity_function,
            )

        return vectorstore.as_retriever(search_type=config.search_type, search_kwargs=config.search_kwargs)

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
        corpus_id = RetrieverFactory._get_corpus_id(input_chunks)
        cache_key = (k1, b, similarity, corpus_id)

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
    def clear_cache(*, include_disk: bool = False) -> None:
        """Clear all cached vector stores and BM25 indexes.

        :param include_disk: If ``True``, also delete the persistent Chroma
            directories under ``CHROMA_CACHE_DIR``.  Defaults to ``False``
            (memory-only clear).
        :type include_disk: bool
        """
        RetrieverFactory._corpus_id_cache.clear()
        RetrieverFactory._bm25_cache.clear()

        if include_disk and CHROMA_CACHE_DIR.exists():
            RetrieverFactory._chroma_client = None
            shutil.rmtree(CHROMA_CACHE_DIR)
            logger.info("[CACHE] Deleted persistent Chroma cache at %s", CHROMA_CACHE_DIR)

        logger.info("[CACHE] Langchain retriever factory cache cleared")
