"""Define a factory to make retriever with langchain."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
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
    from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

    from rago.model.configs.retriever_config.langchain import LangchainRetrieverConfig

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "")
OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "")

#: Base directory for the persistent Chroma cache.
#: Override via the ``RAGO_CHROMA_CACHE_DIR`` environment variable.
CHROMA_CACHE_DIR = Path(
    os.getenv("RAGO_CHROMA_CACHE_DIR", str(Path(PATH_PROJECT) / ".cache" / "rago" / "chroma")),
)

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


def _safe_dir_name(*parts: Optional[str]) -> str:
    """Build a filesystem-safe directory name from arbitrary key parts.

    :param parts: Strings (or None) to combine into a directory name.
    :return: A sanitised string usable as a directory name.
    :rtype: str
    """
    joined = "_".join(str(p) if p is not None else "none" for p in parts)
    # Replace any character that could be problematic in a path
    return joined.replace("/", "--").replace("\\", "--").replace(":", "-")


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

        Caching strategy (two levels):

        1. **In-memory** - the ``_vectorstore_cache`` dict keeps a reference to
           the ``Chroma`` object so that subsequent trials in the *same process*
           pay zero cost.
        2. **On-disk** - each unique ``(encoder, similarity, corpus)`` combination
           is persisted under ``CHROMA_CACHE_DIR`` via
           ``chromadb.PersistentClient``.  When a *new process* starts, the
           factory detects the existing collection and skips re-embedding /
           re-indexing entirely.

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
        c_hash = _corpus_hash(input_chunks)
        cache_key = (encoder_name, config.similarity_function, c_hash)

        # --- Level 1: in-memory hit (same process) ---
        if cache_key in RetrieverFactory._vectorstore_cache:
            logger.debug(
                "[CACHE HIT · memory] Reusing Chroma vectorstore (encoder=%s, sim=%s)",
                encoder_name,
                config.similarity_function,
            )
            vectorstore = RetrieverFactory._vectorstore_cache[cache_key]
        else:
            # Build the persistent path: .cache/rago/chroma/<encoder>_<sim>_<hash>/
            persist_dir = CHROMA_CACHE_DIR / _safe_dir_name(encoder_name, config.similarity_function, c_hash)
            collection_name = "default"

            # --- Level 2: on-disk hit (new process, same corpus) ---
            if persist_dir.exists():
                logger.info(
                    "[CACHE HIT · disk] Loading Chroma from %s (encoder=%s, sim=%s)",
                    persist_dir,
                    encoder_name,
                    config.similarity_function,
                )
                vectorstore = RetrieverFactory._make_persistent_chroma(
                    persist_dir=str(persist_dir),
                    collection_name=collection_name,
                    encoder=encoder,
                    similarity_function=config.similarity_function,
                )
            else:
                # --- Full miss: embed + index + persist ---
                logger.info(
                    "[CACHE MISS] Building & persisting Chroma vectorstore (encoder=%s, sim=%s, %d chunks) → %s",
                    encoder_name,
                    config.similarity_function,
                    len(input_chunks),
                    persist_dir,
                )
                persist_dir.mkdir(parents=True, exist_ok=True)
                vectorstore = RetrieverFactory._make_persistent_chroma(
                    persist_dir=str(persist_dir),
                    collection_name=collection_name,
                    encoder=encoder,
                    similarity_function=config.similarity_function,
                )
                batch_size = 5000
                for i in range(0, len(input_chunks), batch_size):
                    batch = input_chunks[i : i + batch_size]
                    vectorstore.add_documents(batch)

            # Store in memory for subsequent trials in this process
            RetrieverFactory._vectorstore_cache[cache_key] = vectorstore

        # The retriever is cheap to create — only search_kwargs change between trials
        return vectorstore.as_retriever(search_type=config.search_type, search_kwargs=config.search_kwargs)

    @staticmethod
    def _make_persistent_chroma(
        persist_dir: str,
        collection_name: str,
        similarity_function: Optional[str],
        encoder: Embeddings,
    ) -> VectorStore:
        """Create a Chroma index backed by a persistent on-disk store.

        :param persist_dir: Path to the directory where Chroma persists data.
        :type persist_dir: str
        :param collection_name: The name of the collection in the Chroma index.
        :type collection_name: str
        :param similarity_function: HNSW space used by Chroma (e.g. ``"cosine"``).
        :type similarity_function: Optional[str]
        :param encoder: Embedding function for documents and queries.
        :type encoder: Embeddings
        :return: The created (or reopened) Chroma vector store.
        :rtype: VectorStore
        """
        client = chromadb.PersistentClient(path=persist_dir)
        return Chroma(
            client=client,
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
    def clear_cache(*, include_disk: bool = False) -> None:
        """Clear all cached vector stores and BM25 indexes.

        :param include_disk: If ``True``, also delete the persistent Chroma
            directories under ``CHROMA_CACHE_DIR``.  Defaults to ``False``
            (memory-only clear).
        :type include_disk: bool
        """
        RetrieverFactory._vectorstore_cache.clear()
        RetrieverFactory._bm25_cache.clear()

        if include_disk and CHROMA_CACHE_DIR.exists():
            shutil.rmtree(CHROMA_CACHE_DIR)
            logger.info("[CACHE] Deleted persistent Chroma cache at %s", CHROMA_CACHE_DIR)

        logger.info("[CACHE] Langchain retriever factory cache cleared")
