"""Define a retriever wrapper for Qdrant hybrid search.

This wrapper executes a 2-stage or 3-stage hybrid search against a single
Qdrant collection using the ``qdrant_client`` SDK:

* **Stage 1**: Two parallel *prefetch* sub-queries (dense + sparse/BM25).
* **Stage 2**: Fusion of the prefetch results via **RRF** (weighted) or
  **DBSF**.
* **Stage 3** *(optional)*: ColBERT late-interaction re-scoring.

The wrapper is driven entirely by a :class:`QdrantRetrieverConfig`.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from pylate.models import ColBERT
from pylate.rank import rerank as pylate_rerank
from qdrant_client import QdrantClient as _QdrantClient
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Prefetch,
    Rrf,
    RrfQuery,
    SparseVector,
)
from sentence_transformers import CrossEncoder

from rago.data_objects import RetrievedContext
from rago.model.wrapper.retriever.base import Retriever

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

    from rago.model.configs.retriever_config.qdrant import QdrantRetrieverConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding provider helpers
# ---------------------------------------------------------------------------

# Module-level caches so that embedders are instantiated only once per process
# even when Optuna calls make() hundreds of times sequentially.
# Keys: (provider, model, base_url) for dense; () singleton for sparse/late.
_DENSE_EMBEDDER_CACHE: dict[tuple[str, str, str], Any] = {}
_SPARSE_EMBEDDER_CACHE: dict[str, Any] = {}
_LATE_INTERACTION_EMBEDDER_CACHE: dict[str, Any] = {}


def _build_dense_embedder(config: QdrantRetrieverConfig) -> _LangchainEmbeddingAdapter:
    """Return a cached dense embedding provider built from the collection metadata.

    Supports:
    * ``"ollama"`` → uses ``langchain_ollama.OllamaEmbeddings``
    * ``"huggingface"`` → uses ``langchain_huggingface.HuggingFaceEmbeddings``

    Returns an object with an ``embed([text]) → list[list[float]]``-compatible
    interface (we wrap it with a thin adapter).
    """
    coll = config.collection
    provider = coll.encoder_provider.lower()
    cache_key = (provider, coll.encoder_model, coll.encoder_base_url)

    if cache_key not in _DENSE_EMBEDDER_CACHE:
        logger.info("Building dense embedder (%s / %s) — will be cached", provider, coll.encoder_model)
        if provider == "ollama":
            base_url = coll.encoder_base_url or os.environ.get("QDRANT_URL", "")
            embedder = OllamaEmbeddings(model=coll.encoder_model, base_url=base_url)
            _DENSE_EMBEDDER_CACHE[cache_key] = _LangchainEmbeddingAdapter(embedder)

        elif provider == "huggingface":
            embedder = HuggingFaceEmbeddings(model_name=coll.encoder_model)
            _DENSE_EMBEDDER_CACHE[cache_key] = _LangchainEmbeddingAdapter(embedder)

        else:
            msg = f"Unsupported dense encoder provider: '{provider}'"
            raise ValueError(msg)

    return _DENSE_EMBEDDER_CACHE[cache_key]


def _build_sparse_embedder() -> _FastEmbedSparseAdapter:
    """Return a cached FastEmbed BM25 sparse embedding provider."""
    model_name = "Qdrant/bm25"
    if model_name not in _SPARSE_EMBEDDER_CACHE:
        logger.info("Building sparse embedder (%s) — will be cached", model_name)

        _SPARSE_EMBEDDER_CACHE[model_name] = _FastEmbedSparseAdapter(
            SparseTextEmbedding(model_name=model_name),
        )
    return _SPARSE_EMBEDDER_CACHE[model_name]


def _build_late_interaction_embedder() -> _FastEmbedLateInteractionAdapter:
    """Return a cached FastEmbed ColBERT late-interaction embedding provider."""
    model_name = "colbert-ir/colbertv2.0"
    if model_name not in _LATE_INTERACTION_EMBEDDER_CACHE:
        logger.info("Building late-interaction embedder (%s) — will be cached", model_name)

        _LATE_INTERACTION_EMBEDDER_CACHE[model_name] = _FastEmbedLateInteractionAdapter(
            LateInteractionTextEmbedding(model_name=model_name),
        )
    return _LATE_INTERACTION_EMBEDDER_CACHE[model_name]


# ---------------------------------------------------------------------------
# Cross-encoder reranker helper
# ---------------------------------------------------------------------------

# Cache reranker models so they are loaded only once per process even
# when Optuna spawns many trials sequentially.
# Supports two backends, distinguished by the model name prefix:
#   - "colbert:<model>"  → pylate ColBERT late-interaction reranker
#   - "<model>"          → sentence-transformers CrossEncoder (default)
_CROSS_ENCODER_CACHE: dict[str, Any] = {}

_COLBERT_PREFIX = "colbert:"


def _get_cross_encoder(model_name: str) -> ColBERT | CrossEncoder:
    """Return a cached reranker instance.

    Two backends are supported, selected by the *model_name* prefix:

    * ``"colbert:<hf-model>"`` uses :class:`pylate.models.ColBERT` for
      late-interaction reranking (token-level MaxSim scoring). Best for
      technical / out-of-domain content.
      Example: ``"colbert:colbert-ir/colbertv2.0"``
    * ``"<hf-model>"`` uses ``sentence_transformers.CrossEncoder`` for
      standard cross-encoder reranking.
      Example: ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``

    :param model_name: Model identifier, optionally prefixed with ``"colbert:"``.
    :return: A reranker instance (either a ``ColBERT`` or a ``CrossEncoder``).
    """
    if model_name not in _CROSS_ENCODER_CACHE:
        if model_name.startswith(_COLBERT_PREFIX):
            hf_name = model_name[len(_COLBERT_PREFIX) :]

            logger.info("Loading ColBERT reranker model: %s — will be cached", hf_name)
            _CROSS_ENCODER_CACHE[model_name] = ColBERT(model_name_or_path=hf_name)
        else:
            logger.info("Loading CrossEncoder reranker model: %s — will be cached", model_name)
            _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODER_CACHE[model_name]


def _rerank(
    query: str,
    contexts: list[RetrievedContext],
    model_name: str,
    top_n: int | None = None,
) -> list[RetrievedContext]:
    """Re-score *contexts* using a reranker and return the top-N.

    Supports two backends based on *model_name* (see :func:`_get_cross_encoder`):

    * **ColBERT** (``"colbert:<model>"``): encodes query and all passages with
      the ColBERT model then scores via MaxSim over token embeddings.
    * **CrossEncoder** (plain model name): scores each ``(query, passage)``
      pair with a bi-directional attention model.

    :param query: The original user query.
    :param contexts: Retrieved contexts to re-rank.
    :param model_name: Reranker model identifier (see :func:`_get_cross_encoder`).
    :param top_n: How many results to keep.  ``None`` keeps all.
    :return: Re-ranked (and possibly truncated) context list.
    """
    if not contexts:
        return contexts

    model = _get_cross_encoder(model_name)
    passages = [ctx.text for ctx in contexts]

    if model_name.startswith(_COLBERT_PREFIX):
        # pylate ColBERT reranking:
        # 1. encode query and all passages into multi-vector representations
        # 2. score via MaxSim (late interaction)

        queries_embeddings = model.encode(
            [query],
            is_query=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        documents_embeddings = model.encode(
            passages,
            is_query=False,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        # pylate_rerank expects list-of-lists for batch use; we have one query
        documents_ids = [[str(i) for i in range(len(passages))]]
        reranked = pylate_rerank(
            documents_ids=documents_ids,
            queries_embeddings=queries_embeddings,
            documents_embeddings=documents_embeddings,
        )[0]  # first (only) query
        # reranked is a list of {"id": str, "score": float} sorted desc
        id_to_ctx = {str(i): ctx for i, ctx in enumerate(contexts)}
        ranked_contexts = [
            RetrievedContext(
                text=id_to_ctx[entry["id"]].text,
                embedding=id_to_ctx[entry["id"]].embedding,
                score=float(entry["score"]),
            )
            for entry in reranked
        ]
    else:
        # sentence-transformers CrossEncoder reranking
        pairs = [[query, text] for text in passages]
        scores = model.predict(pairs)
        ranked_contexts = [
            RetrievedContext(text=ctx.text, embedding=ctx.embedding, score=float(score))
            for score, ctx in sorted(
                zip(scores, contexts, strict=True),
                key=lambda t: t[0],
                reverse=True,
            )
        ]

    limit = top_n if top_n is not None else len(ranked_contexts)
    return ranked_contexts[:limit]


# ---------------------------------------------------------------------------
# Thin adapters so that every embedder exposes ``embed([text]) → list``
# ---------------------------------------------------------------------------


class _LangchainEmbeddingAdapter:
    """Wrap a LangChain ``Embeddings`` object to expose ``embed([text])``."""

    def __init__(self, embeddings: OllamaEmbeddings | HuggingFaceEmbeddings) -> None:
        self._embeddings = embeddings

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)


class _FastEmbedSparseAdapter:
    """Wrap a FastEmbed ``SparseTextEmbedding`` to expose ``embed([text])``."""

    def __init__(self, model: SparseTextEmbedding) -> None:
        self._model = model

    def embed(self, texts: list[str]) -> list[Any]:
        return list(self._model.embed(texts))


class _FastEmbedLateInteractionAdapter:
    """Wrap a FastEmbed ``LateInteractionTextEmbedding`` to expose ``embed([text])``."""

    def __init__(self, model: LateInteractionTextEmbedding) -> None:
        self._model = model

    def embed(self, texts: list[str]) -> list[Any]:
        return list(self._model.embed(texts))


class QdrantRetrieverWrapper(Retriever):
    """Retrieve documents from a Qdrant collection using hybrid search.

    The wrapper holds a ``QdrantClient`` and the embedding providers needed to
    encode a query at search time.  It builds the prefetch / fusion / re-score
    pipeline described by the associated :class:`QdrantRetrieverConfig`.
    """

    def __init__(
        self,
        client: QdrantClient,
        config: QdrantRetrieverConfig,
        dense_embedder: _LangchainEmbeddingAdapter | None = None,
        sparse_embedder: _FastEmbedSparseAdapter | None = None,
        late_interaction_embedder: _FastEmbedLateInteractionAdapter | None = None,
    ) -> None:
        """Instantiate a Qdrant hybrid retriever.

        :param client: An already-connected ``QdrantClient``.
        :type client: QdrantClient
        :param config: The retriever configuration (fusion params, limits, …).
        :type config: QdrantRetrieverConfig
        :param dense_embedder: Callable / object whose ``embed([text])`` returns a
            dense vector (list[float]).
        :param sparse_embedder: Callable / object whose ``embed([text])`` returns a
            sparse result with ``.as_object()`` → ``{"indices": …, "values": …}``.
        :param late_interaction_embedder: Callable / object whose ``embed([text])``
            returns a list of token vectors for ColBERT.
        """
        self.client = client
        self.config = config
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.late_interaction_embedder = late_interaction_embedder

    # ---------------------------------------------------------------------- #
    # Factory
    # ---------------------------------------------------------------------- #

    @classmethod
    def make(
        cls,
        config: QdrantRetrieverConfig,
        input_chunks: Optional[list[str]] = None,  # noqa: ARG003 API compat
        *,
        client: Optional[QdrantClient] = None,
        dense_embedder: _LangchainEmbeddingAdapter | None = None,
        sparse_embedder: _FastEmbedSparseAdapter | None = None,
        late_interaction_embedder: _FastEmbedLateInteractionAdapter | None = None,
    ) -> QdrantRetrieverWrapper:
        """Build a :class:`QdrantRetrieverWrapper` from its configuration.

        If no ``client`` is supplied, one is created from the URL / API-key
        stored in *config*.

        When embedders are not explicitly provided they are automatically built
        from the collection metadata stored in *config*:

        * **dense**: via Ollama or HuggingFace (see ``encoder_provider``).
        * **sparse**: FastEmbed ``Qdrant/bm25``.
        * **late_interaction**: FastEmbed ``colbert-ir/colbertv2.0``
          (only built when ``late_interaction_rescore`` is enabled).

        :param config: Qdrant retriever configuration.
        :type config: QdrantRetrieverConfig
        :param input_chunks: Unused; kept for API compatibility with other
            retriever wrappers.
        :param client: Optional pre-existing Qdrant client.
        :param dense_embedder: Dense embedding provider.
        :param sparse_embedder: Sparse embedding provider.
        :param late_interaction_embedder: Late-interaction (ColBERT) provider.
        :return: The configured wrapper.
        :rtype: QdrantRetrieverWrapper
        """
        if client is None:
            client = _QdrantClient(
                url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
                api_key=os.environ.get("QDRANT_API_KEY"),
            )

        # Auto-build embedding providers when not supplied ─────────────────────
        if dense_embedder is None:
            dense_embedder = _build_dense_embedder(config)

        if sparse_embedder is None:
            sparse_embedder = _build_sparse_embedder()

        if late_interaction_embedder is None and config.late_interaction_rescore:
            late_interaction_embedder = _build_late_interaction_embedder()

        return cls(
            client=client,
            config=config,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            late_interaction_embedder=late_interaction_embedder,
        )

    # ---------------------------------------------------------------------- #
    # Embedding helpers
    # ---------------------------------------------------------------------- #

    def _embed_dense(self, text: str) -> list[float]:
        """Embed *text* with the dense provider.

        :raises RuntimeError: If no dense embedder was provided.
        """
        if self.dense_embedder is None:
            msg = "No dense embedder configured; cannot build dense prefetch."
            raise RuntimeError(msg)
        return self.dense_embedder.embed([text])[0]

    def _embed_sparse(self, text: str) -> dict[str, Any]:
        """Embed *text* with the sparse provider and return ``{"indices": …, "values": …}``."""
        if self.sparse_embedder is None:
            msg = "No sparse embedder configured; cannot build sparse prefetch."
            raise RuntimeError(msg)
        raw = self.sparse_embedder.embed([text])[0]
        # The result may already be a dict or expose ``.as_object()``.
        if hasattr(raw, "as_object"):
            return raw.as_object()
        return raw  # type: ignore[return-value]

    def _embed_late_interaction(self, text: str) -> list[list[float]]:
        """Embed *text* with the late-interaction (ColBERT) provider."""
        if self.late_interaction_embedder is None:
            msg = "No late-interaction embedder configured; cannot re-score."
            raise RuntimeError(msg)
        raw = self.late_interaction_embedder.embed([text])[0]
        return [tok.tolist() if hasattr(tok, "tolist") else list(tok) for tok in raw]

    # ---------------------------------------------------------------------- #
    # Query execution
    # ---------------------------------------------------------------------- #

    def get_retriever_output(self, query: str) -> list[RetrievedContext]:
        """Execute a hybrid search and return scored contexts.

        :param query: The user query.
        :type query: str
        :return: Retrieved contexts sorted by descending score.
        :rtype: list[RetrievedContext]
        """
        cfg = self.config
        coll = cfg.collection

        # ── Build prefetches ──────────────────────────────────────────────────
        # Order matters: weights[i] in RrfQuery corresponds to prefetches[i].
        # We put dense first, sparse second → weights=[dense_weight, sparse_weight].
        prefetches: list[Prefetch] = []

        # Dense
        dense_vector = self._embed_dense(query)
        prefetches.append(
            Prefetch(
                query=dense_vector,
                using=coll.dense_vector_name,
                limit=cfg.prefetch_limit,
            ),
        )

        # Sparse / BM25
        sparse_obj = self._embed_sparse(query)
        indices = sparse_obj["indices"]
        values = sparse_obj["values"]
        if hasattr(indices, "tolist"):
            indices = indices.tolist()
        if hasattr(values, "tolist"):
            values = values.tolist()
        prefetches.append(
            Prefetch(
                query=SparseVector(indices=list(indices), values=list(values)),
                using=coll.sparse_vector_name,
                limit=cfg.prefetch_limit,
            ),
        )

        # ── Build fusion query ────────────────────────────────────────────────
        if cfg.fusion_method == "rrf":
            fusion_query: RrfQuery | FusionQuery = RrfQuery(
                rrf=Rrf(
                    k=cfg.rrf_k,
                    weights=[cfg.dense_weight, cfg.sparse_weight],
                ),
            )
        else:
            fusion_query = FusionQuery(fusion=Fusion.DBSF)

        # ── Optional ColBERT re-score (3-stage) ──────────────────────────────
        if cfg.late_interaction_rescore and self.late_interaction_embedder is not None:
            colbert_vector = self._embed_late_interaction(query)
            logger.info("Running 3-stage hybrid query: prefetch → fusion → ColBERT re-score")
            results = self.client.query_points(
                collection_name=coll.collection_name,
                prefetch=Prefetch(
                    prefetch=prefetches,
                    query=fusion_query,
                    limit=cfg.prefetch_limit,
                ),
                query=colbert_vector,
                using=coll.late_interaction_vector_name,
                with_payload=True,
                limit=cfg.limit,
            )
        else:
            logger.info("Running 2-stage hybrid query: prefetch → fusion")
            results = self.client.query_points(
                collection_name=coll.collection_name,
                prefetch=prefetches,
                query=fusion_query,
                with_payload=True,
                limit=cfg.limit,
            )

        # ── Convert to RetrievedContext ───────────────────────────────────────
        retrieved: list[RetrievedContext] = []
        for point in results.points:
            # Apply score threshold: discard results below the minimum
            if cfg.score_threshold is not None and point.score < cfg.score_threshold:
                continue
            payload = point.payload or {}
            text = payload.get("text", payload.get("page_content", ""))
            if not text:
                # Fall back to stringified payload if no text field
                text = str(payload)
            retrieved.append(
                RetrievedContext(
                    text=text,
                    score=point.score,
                ),
            )

        logger.info(
            "Returned %d results (threshold=%s, limit=%d)",
            len(retrieved),
            cfg.score_threshold,
            cfg.limit,
        )

        # ── Optional cross-encoder reranking ──────────────────────────────────
        if cfg.reranker_model is not None:
            logger.info(
                "Reranking %d results with cross-encoder '%s' (top_n=%s)",
                len(retrieved),
                cfg.reranker_model,
                cfg.reranker_top_n,
            )
            retrieved = _rerank(
                query=query,
                contexts=retrieved,
                model_name=cfg.reranker_model,
                top_n=cfg.reranker_top_n,
            )
            logger.info("After reranking: %d results", len(retrieved))

        return retrieved
