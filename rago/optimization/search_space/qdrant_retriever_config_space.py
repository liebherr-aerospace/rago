"""Define the Qdrant hybrid retriever config space for Optuna optimisation.

This config space allows tuning all the parameters of a Qdrant hybrid search:

* **Collection selection** pick among several pre-indexed Qdrant collections,
  each backed by its own dense encoder.
* **Fusion method** ``"rrf"`` (Reciprocal Rank Fusion) or ``"dbsf"``
  (Distribution-Based Score Fusion).
* **RRF parameters** ``rrf_k`` and per-prefetch *weights*
  (``dense_weight``, ``sparse_weight``).
* **Prefetch / result limits** ``prefetch_limit``, ``limit``.
* **Late-interaction re-scoring** enable / disable ColBERT re-scoring as a
  3rd stage after fusion.

Default behaviour (when instantiated with no arguments) matches the YAML
reference config: RRF fusion with ``k=60``, weights ``[3.0, 1.0]``
(dense, sparse), ``prefetch_limit=100``, ``limit=20``, ColBERT off,
sparse provider ``Qdrant/bm25``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.model.configs.retriever_config.qdrant import (
    QdrantCollectionConfig,
    QdrantRetrieverConfig,
)
from rago.optimization.search_space.config_space import ConfigSpace
from rago.optimization.search_space.param_space import (
    CategoricalParamSpace,
    FloatParamSpace,
    IntParamSpace,
)

if TYPE_CHECKING:
    import optuna


# ---------------------------------------------------------------------------
# Helper: a "collection slot" that ties together a collection name and its
# associated dense encoder (provider, model, base_url).
# ---------------------------------------------------------------------------


@dataclass
class QdrantCollectionSlot:
    """One pre-indexed Qdrant collection with its dense encoder metadata.

    Instances of this class are *not* sampled they describe a fixed,
    already-indexed collection.  The config space picks *which* slot to
    use during a trial (via a categorical over the collection names).

    :param collection_name: Qdrant collection name.
    :type collection_name: str
    :param dense_vector_name: Named vector field for dense search.
    :type dense_vector_name: str
    :param sparse_vector_name: Named vector field for sparse search.
    :type sparse_vector_name: str
    :param late_interaction_vector_name: Named vector field for ColBERT.
    :type late_interaction_vector_name: str
    :param encoder_model: Dense encoder model identifier.
    :type encoder_model: str
    :param encoder_provider: Dense encoder provider (e.g. ``"ollama"``).
    :type encoder_provider: str
    :param encoder_base_url: Base URL for the encoder provider.
    :type encoder_base_url: str
    """

    collection_name: str
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "bm25"
    late_interaction_vector_name: str = "colbert"
    encoder_model: str = "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:f16"
    encoder_provider: str = "ollama"
    encoder_base_url: str = Field(default_factory=lambda: os.environ.get("TEST_OLLAMA_HOST", ""))


# ---------------------------------------------------------------------------
# Config space
# ---------------------------------------------------------------------------


@dataclass
class QdrantRetrieverConfigSpace(ConfigSpace):
    """Search space for the Qdrant hybrid retriever.

    Example:
    -------
    >>> from rago.optimization.search_space.qdrant_retriever_config_space import (
    ...     QdrantCollectionSlot,
    ...     QdrantRetrieverConfigSpace,
    ... )
    >>> space = QdrantRetrieverConfigSpace(
    ...     collections=[
    ...         QdrantCollectionSlot(
    ...             collection_name="my_collection_bge",
    ...             encoder_model="BAAI/bge-m3",
    ...             encoder_provider="ollama",
    ...             encoder_base_url="https://ollama.example.com",
    ...         ),
    ...         QdrantCollectionSlot(
    ...             collection_name="my_collection_qwen",
    ...             encoder_model="hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:f16",
    ...             encoder_provider="ollama",
    ...             encoder_base_url="https://ollama.example.com",
    ...         ),
    ...     ],
    ... )

    """

    # ── Collection slots (each one maps to a pre-indexed collection) ─────────
    collections: list[QdrantCollectionSlot] = Field(
        default_factory=lambda: [
            QdrantCollectionSlot(collection_name="default_collection"),
        ],
    )

    # ── Tunable: fusion method ───────────────────────────────────────────────
    fusion_method: CategoricalParamSpace = Field(
        default=CategoricalParamSpace(choices=["rrf", "dbsf"]),
    )

    # ── Tunable: RRF k ──────────────────────────────────────────────────────
    rrf_k: IntParamSpace = Field(
        default=IntParamSpace(low=1, high=200, step=1),
    )

    # ── Tunable: prefetch_limit ──────────────────────────────────────────────
    prefetch_limit: IntParamSpace = Field(
        default=IntParamSpace(low=20, high=500, step=10),
    )

    # ── Tunable: final result limit ──────────────────────────────────────────
    limit: IntParamSpace = Field(
        default=IntParamSpace(low=1, high=50, step=1),
    )

    # ── Tunable: late-interaction re-scoring (ColBERT on / off) ──────────────
    late_interaction_rescore: CategoricalParamSpace = Field(
        default=CategoricalParamSpace(choices=[True, False]),
    )

    # ── Tunable: RRF weights (dense then sparse) ────────────────────────────
    dense_weight: FloatParamSpace = Field(
        default=FloatParamSpace(low=0.1, high=10.0),
    )
    sparse_weight: FloatParamSpace = Field(
        default=FloatParamSpace(low=0.1, high=10.0),
    )

    # ── Tunable: minimum score threshold (post-fusion) ───────────────────────
    # Set to None to disable.  When a FloatParamSpace is provided, the sampled
    # value is used as a hard floor: results below it are discarded.
    score_threshold: FloatParamSpace | None = None

    # ── Tunable: post-retrieval cross-encoder reranker ───────────────────────
    # Set to None (default) to disable reranking entirely.
    # When a CategoricalParamSpace is provided, the choices should contain
    # ``None`` (no reranker) and/or model identifiers, e.g.:
    #   a CategoricalParamSpace "cross-encoder/ms-marco-MiniLM-L-6-v2".
    reranker_model: CategoricalParamSpace | None = None

    # ── Tunable: how many results the reranker keeps ─────────────────────────
    # Only used when reranker_model is not None.
    reranker_top_n: IntParamSpace | None = None

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _collection_names(self) -> list[str]:
        """Return the list of collection names available for sampling."""
        return [slot.collection_name for slot in self.collections]

    def _slot_by_name(self, name: str) -> QdrantCollectionSlot:
        """Return the slot matching *name*.

        :raises ValueError: If *name* is not found.
        """
        for slot in self.collections:
            if slot.collection_name == name:
                return slot
        msg = f"Collection '{name}' not found in the configured slots."
        raise ValueError(msg)

    # ---------------------------------------------------------------------- #
    # Sampling
    # ---------------------------------------------------------------------- #

    def sample(self, trial: optuna.trial.BaseTrial) -> QdrantRetrieverConfig:
        """Sample a ``QdrantRetrieverConfig`` from the search space.

        :param trial: The Optuna trial driving the sampling.
        :type trial: optuna.trial.BaseTrial
        :return: A fully-specified Qdrant retriever configuration.
        :rtype: QdrantRetrieverConfig
        """
        # ── Collection selection ──────────────────────────────────────────────
        if len(self.collections) == 1:
            chosen_name = self.collections[0].collection_name
        else:
            chosen_name = trial.suggest_categorical(
                "QdrantRetrieverConfigSpace_collection_name",
                self._collection_names(),
            )
        if not isinstance(chosen_name, str):
            raise TypeError(chosen_name)
        slot = self._slot_by_name(chosen_name)

        # ── Fusion method ─────────────────────────────────────────────────────
        fusion_method = self.fusion_method.sample(trial)
        if not isinstance(fusion_method, str):
            raise TypeError(fusion_method)

        # ── Numeric / boolean parameters ──────────────────────────────────────
        rrf_k = self.rrf_k.sample(trial) if fusion_method == "rrf" else 60
        prefetch_limit = self.prefetch_limit.sample(trial)
        limit = self.limit.sample(trial)
        late_interaction_rescore = self.late_interaction_rescore.sample(trial)
        dense_weight = self.dense_weight.sample(trial)
        sparse_weight = self.sparse_weight.sample(trial)
        score_threshold = self.score_threshold.sample(trial) if self.score_threshold is not None else None

        # ── Reranker ──────────────────────────────────────────────────────────
        reranker_model = self.reranker_model.sample(trial) if self.reranker_model is not None else None
        # Cast to str | None (Optuna categorical returns the value as-is).
        reranker_model = str(reranker_model) if reranker_model is not None else None

        # reranker_top_n is only meaningful when a reranker is active.
        if reranker_model is not None and self.reranker_top_n is not None:
            reranker_top_n: int | None = self.reranker_top_n.sample(trial)
        else:
            reranker_top_n = None

        return QdrantRetrieverConfig(
            collection=QdrantCollectionConfig(
                collection_name=slot.collection_name,
                dense_vector_name=slot.dense_vector_name,
                sparse_vector_name=slot.sparse_vector_name,
                late_interaction_vector_name=slot.late_interaction_vector_name,
                encoder_model=slot.encoder_model,
                encoder_provider=slot.encoder_provider,
                encoder_base_url=slot.encoder_base_url,
            ),
            fusion_method=fusion_method,
            rrf_k=rrf_k,
            prefetch_limit=prefetch_limit,
            limit=limit,
            late_interaction_rescore=bool(late_interaction_rescore),
            dense_weight=float(dense_weight),
            sparse_weight=float(sparse_weight),
            score_threshold=float(score_threshold) if score_threshold is not None else None,
            reranker_model=reranker_model,
            reranker_top_n=reranker_top_n,
        )
