"""Define the Qdrant hybrid retriever config."""

from __future__ import annotations

import os

from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.model.configs.retriever_config.base import RetrieverConfig


@dataclass
class QdrantCollectionConfig:
    """Describes one Qdrant collection and its associated dense encoder.

    :param collection_name: The name of the Qdrant collection.
    :type collection_name: str
    :param dense_vector_name: The named vector field used for dense search (e.g. ``"dense"``).
    :type dense_vector_name: str
    :param sparse_vector_name: The named vector field used for sparse/BM25 search (e.g. ``"bm25"``).
    :type sparse_vector_name: str
    :param late_interaction_vector_name: Named vector for ColBERT re-scoring (e.g. ``"colbert"``).
    :type late_interaction_vector_name: str
    :param encoder_model: The dense encoder model identifier used at indexing time
        (e.g. ``"hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:f16"``).
    :type encoder_model: str
    :param encoder_provider: The provider for the dense encoder (``"ollama"``).
    :type encoder_provider: str
    :param encoder_base_url: Base URL for the dense encoder provider.
    :type encoder_base_url: str
    """

    collection_name: str
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "bm25"
    late_interaction_vector_name: str = "colbert"
    encoder_model: str = "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:f16"
    encoder_provider: str = "ollama"
    encoder_base_url: str = Field(default_factory=lambda: os.environ.get("TEST_OLLAMA_HOST", ""))


@dataclass
class QdrantRetrieverConfig(RetrieverConfig):
    """Configuration of the Qdrant hybrid retriever.

    This captures every parameter needed to execute a 2-stage (dense + sparse → fusion)
    or 3-stage (dense + sparse → fusion → ColBERT re-score) hybrid search on a
    single Qdrant collection.

    Tunable parameters
    ------------------
    * ``fusion_method``: ``"rrf"`` or ``"dbsf"``.
    * ``rrf_k``: The *k* constant in the RRF formula (only used when ``fusion_method == "rrf"``).
    * ``prefetch_limit``: How many candidates each sub-query returns.
    * ``limit``: Final result limit returned to the caller.
    * ``late_interaction_rescore``: Whether to add ColBERT re-scoring as a third stage.
    * ``dense_weight``: Weight applied to the dense prefetch in RRF weighted mode.
    * ``sparse_weight``: Weight applied to the sparse prefetch in RRF weighted mode.
    * ``score_threshold``: Minimum fusion score to keep a result. ``None`` disables filtering.
    """

    collection: QdrantCollectionConfig
    fusion_method: str = "rrf"
    rrf_k: int = 60
    prefetch_limit: int = 100
    limit: int = 20
    late_interaction_rescore: bool = False
    dense_weight: float = 3.0
    sparse_weight: float = 1.0
    score_threshold: float | None = None
    reranker_model: str | None = None
    reranker_top_n: int | None = None
