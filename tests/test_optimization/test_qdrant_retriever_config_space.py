"""Tests for the Qdrant hybrid retriever config space and config."""

from __future__ import annotations

import optuna
import pytest

from rago.model.configs.retriever_config.qdrant import (
    QdrantCollectionConfig,
    QdrantRetrieverConfig,
)
from rago.optimization.search_space.param_space import CategoricalParamSpace
from rago.optimization.search_space.qdrant_retriever_config_space import (
    QdrantCollectionSlot,
    QdrantRetrieverConfigSpace,
)

DEFAULT_RRF_K = 60
DEFAULT_PREFETCH_LIMIT = 100
DEFAULT_LIMIT = 20
DEFAULT_DENSE_WEIGHT = 3.0
DEFAULT_SPARSE_WEIGHT = 1.0

CUSTOM_RRF_K = 30
CUSTOM_PREFETCH_LIMIT = 200
CUSTOM_LIMIT = 10
CUSTOM_DENSE_WEIGHT = 2.0
CUSTOM_SPARSE_WEIGHT = 5.0

PREFETCH_LIMIT_RANGE = (20, 500)
LIMIT_RANGE = (1, 50)
WEIGHT_RANGE = (0.1, 10.0)

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def single_collection_space() -> QdrantRetrieverConfigSpace:
    """Config space with a single collection (no collection selection to tune)."""
    return QdrantRetrieverConfigSpace(
        collections=[
            QdrantCollectionSlot(
                collection_name="test_collection",
                encoder_model="BAAI/bge-m3",
                encoder_provider="ollama",
                encoder_base_url="https://ollama.example.com",
            ),
        ],
    )


@pytest.fixture
def multi_collection_space() -> QdrantRetrieverConfigSpace:
    """Config space with multiple collections (collection is tuned)."""
    return QdrantRetrieverConfigSpace(
        collections=[
            QdrantCollectionSlot(
                collection_name="collection_bge",
                encoder_model="BAAI/bge-m3",
                encoder_provider="ollama",
                encoder_base_url="https://ollama.example.com",
            ),
            QdrantCollectionSlot(
                collection_name="collection_qwen",
                encoder_model="hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:f16",
                encoder_provider="ollama",
                encoder_base_url="https://ollama.example.com",
            ),
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# QdrantRetrieverConfig dataclass tests
# ──────────────────────────────────────────────────────────────────────────────


class TestQdrantRetrieverConfig:
    """Tests for the QdrantRetrieverConfig dataclass."""

    def test_default_values(self) -> None:
        """Defaults match the YAML reference: rrf, k=60, etc."""
        cfg = QdrantRetrieverConfig(collection=QdrantCollectionConfig(collection_name="test_collection"))
        assert cfg.fusion_method == "rrf"
        assert cfg.rrf_k == DEFAULT_RRF_K
        assert cfg.prefetch_limit == DEFAULT_PREFETCH_LIMIT
        assert cfg.limit == DEFAULT_LIMIT
        assert cfg.late_interaction_rescore is False
        assert cfg.dense_weight == DEFAULT_DENSE_WEIGHT
        assert cfg.sparse_weight == DEFAULT_SPARSE_WEIGHT

    def test_custom_values(self) -> None:
        """Config can be created with custom values."""
        cfg = QdrantRetrieverConfig(
            fusion_method="dbsf",
            rrf_k=CUSTOM_RRF_K,
            prefetch_limit=CUSTOM_PREFETCH_LIMIT,
            limit=CUSTOM_LIMIT,
            late_interaction_rescore=True,
            dense_weight=CUSTOM_DENSE_WEIGHT,
            sparse_weight=CUSTOM_SPARSE_WEIGHT,
            collection=QdrantCollectionConfig(
                collection_name="my_coll",
                dense_vector_name="my_dense",
                sparse_vector_name="my_sparse",
            ),
        )
        assert cfg.fusion_method == "dbsf"
        assert cfg.rrf_k == CUSTOM_RRF_K
        assert cfg.prefetch_limit == CUSTOM_PREFETCH_LIMIT
        assert cfg.limit == CUSTOM_LIMIT
        assert cfg.late_interaction_rescore is True
        assert cfg.dense_weight == CUSTOM_DENSE_WEIGHT
        assert cfg.sparse_weight == CUSTOM_SPARSE_WEIGHT
        assert cfg.collection.collection_name == "my_coll"
        assert cfg.collection.dense_vector_name == "my_dense"
        assert cfg.collection.sparse_vector_name == "my_sparse"


# ──────────────────────────────────────────────────────────────────────────────
# QdrantRetrieverConfigSpace sampling tests
# ──────────────────────────────────────────────────────────────────────────────


class TestQdrantRetrieverConfigSpaceSampling:
    """Tests for sampling from QdrantRetrieverConfigSpace."""

    def test_sample_returns_qdrant_config(
        self,
        single_collection_space: QdrantRetrieverConfigSpace,
    ) -> None:
        """Sampling produces a QdrantRetrieverConfig instance."""
        study = optuna.create_study()
        trial = study.ask()
        config = single_collection_space.sample(trial)
        assert isinstance(config, QdrantRetrieverConfig)

    def test_single_collection_not_tuned(
        self,
        single_collection_space: QdrantRetrieverConfigSpace,
    ) -> None:
        """With a single collection the name is fixed (not an Optuna param)."""
        study = optuna.create_study()
        trial = study.ask()
        config = single_collection_space.sample(trial)
        assert config.collection.collection_name == "test_collection"
        assert config.collection.encoder_model == "BAAI/bge-m3"
        # No collection_name param should have been suggested
        assert "QdrantRetrieverConfigSpace_collection_name" not in trial.params

    def test_multi_collection_is_tuned(
        self,
        multi_collection_space: QdrantRetrieverConfigSpace,
    ) -> None:
        """With multiple collections the collection name becomes a categorical param."""
        study = optuna.create_study()
        trial = study.ask()
        config = multi_collection_space.sample(trial)
        assert config.collection.collection_name in {
            "collection_bge",
            "collection_qwen",
        }
        assert "QdrantRetrieverConfigSpace_collection_name" in trial.params

    def test_fusion_method_is_sampled(
        self,
        single_collection_space: QdrantRetrieverConfigSpace,
    ) -> None:
        """fusion_method should be one of the allowed choices."""
        study = optuna.create_study()
        trial = study.ask()
        config = single_collection_space.sample(trial)
        assert config.fusion_method in {"rrf", "dbsf"}

    def test_limits_in_range(
        self,
        single_collection_space: QdrantRetrieverConfigSpace,
    ) -> None:
        """prefetch_limit and limit should be within the configured ranges."""
        study = optuna.create_study()
        trial = study.ask()
        config = single_collection_space.sample(trial)
        assert PREFETCH_LIMIT_RANGE[0] <= config.prefetch_limit <= PREFETCH_LIMIT_RANGE[1]
        assert LIMIT_RANGE[0] <= config.limit <= LIMIT_RANGE[1]

    def test_weights_in_range(
        self,
        single_collection_space: QdrantRetrieverConfigSpace,
    ) -> None:
        """Dense and sparse weights should be within the configured ranges."""
        study = optuna.create_study()
        trial = study.ask()
        config = single_collection_space.sample(trial)
        assert WEIGHT_RANGE[0] <= config.dense_weight <= WEIGHT_RANGE[1]
        assert WEIGHT_RANGE[0] <= config.sparse_weight <= WEIGHT_RANGE[1]

    def test_late_interaction_rescore_is_bool(
        self,
        single_collection_space: QdrantRetrieverConfigSpace,
    ) -> None:
        """late_interaction_rescore is cast to a proper bool."""
        study = optuna.create_study()
        trial = study.ask()
        config = single_collection_space.sample(trial)
        assert isinstance(config.late_interaction_rescore, bool)

    def test_rrf_k_not_sampled_for_dbsf(self) -> None:
        """When fusion_method is forced to 'dbsf', rrf_k should stay at default 60."""
        space = QdrantRetrieverConfigSpace(
            fusion_method=CategoricalParamSpace(choices=["dbsf"]),
        )
        study = optuna.create_study()
        trial = study.ask()
        config = space.sample(trial)
        assert config.fusion_method == "dbsf"
        assert config.rrf_k == DEFAULT_RRF_K

    def test_collection_metadata_preserved(
        self,
        multi_collection_space: QdrantRetrieverConfigSpace,
    ) -> None:
        """Encoder metadata from the chosen slot is correctly propagated."""
        study = optuna.create_study()
        # Run enough trials to cover both collections
        seen_models: set[str] = set()
        for _ in range(20):
            trial = study.ask()
            config = multi_collection_space.sample(trial)
            seen_models.add(config.collection.encoder_model)
            study.tell(trial, 1.0)  # dummy value

        # With 20 trials, we should have seen both models
        assert "BAAI/bge-m3" in seen_models or "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:f16" in seen_models


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────────────


class TestQdrantRetrieverConfigSpaceEdgeCases:
    """Edge-case tests for the Qdrant config space."""

    def test_default_space_has_one_collection(self) -> None:
        """Instantiating with no arguments gives a single default collection."""
        space = QdrantRetrieverConfigSpace()
        assert len(space.collections) == 1
        assert space.collections[0].collection_name == "default_collection"

    def test_slot_by_name_raises_on_unknown(
        self,
        single_collection_space: QdrantRetrieverConfigSpace,
    ) -> None:
        """Accessing an unknown collection name raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            single_collection_space._slot_by_name("nonexistent")  # noqa: SLF001

    def test_deterministic_with_fixed_seed(
        self,
        single_collection_space: QdrantRetrieverConfigSpace,
    ) -> None:
        """Two studies with the same seed produce the same first sample."""
        configs = []
        for _ in range(2):
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(sampler=sampler)
            trial = study.ask()
            configs.append(single_collection_space.sample(trial))
        assert configs[0].fusion_method == configs[1].fusion_method
        assert configs[0].prefetch_limit == configs[1].prefetch_limit
        assert configs[0].limit == configs[1].limit
