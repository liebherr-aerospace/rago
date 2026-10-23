"""Define the langchain retriever config."""

from __future__ import annotations

from typing import Any, Optional

from pydantic.dataclasses import dataclass

from rago.model.configs.encoder_config.langchain import LangchainEncoderConfig  # noqa: TC001
from rago.model.configs.post_processor_config.llama_index import NodePostProcessorConfig  # noqa: TC001
from rago.model.configs.retriever_config.base import RetrieverConfig


@dataclass
class LangchainRetrieverConfig(RetrieverConfig):
    """Configuration of the langchain retriever."""

    type: str
    similarity_function: Optional[str] = None
    search_type: Optional[str] = None
    search_kwargs: Optional[dict[str, Any]] = None
    encoder: Optional[LangchainEncoderConfig] = None
    node_post_processor_config: Optional[list[NodePostProcessorConfig]] = None

    # Hybrid retriever fields
    vector_config: Optional[LangchainRetrieverConfig] = None
    bm25_config: Optional[LangchainRetrieverConfig] = None
    hybrid_weight: Optional[float] = None  # Weight for vector retriever (1 - weight for BM25)
