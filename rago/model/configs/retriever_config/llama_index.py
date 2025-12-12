"""Define the retriever."""

from __future__ import annotations

from typing import Optional

from pydantic.dataclasses import dataclass

from rago.model.configs.encoder_config.llama_index import LlamaIndexEncoderConfig  # noqa: TC001
from rago.model.configs.index_config.llama_index import IndexConfig  # noqa: TC001
from rago.model.configs.llm_config.llama_index import LLamaIndexLLMConfig  # noqa: TC001
from rago.model.configs.post_processor_config.llama_index import NodePostProcessorConfig  # noqa: TC001
from rago.model.configs.retriever_config.base import RetrieverConfig


@dataclass
class SelectorConfig:
    """Configurations parameters of the selector."""

    selector_type: str
    max_outputs: int


@dataclass
class LlamaIndexRetrieverConfig(RetrieverConfig):
    """Configuration parameters of the retriever."""

    type: str
    index: IndexConfig
    encoder: Optional[LlamaIndexEncoderConfig] = None
    node_post_processor_config: Optional[list[NodePostProcessorConfig]] = None
    llm: Optional[LLamaIndexLLMConfig] = None
    similarity_top_k: Optional[int] = None
    vector_store_query_mode: Optional[str] = None
    retriever_tools: Optional[list[LlamaIndexRetrieverConfig]] = None
    selector: Optional[SelectorConfig] = None
    child_branch_factor: Optional[int] = None
