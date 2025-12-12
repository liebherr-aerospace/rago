"""Define the node post-processor config."""

from __future__ import annotations

from typing import Optional

from pydantic.dataclasses import dataclass

from rago.model.configs.base import Config
from rago.model.configs.encoder_config.llama_index import LlamaIndexEncoderConfig  # noqa: TC001
from rago.model.configs.llm_config.llama_index import LLamaIndexLLMConfig  # noqa: TC001


@dataclass
class NodePostProcessorConfig(Config):
    """Configuration parameters of the node post-processor."""

    post_processor_type: str
    similarity_cutoff: Optional[float] = None
    percentile_cutoff: Optional[float] = None
    threshold_cutoff: Optional[float] = None
    choice_batch_size: Optional[int] = None
    top_n: Optional[int] = None
    reranker_name: Optional[str] = None
    llm: Optional[LLamaIndexLLMConfig] = None
    encoder: Optional[LlamaIndexEncoderConfig] = None
