"""Define the llama-index reader config."""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.model.configs.llm_config.llama_index import LLamaIndexLLMConfig  # noqa: TC001
from rago.model.configs.reader_config.base import ReaderConfig


@dataclass
class LLamaIndexReaderConfig(ReaderConfig):
    """Configuration parameters of the reader."""

    type: str = Field(default = "compact")
    llm: Optional[LLamaIndexLLMConfig] = None
