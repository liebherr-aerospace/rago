"""Define the langchain reader config."""

from __future__ import annotations

from typing import Optional

from pydantic.dataclasses import dataclass

from rago.model.configs.llm_config.langchain import LangchainLLMConfig  # noqa: TC001
from rago.model.configs.reader_config.base import ReaderConfig


@dataclass
class LangchainReaderConfig(ReaderConfig):
    """Configuration parameters of the reader."""

    llm: Optional[LangchainLLMConfig] = None
