"""Define the langchain llm config."""

from __future__ import annotations

import os
from typing import Optional

from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.model.configs.llm_config.base import LLMConfig


@dataclass
class LangchainLLMConfig(LLMConfig):
    """Configuration parameters of the LLM."""

    model_name: str
    temperature: float = Field(default=0.5)
    top_k: int = Field(default=40)
    top_p: float = Field(default=0.9)
    max_new_tokens: int = Field(default=128)


@dataclass
class LangchainOllamaConfig(LangchainLLMConfig):
    """Configuration parameters of the Ollama LLM."""

    model_name: str = Field(default="qwen2.5:1.5b")
    num_ctx: int = Field(default=500)
    repeat_last_n: int = Field(default=64)
    mirostat: int = Field(default=0)
    mirostat_eta: Optional[float] = Field(default=None)
    mirostat_tau: Optional[float] = Field(default=None)
    base_url: str = Field(default=os.environ.get("TEST_OLLAMA_HOST", ""))
    client_kwargs: dict[str, bool] = Field(default={"verify": False})


@dataclass
class LangchainHuggingFaceConfig(LangchainLLMConfig):
    """Configuration parameters for HuggingFace Inference API LLM."""

    model_name: str = Field(default="meta-llama/Llama-3.1-8B-Instruct")
    huggingfacehub_api_token: Optional[str] = Field(default=os.environ.get("HF_API_TOKEN"))
    endpoint_url: Optional[str] = Field(default=os.environ.get("HF_ENDPOINT_URL"))
