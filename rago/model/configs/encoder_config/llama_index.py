"""Define the llama-index encoder config."""

import os

from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.model.configs.base import Config


@dataclass
class LlamaIndexEncoderConfig(Config):
    """Configuration parameters of the encoder."""

    model_name: str


@dataclass
class OllamaLlamaIndexEncoderConfig(LlamaIndexEncoderConfig):
    """Configuration parameters of the Ollama encoder."""

    base_url: str = Field(default=os.environ.get("TEST_OLLAMA_HOST", ""))
    client_kwargs: dict[str, bool] = Field(default={"verify": False})


@dataclass
class HuggingFaceLlaIndexEncoderConfig(LlamaIndexEncoderConfig):
    """Configuration parameters of the HuggingFace encoder."""
