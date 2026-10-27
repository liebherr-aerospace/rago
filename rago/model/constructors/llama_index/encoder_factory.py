"""Defines a factory of LlamaIndex encoders."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rago.model.configs.encoder_config import (
    HuggingFaceLlaIndexEncoderConfig,
    LlamaIndexEncoderConfig,
    OllamaLlamaIndexEncoderConfig,
)

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)


class EncoderFactory:
    """A encoder Factory to build llama-index encoders.

    Encoder instances are cached by model name so that the same model is
    loaded only once across multiple Optuna trials.
    """

    _hf_cache: ClassVar[dict[tuple[str, int], HuggingFaceEmbedding]] = {}

    @staticmethod
    def make(config: LlamaIndexEncoderConfig) -> BaseEmbedding:
        """Build the encoder from its name and backend config.

        :param model_name: The name of the encoder to build.
        :type model_name: str
        :return: The built encoder.
        :rtype: BaseEmbedding
        """
        match config:
            case HuggingFaceLlaIndexEncoderConfig():
                return EncoderFactory.get_hugging_face_embedding(config.model_name)
            case OllamaLlamaIndexEncoderConfig():
                raise NotImplementedError
            case _:
                raise TypeError(config)

    @staticmethod
    def get_hugging_face_embedding(encoder_name: str, embed_batch_size: int = 32) -> HuggingFaceEmbedding:
        """Get a HuggingFace encoder, returning a cached instance if available.

        :param encoder_name: Name of the encoder to build.
        :type encoder_name: str
        :param embed_batch_size: Batch size to use for the encoder, defaults to 32.
        :type embed_batch_size: Optional[int], optional
        :return: The built HuggingFace encoder.
        :rtype: HuggingFaceEmbedding
        """
        cache_key = (encoder_name, embed_batch_size)
        if cache_key in EncoderFactory._hf_cache:
            logger.debug("[CACHE HIT] Reusing LlamaIndex HuggingFace encoder '%s'", encoder_name)
            return EncoderFactory._hf_cache[cache_key]

        logger.info("[CACHE MISS] Loading LlamaIndex HuggingFace encoder '%s'", encoder_name)
        encoder = HuggingFaceEmbedding(model_name=encoder_name, embed_batch_size=embed_batch_size)
        EncoderFactory._hf_cache[cache_key] = encoder
        return encoder

    @staticmethod
    def clear_cache() -> None:
        """Clear all cached encoder instances."""
        EncoderFactory._hf_cache.clear()
        logger.info("[CACHE] LlamaIndex encoder cache cleared")
