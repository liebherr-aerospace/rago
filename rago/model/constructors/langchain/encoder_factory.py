"""Defines a factory of Langchain encoders."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

from rago.model.configs.encoder_config import (
    HuggingFaceLangchainEncoderConfig,
    LangchainEncoderConfig,
    OllamaLangchainEncoderConfig,
)

if TYPE_CHECKING:
    from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)

ENCODER_CACHE_MAX_SIZE = int(os.getenv("RAGO_ENCODER_CACHE_MAX_SIZE", "0"))


class EncoderFactory:
    """An encoder factory to make Langchain encoders.

    Encoder instances are cached by their configuration key so that the same
    model is loaded only once, even when the search space contains multiple
    encoder candidates evaluated across many Optuna trials.
    """

    _hf_cache: ClassVar[dict[tuple[str, Optional[int]], HuggingFaceEmbeddings]] = {}
    _ollama_cache: ClassVar[dict[tuple[str, Optional[str]], OllamaEmbeddings]] = {}

    @staticmethod
    def make(encoder_config: LangchainEncoderConfig) -> Embeddings:
        """Build the encoder from its name and backend config.

        :param model_name: The name of the encoder to build.
        :type model_name: str
        :return: The built encoder.
        :rtype: BaseEmbedding
        """
        match encoder_config:
            case OllamaLangchainEncoderConfig():
                return EncoderFactory.get_ollama_embedding(
                    encoder_name=encoder_config.model_name,
                    base_url=encoder_config.base_url,
                    client_kwargs=encoder_config.client_kwargs,
                )
            case HuggingFaceLangchainEncoderConfig():
                return EncoderFactory.get_hugging_face_embedding(encoder_config.model_name)
            case _:
                raise TypeError(encoder_config)

    @staticmethod
    def get_ollama_embedding(
        encoder_name: str,
        base_url: Optional[str] = None,
        client_kwargs: Optional[dict[str, Any]] = None,
    ) -> OllamaEmbeddings:
        """Get an Ollama encoder, returning a cached instance if available.

        :param encoder_name: Name of the encoder to build
        :type encoder_name: str
        :param base_url: Url of the Ollama server.
        :type base_url: str
        :param client_kwargs: The kwargs used by the ollama client, defaults to None
        :type client_kwargs: Optional[dict[str, Any]], optional
        :return: The built ollama embedding.
        :rtype: OllamaEmbeddings
        """
        cache_key = (encoder_name, base_url)
        if cache_key in EncoderFactory._ollama_cache:
            logger.debug("[CACHE HIT] Reusing Ollama encoder '%s'", encoder_name)
            return EncoderFactory._ollama_cache[cache_key]

        logger.info("[CACHE MISS] Loading Ollama encoder '%s'", encoder_name)
        client_kwargs = client_kwargs if client_kwargs is not None else {"verify": False}
        encoder = OllamaEmbeddings(
            model=encoder_name,
            base_url=base_url,
            client_kwargs=client_kwargs,
        )
        EncoderFactory._evict_if_needed(EncoderFactory._ollama_cache)
        EncoderFactory._ollama_cache[cache_key] = encoder
        return encoder

    @staticmethod
    def get_hugging_face_embedding(encoder_name: str, batch_size: Optional[int] = 32) -> HuggingFaceEmbeddings:
        """Get a HuggingFace encoder, returning a cached instance if available.

        :param encoder_name: Name of the encoder to build
        :type encoder_name: str
        :param embed_batch_size: Batch size to use for the encoder, defaults to 32.
        :type embed_batch_size: Optional[int], optional
        :return: The built HuggingFace encoder.
        :rtype: HuggingFaceEmbeddings
        """
        cache_key = (encoder_name, batch_size)
        if cache_key in EncoderFactory._hf_cache:
            logger.debug("[CACHE HIT] Reusing HuggingFace encoder '%s'", encoder_name)
            return EncoderFactory._hf_cache[cache_key]

        logger.info("[CACHE MISS] Loading HuggingFace encoder '%s'", encoder_name)
        encoder = HuggingFaceEmbeddings(model_name=encoder_name, encode_kwargs={"batch_size": batch_size})
        EncoderFactory._evict_if_needed(EncoderFactory._hf_cache)
        EncoderFactory._hf_cache[cache_key] = encoder
        return encoder

    @staticmethod
    def _evict_if_needed(cache: dict) -> None:
        """Evict the oldest entry from *cache* when ``ENCODER_CACHE_MAX_SIZE`` is exceeded."""
        if ENCODER_CACHE_MAX_SIZE > 0 and len(cache) >= ENCODER_CACHE_MAX_SIZE:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key)
            logger.info("[CACHE EVICT] Removed oldest encoder entry (max=%d)", ENCODER_CACHE_MAX_SIZE)

    @staticmethod
    def clear_cache() -> None:
        """Clear all cached encoder instances."""
        EncoderFactory._hf_cache.clear()
        EncoderFactory._ollama_cache.clear()
        logger.info("[CACHE] Langchain encoder cache cleared")
