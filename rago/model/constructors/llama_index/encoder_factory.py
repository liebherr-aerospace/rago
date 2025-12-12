"""Defines a factory of LlamaIndex encoders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rago.model.configs.encoder_config import (
    HuggingFaceLlaIndexEncoderConfig,
    LlamaIndexEncoderConfig,
    OllamaLlamaIndexEncoderConfig,
)

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding


class EncoderFactory:
    """A encoder Factory to build llama-index encoders."""

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
        """Get hugging face encoder from embedding and batch size.

        :param encoder_name: Name of the encoder to build.
        :type encoder_name: str
        :param embed_batch_size: Batch size to use for the encoder, defaults to 32.
        :type embed_batch_size: Optional[int], optional
        :return: The built HuggingFace encoder.
        :rtype: HuggingFaceEmbedding
        """
        return HuggingFaceEmbedding(model_name=encoder_name, embed_batch_size=embed_batch_size)
