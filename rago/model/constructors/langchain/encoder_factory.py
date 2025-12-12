"""Defines a factory of Langchain encoders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

from rago.model.configs.encoder_config import (
    HuggingFaceLangchainEncoderConfig,
    LangchainEncoderConfig,
    OllamaLangchainEncoderConfig,
)

if TYPE_CHECKING:
    from langchain.embeddings.base import Embeddings


class EncoderFactory:
    """An encoder factory to make Langchain encoders."""

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
        """Get hugging face encoder from name and base_url and client_kwargs.

        :param encoder_name: Name of the encoder to build
        :type encoder_name: str
        :param base_url: Url of the Ollama server.
        :type base_url: str
        :param client_kwargs: The kwargs used by the ollama client, defaults to None
        :type client_kwargs: Optional[dict[str, Any]], optional
        :return: The built ollama embedding.
        :rtype: OllamaEmbeddings
        """
        client_kwargs = client_kwargs if client_kwargs is not None else {"verify": False}
        return OllamaEmbeddings(
            model=encoder_name,
            base_url=base_url,
            client_kwargs=client_kwargs,
        )

    @staticmethod
    def get_hugging_face_embedding(encoder_name: str, batch_size: Optional[int] = 32) -> HuggingFaceEmbeddings:
        """Get hugging face encoder from name and batch size.

        :param encoder_name: Name of the encoder to build
        :type encoder_name: str
        :param embed_batch_size: Batch size to use for the encoder, defaults to 32.
        :type embed_batch_size: Optional[int], optional
        :return: The built HuggingFace encoder.
        :rtype: HuggingFaceEmbeddings
        """
        return HuggingFaceEmbeddings(model_name=encoder_name, encode_kwargs={"batch_size": batch_size})
