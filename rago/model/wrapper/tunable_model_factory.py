"""Define the factory building any TunableModel from its configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rago.model.configs.reader_config.base import ReaderConfig
from rago.model.configs.retriever_config.base import RetrieverConfig
from rago.model.wrapper.rag.base import RAG, RAGConfig
from rago.model.wrapper.reader.reader_wrapper_factory import ReaderWrapperFactory
from rago.model.wrapper.retriever.retriever_wrapper_factory import RetrieverWrapperFactory
from rago.prompts import PromptConfig

if TYPE_CHECKING:
    from rago.model.configs.tunable_model_config import TunableModelConfig
    from rago.model.wrapper.tunable_model import TunableModel


def make_tunable_model(
    config: TunableModelConfig,
    prompt_config: Optional[PromptConfig] = None,
    inputs_chunks: Optional[list[str]] = None,
) -> TunableModel:
    """Build a RAG, a retriever or a reader depending on the config type.

    :param config: The model configuration.
    :type config: TunableModelConfig
    :param prompt_config: Prompt configuration of the reader, defaults to ``PromptConfig()``.
    :type prompt_config: Optional[PromptConfig]
    :param inputs_chunks: Document chunks used by the retriever if any, defaults to None.
    :type inputs_chunks: Optional[list[str]]
    :raises TypeError: If the config type is not supported.
    :return: The built model.
    :rtype: TunableModel
    """
    if prompt_config is None:
        prompt_config = PromptConfig()
    match config:
        case RAGConfig():
            return RAG.make(rag_config=config, prompt_config=prompt_config, inputs_chunks=inputs_chunks)
        case RetrieverConfig():
            return RetrieverWrapperFactory.make(config=config, input_chunks=inputs_chunks)
        case ReaderConfig():
            return ReaderWrapperFactory.make(config, prompt_config=prompt_config)
        case _:
            msg = f"Unsupported config type: {type(config)}"
            raise TypeError(msg)
