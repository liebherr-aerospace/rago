"""Define a retriever-only TunableModel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic.dataclasses import dataclass

from rago.data_objects import RAGOutput
from rago.model.configs.retriever_config.base import RetrieverConfig  # noqa: TC001
from rago.model.configs.retriever_config.qdrant import QdrantRetrieverConfig
from rago.model.configs.tunable_model_config import TunableModelConfig
from rago.model.wrapper.retriever.retriever_wrapper_factory import RetrieverWrapperFactory
from rago.model.wrapper.tunable_model import TunableModel

if TYPE_CHECKING:
    from rago.model.wrapper.retriever.base import Retriever


@dataclass
class RetrieverModelConfig(TunableModelConfig):
    """Configuration for a retriever-only :class:`TunableModel`."""

    retriever: RetrieverConfig


class RetrieverModel(TunableModel):
    """A :class:`TunableModel` that only performs retrieval (no reader / LLM)."""

    def __init__(self, retriever: Retriever) -> None:
        """Instantiate a retriever-only model.

        :param retriever: The retriever instance.
        :type retriever: Retriever
        """
        self.retriever = retriever

    @classmethod
    def make(
        cls,
        config: RetrieverModelConfig,
        inputs_chunks: Optional[list[str]] = None,
    ) -> RetrieverModel:
        """Build a :class:`RetrieverModel` from its configuration.

        :param config: Retriever model configuration.
        :type config: RetrieverModelConfig
        :param inputs_chunks: Document chunks for indexing (not needed for Qdrant).
        :type inputs_chunks: Optional[list[str]]
        :return: The built retriever model.
        :rtype: RetrieverModel
        """
        if not isinstance(config.retriever, QdrantRetrieverConfig) and inputs_chunks is None:
            raise ValueError(inputs_chunks)
        retriever = RetrieverWrapperFactory.make(
            config=config.retriever,
            input_chunks=inputs_chunks or [],
        )
        return cls(retriever)

    def get_output(self, query: str) -> RAGOutput:
        """Retrieve documents relevant to *query*.

        :param query: The user query.
        :type query: str
        :return: Output with ``answer=None`` and populated ``retrieved_context``.
        :rtype: RAGOutput
        """
        retrieved_context = self.retriever.get_retriever_output(query)
        return RAGOutput(answer=None, retrieved_context=retrieved_context)
