"""Define the RAG used to answer queries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic.dataclasses import dataclass

from rago.data_objects import RAGOutput
from rago.model.configs.reader_config.base import ReaderConfig  # noqa: TC001
from rago.model.configs.retriever_config.base import RetrieverConfig  # noqa: TC001
from rago.model.configs.retriever_config.qdrant import QdrantRetrieverConfig
from rago.model.configs.tunable_model_config import TunableModelConfig
from rago.model.wrapper.reader.reader_wrapper_factory import ReaderWrapperFactory
from rago.model.wrapper.retriever.retriever_wrapper_factory import RetrieverWrapperFactory
from rago.model.wrapper.tunable_model import TunableModel
from rago.prompts import PromptConfig

if TYPE_CHECKING:
    from rago.model.wrapper.reader.base import Reader
    from rago.model.wrapper.retriever.base import Retriever


@dataclass
class RAGConfig(TunableModelConfig):
    """Configuration parameters of the RAG.

    Both ``reader`` and ``retriever`` are required.  For retriever-only or
    reader-only optimisation use :class:`RetrieverModelConfig` or
    :class:`ReaderModelConfig` instead.
    """

    reader: ReaderConfig
    retriever: RetrieverConfig


class RAG(TunableModel):
    """A RAG answers queries based on its parametric (llm params) and non-parametric (database) knowledge."""

    def __init__(self, reader: Reader, retriever: Retriever) -> None:
        """Instantiate a RAG from its reader and retriever.

        :param reader: The reader used by the rag to generate answer.
        :type reader: Reader
        :param retriever: The retriever used by the rag to query its non-parametric knowledge.
        :type retriever: Retriever
        """
        self.reader = reader
        self.retriever = retriever

    @classmethod
    def make(
        cls,
        rag_config: RAGConfig,
        prompt_config: Optional[PromptConfig] = None,
        inputs_chunks: Optional[list[str]] = None,
    ) -> RAG:
        """Build a RAG instance from its configuration parameters.

        :param rag_config: The configuration of the rag (reader + retriever).
        :type rag_config: RAGConfig
        :param prompt_config: The configurations params of the reader's prompt template, defaults to None.
        :type prompt_config: Optional[PromptConfig], optional
        :param inputs_chunks: The chunks used by the retriever if any, defaults to None.
        :type inputs_chunks: Optional[list[str]], optional
        :return: The rag instance corresponding to the input config.
        :rtype: RAG
        """
        if prompt_config is None:
            prompt_config = PromptConfig()
        reader = ReaderWrapperFactory.make(
            rag_config.reader,
            prompt_config=prompt_config,
        )

        # Qdrant retrievers query an external collection; no local chunks needed.
        if not isinstance(rag_config.retriever, QdrantRetrieverConfig) and inputs_chunks is None:
            raise ValueError(inputs_chunks)
        retriever = RetrieverWrapperFactory.make(
            config=rag_config.retriever,
            input_chunks=inputs_chunks or [],
        )

        return cls(reader, retriever)

    def get_output(self, query: str) -> RAGOutput:
        """Get the rag response to a query.

        :param query: The query the rag needs to answer.
        :type query: str
        :return: The rag's response to the input query.
        :rtype: RAGOutput
        """
        retrieved_context = self.retriever.get_retriever_output(query)
        answer = self.reader.get_reader_output(query, retrieved_context)
        return RAGOutput(answer=answer, retrieved_context=retrieved_context)

    def get_rag_output(self, query: str) -> RAGOutput:
        """Return output via :meth:`get_output`."""
        return self.get_output(query)
