"""Define the RAG used to answer queries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic.dataclasses import dataclass

from rago.data_objects import RAGOutput
from rago.model.configs.base import Config
from rago.model.configs.reader_config.base import ReaderConfig  # noqa: TC001
from rago.model.configs.retriever_config.base import RetrieverConfig  # noqa: TC001
from rago.model.wrapper.reader.reader_wrapper_factory import ReaderWrapperFactory
from rago.model.wrapper.retriever.retriever_wrapper_factory import RetrieverWrapperFactory

if TYPE_CHECKING:
    from rago.model.wrapper.reader.base import Reader
    from rago.model.wrapper.retriever.base import Retriever
    from rago.prompts import PromptConfig


@dataclass
class RAGConfig(Config):
    """Configuration parameters of the RAG."""

    reader: ReaderConfig
    retriever: Optional[RetrieverConfig] = None


class RAG:
    """A RAG answers queries based on its parametric (llm params) and optionally non-parametric (database) knowledge."""

    reader: Optional[Reader] = None
    retriever: Optional[Retriever] = None

    def __init__(self, reader: Optional[Reader] = None, retriever: Optional[Retriever] = None) -> None:
        """Instantiate a RAG from its reader, (retriever Optionally if use_retriever is True).

        :param reader: The reader used by the rag to generate answer.
        :type reader: Reader
        :param retriever: The reader optionally used by the rag to query its parametric knowledge, defaults to None
        :type retriever: Optional[Retriever], optional
        """
        self.reader = reader
        self.retriever = retriever

    @classmethod
    def make(
        cls,
        rag_config: RAGConfig,
        prompt_config: PromptConfig,
        inputs_chunks: Optional[list[str]] = None,
    ) -> RAG:
        """Build a RAG instance from its configuration parameters.

        :param config: The configuration of the rag (i.e configuration of the reader, the retriever and so on)
        :type config: dict
        :param prompt_config: The configurations params of the reader's prompt template.
        :type prompt_config: PromptConfig
        :param inputs_chunks: The chunks used by the retriever if any, defaults to None
        :type inputs_chunks: Optional[list[str]], optional
        :return: The rag instance corresponding to the input config.
        :rtype: RAG
        """
        reader = ReaderWrapperFactory.make(
            rag_config.reader,
            prompt_config=prompt_config,
        )
        if rag_config.retriever is not None:
            if inputs_chunks is None:
                raise ValueError(inputs_chunks)
            retriever = RetrieverWrapperFactory.make(
                config=rag_config.retriever,
                input_chunks=inputs_chunks,
            )
        else:
            retriever = None

        return cls(reader, retriever)

    def get_rag_output(self, query: str) -> RAGOutput:
        """Get the rag response to a query.

        :param query: The query the rag needs to answer.
        :type query: str
        :return: The rag's response to the input query.
        :rtype: RagOutput
        """
        retrieved_context = self.retriever.get_retriever_output(query) if self.retriever is not None else None
        answer = self.reader.get_reader_output(query, retrieved_context) if self.reader is not None else None

        return RAGOutput(answer=answer, retrieved_context=retrieved_context)
