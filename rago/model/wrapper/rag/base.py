"""Define the RAG used to answer queries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic.dataclasses import dataclass

from rago.data_objects import EvalSample, ModelInput, RAGOutput, ReaderInput
from rago.model.configs.reader_config.base import ReaderConfig  # noqa: TC001
from rago.model.configs.retriever_config.base import RetrieverConfig  # noqa: TC001
from rago.model.configs.tunable_model_config import TunableModelConfig
from rago.model.wrapper.reader.reader_wrapper_factory import ReaderWrapperFactory
from rago.model.wrapper.retriever.retriever_wrapper_factory import RetrieverWrapperFactory
from rago.model.wrapper.tunable_model import BuildContext, TunableModel

if TYPE_CHECKING:
    from rago.model.wrapper.reader.base import Reader
    from rago.model.wrapper.retriever.base import Retriever


@dataclass
class RAGConfig(TunableModelConfig):
    """Configuration parameters of the RAG.

    Both ``reader`` and ``retriever`` are required.  For retriever-only or
    reader-only optimisation use a :class:`RetrieverConfig` or
    :class:`ReaderConfig` directly.
    """

    reader: ReaderConfig
    retriever: RetrieverConfig


class RAG(TunableModel[ModelInput, RAGOutput, RAGConfig]):
    """A RAG answers queries based on its parametric (llm params) and non-parametric (database) knowledge."""

    def __init__(self, reader: Reader[ReaderConfig], retriever: Retriever[RetrieverConfig]) -> None:
        """Instantiate a RAG from its reader and retriever.

        :param reader: The reader used by the rag to generate answer.
        :type reader: Reader
        :param retriever: The retriever used by the rag to query its non-parametric knowledge.
        :type retriever: Retriever
        """
        self.reader = reader
        self.retriever = retriever
    @classmethod
    def get_input(cls, eval_sample: EvalSample) -> ModelInput:
        """Extract the RAG input from the eval sample.

        :param eval_sample: The eval sample to extract the input from.
        :type eval_sample: EvalSample
        :return: The extracted RAG input.
        :rtype: ModelInput
        """
        return ModelInput(eval_sample.query)

    @classmethod
    def make(cls, config: RAGConfig, ctx: Optional[BuildContext] = None) -> RAG:
        """Build a RAG instance from its configuration parameters.

        :param config: The configuration of the rag (reader + retriever).
        :type config: RAGConfig
        :param ctx: Runtime resources (prompt config, input chunks), defaults to ``BuildContext()``.
        :type ctx: Optional[BuildContext]
        :return: The rag instance corresponding to the input config.
        :rtype: RAG
        """
        ctx = ctx or BuildContext()
        reader = ReaderWrapperFactory.make(config.reader, ctx)
        retriever = RetrieverWrapperFactory.make(config.retriever, ctx)
        return cls(reader, retriever)

    def _get_output(self, model_input: ModelInput) -> RAGOutput:
        """Get the rag response to a query.

        :param query: The query the rag needs to answer.
        :type query: str
        :return: The rag's response to the input query.
        :rtype: RAGOutput
        """
        retriever_output = self.retriever.get_output(model_input)
        reader_input = ReaderInput(model_input.query, retriever_output.retrieved_context)
        reader_output = self.reader.get_output(reader_input)
        return RAGOutput(answer=reader_output.answer, retrieved_context=retriever_output.retrieved_context)
