"""Define a reader-only TunableModel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic.dataclasses import dataclass

from rago.data_objects import RAGOutput
from rago.model.configs.reader_config.base import ReaderConfig  # noqa: TC001
from rago.model.configs.tunable_model_config import TunableModelConfig
from rago.model.wrapper.reader.reader_wrapper_factory import ReaderWrapperFactory
from rago.model.wrapper.tunable_model import TunableModel
from rago.prompts import PromptConfig

if TYPE_CHECKING:
    from rago.model.wrapper.reader.base import Reader


@dataclass
class ReaderModelConfig(TunableModelConfig):
    """Configuration for a reader-only :class:`TunableModel`."""

    reader: ReaderConfig


class ReaderModel(TunableModel):
    """A :class:`TunableModel` that only performs reading / generation (no retriever)."""

    def __init__(self, reader: Reader) -> None:
        """Instantiate a reader-only model.

        :param reader: The reader (LLM) instance.
        :type reader: Reader
        """
        self.reader = reader

    @classmethod
    def make(
        cls,
        config: ReaderModelConfig,
        prompt_config: Optional[PromptConfig] = None,
    ) -> ReaderModel:
        """Build a :class:`ReaderModel` from its configuration.

        :param config: Reader model configuration.
        :type config: ReaderModelConfig
        :param prompt_config: Prompt template configuration, defaults to ``PromptConfig()``.
        :type prompt_config: Optional[PromptConfig]
        :return: The built reader model.
        :rtype: ReaderModel
        """
        if prompt_config is None:
            prompt_config = PromptConfig()
        reader = ReaderWrapperFactory.make(config.reader, prompt_config=prompt_config)
        return cls(reader)

    def get_output(self, query: str) -> RAGOutput:
        """Generate an answer to *query* without retrieval context.

        :param query: The user query.
        :type query: str
        :return: Output with populated ``answer`` and ``retrieved_context=None``.
        :rtype: RAGOutput
        """
        answer = self.reader.get_reader_output(query)
        return RAGOutput(answer=answer, retrieved_context=None)
