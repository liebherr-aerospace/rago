"""Define the TunableModel abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Self

from rago.model.configs.tunable_model_config import TunableModelConfig
from rago.prompts import PromptConfig

if TYPE_CHECKING:
    from rago.data_objects import RAGOutput


@dataclass
class BuildContext:
    """Runtime resources shared by the models built from a config."""

    prompt_config: PromptConfig = field(default_factory=PromptConfig)
    inputs_chunks: Optional[list[str]] = None


class TunableModel[ConfigT: TunableModelConfig](ABC):
    """Abstract base class for any model that can be optimised by the Optuna manager.

    Subclasses include full RAG pipelines, retriever-only models and
    reader-only models.  The optimiser interacts exclusively through the
    :meth:`get_output` interface so it is agnostic of the concrete topology.
    """

    @classmethod
    @abstractmethod
    def make(cls, config: ConfigT, ctx: Optional[BuildContext] = None) -> Self:
        """Build the model from its configuration.

        :param config: The model configuration.
        :type config: ConfigT
        :param ctx: Runtime resources (prompt config, input chunks), defaults to ``BuildContext()``.
        :type ctx: Optional[BuildContext]
        :return: The built model.
        :rtype: Self
        """
        ...

    @abstractmethod
    def get_output(self, query: str) -> RAGOutput:
        """Produce an output for the given *query*.

        :param query: The user query.
        :type query: str
        :return: The model output (answer and/or retrieved context).
        :rtype: RAGOutput
        """
        ...
