"""Define the TunableModel abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rago.data_objects import RAGOutput


class TunableModel(ABC):
    """Abstract base class for any model that can be optimised by the Optuna manager.

    Subclasses include full RAG pipelines, retriever-only models and
    reader-only models.  The optimiser interacts exclusively through the
    :meth:`get_output` interface so it is agnostic of the concrete topology.
    """

    @abstractmethod
    def get_output(self, query: str) -> RAGOutput:
        """Produce an output for the given *query*.

        :param query: The user query.
        :type query: str
        :return: The model output (answer and/or retrieved context).
        :rtype: RAGOutput
        """
        ...
