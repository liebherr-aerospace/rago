"""Define an abstract reader class."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from rago.data_objects import RAGOutput
from rago.model.wrapper.tunable_model import TunableModel

if TYPE_CHECKING:
    from rago.data_objects import RetrievedContext


class Retriever(TunableModel):
    """A retriever that retrieves information relevant to answer a query from a database (non-parametric knowledge)."""

    def get_output(self, query: str) -> RAGOutput:
        """Retrieve the context relevant to *query*, without generating an answer.

        :param query: The query.
        :type query: str
        :return: Output with ``answer=None`` and populated ``retrieved_context``.
        :rtype: RAGOutput
        """
        return RAGOutput(answer=None, retrieved_context=self.get_retriever_output(query))

    @abstractmethod
    def get_retriever_output(self, query: str) -> list[RetrievedContext]:
        """Get the information relevant to answer a query from the retriever's database.

        :param query: The query.
        :type query: str
        :return: The list of texts extracted of the retriever non-parametric knowledge.
        :rtype: list[str]
        """
        ...
