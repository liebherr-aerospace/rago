"""Define an abstract reader class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rago.data_objects import RetrievedContext


class Retriever(ABC):
    """A retriever that retrieves information relevant to answer a query from a database (non-parametric knowledge)."""

    @abstractmethod
    def get_retriever_output(self, query: str) -> list[RetrievedContext]:
        """Get the information relevant to answer a query from the retriever's database.

        :param query: The query.
        :type query: str
        :return: The list of texts extracted of the retriever non-parametric knowledge.
        :rtype: list[str]
        """
        ...
