"""Define an abstract reader class."""

from __future__ import annotations

from abc import abstractmethod

from rago.data_objects import RetrievalOutput
from rago.model.configs.retriever_config.base import RetrieverConfig
from rago.model.wrapper.tunable_model import TunableModel


class Retriever[RetrieverConfigType: RetrieverConfig](TunableModel[RetrievalOutput, RetrieverConfigType]):
    """A retriever that retrieves information relevant to answer a query from a database (non-parametric knowledge)."""

    @abstractmethod
    def _get_output(self, query: str) -> RetrievalOutput:
        """Get the information relevant to answer a query from the retriever's database.

        :param query: The query.
        :type query: str
        :return: The list of texts extracted of the retriever non-parametric knowledge.
        :rtype: list[str]
        """
        ...
