"""Defines the post processor to filter and augment retrieved nodes."""

from abc import ABC, abstractmethod

from rago.data_objects.retrieved_context import RetrievedContext


class ContextPostProcessor(ABC):
    """A set of context post-processor that filter and/or augment extracted information based on a user query."""

    @abstractmethod
    def post_process_context(self, query: str, context: list[RetrievedContext]) -> list[RetrievedContext]:
        """Post-process (i.e filter or augment) a list of retrieved context based on their query.

        :param context: The context to post-process.
        :type context: list[RetrievedContext]
        :param query: The query the retrieved context is for.
        :type query: str
        :return: A refined retrieved context for the query.
        :rtype: list[RetrievedContext]
        """
        ...
