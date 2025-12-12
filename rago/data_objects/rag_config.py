"""Define the classes holding the RAG configuration from the optimization results."""

from __future__ import annotations

from typing import Union


class RAGParam:
    """Class that contains the RAG parameter name and its values."""

    def __init__(self, name: str, value: Union[float, str]) -> None:
        """Initialize the RAG Parameter object."""
        self.name = name
        self.value = value

    def __repr__(self) -> str:
        """Return a string representation of the RAGParam instance.

        :return: A string in the format 'RAGParam(name=<name>, value=<value>)'.
        :rtype: str
        """
        return f"RAGParam(name={self.name}, value={self.value})"


class RAGConfig:
    """Class that contains the RAG COnfiguration that is the set of the RAG parameters."""

    def __init__(self, params: set[RAGParam]) -> None:
        """Initialize the RAG Configuration object."""
        self.params = params

    def __repr__(self) -> str:
        """Return a string representation of the RAGConfig instance.

        :return: a string representation of the RAGConfig instance.
        :rtype: str
        """
        return f"RAGConfig(params={self.params})"
