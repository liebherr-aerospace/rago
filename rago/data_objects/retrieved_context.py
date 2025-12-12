"""Define the retrieved context Data type classes."""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document as LangchainDocument
from llama_index.core.schema import NodeWithScore, TextNode
from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class RetrievedContext:
    """Context retrieved due to its relevancy to a given query.

    :param text: chunk of information (e.g a document, a sentence or paragraph.)
    :type text: str
    :param embedding: vector representation of the text, defaults to None.
    :type embedding: Optional[list[float]]
    :param score: Characterizes the relevancy of the context to its query, defaults to None.
    :type score: float
    """

    text: str = Field(..., min_length=1, strict=True)
    embedding: Optional[list[float]] = Field(default=None, min_length=1, strict=True)
    score: Optional[float] = Field(default=None, strict=True)

    def get_llama_index_node(self) -> TextNode:
        """Get the node in a LLamaIndex format.

        :return: the llama_index node.
        :rtype: TextNode
        """
        return TextNode(text=self.text)

    def get_llama_index_with_score(self) -> NodeWithScore:
        """Get the node with score in a LLamaIndex format.

        :return: the llama_index node.
        :rtype: TextNode
        """
        node = self.get_llama_index_node()
        return NodeWithScore(node=node, score=self.score)

    def get_langchain_node(self) -> LangchainDocument:
        """Get the node in a LlamaIndex format.

        :return: the langchain node.
        :rtype: Document
        """
        return LangchainDocument(self.text, metadata={"score": self.score})

    @classmethod
    def from_llama_index_node(cls, node: TextNode) -> RetrievedContext:
        """Create a Node from a LlamaIndex Node.

        :param node:
        """
        return cls(node.text)

    @classmethod
    def from_llama_index_node_with_score(cls, node_with_score: NodeWithScore) -> RetrievedContext:
        """Create a Node from a LlamaIndex Node.

        :param node:
        """
        return cls(node_with_score.text, score=node_with_score.score, embedding=node_with_score.embedding)

    @classmethod
    def from_langchain_node(cls, node: LangchainDocument) -> RetrievedContext:
        """Create a Node from a Langchain Node.

        :param node:
        """
        return cls(node.page_content)

    @classmethod
    def from_langchain_node_with_score(cls, node: LangchainDocument) -> RetrievedContext:
        """Create a Node from a Langchain Node.

        :param node:
        """
        metadata_score = node.metadata.get("score")
        score = metadata_score if metadata_score and isinstance(metadata_score, float) else None
        return cls(node.page_content, score=score)
