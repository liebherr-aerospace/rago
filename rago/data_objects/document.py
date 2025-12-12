"""Define the document dataclass."""

from __future__ import annotations

import uuid

from langchain_core.documents import Document as LangchainDocument
from llama_index.core import Document as LlamaIndexDocument
from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.data_objects.base import DataObject


@dataclass
class Document(DataObject):
    """A Document containing text.

    In RAGO, it is used by a RAG during training to generate the Index.
    and by the dataset generator to generate the dataset for evaluation.
    The main benefit of this class is that it makes the bridge between llamaIndex and Langchain.

    :param text: The text of the document.
    :type text: str
    """

    text: str = Field(..., min_length=1, strict=True)
    id: str = Field(default_factory=lambda _: str(uuid.uuid4()))

    def get_llama_index_document(self) -> LlamaIndexDocument:
        """Get the document in a llamaIndex format.

        :return: the LlamaIndex Documents.
        :rtype: str
        """
        return LlamaIndexDocument(text=self.text)

    def get_langchain_documents(self) -> LangchainDocument:
        """Get the documents in a Langchain format.

        :return: the Langchain Documents.
        :rtype: str
        """
        return LangchainDocument(page_content=self.text)
