"""Define a BM25 factory to be used in the langchain retriever base factory."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from opensearchpy import OpenSearch

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun

OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "")
OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "")


class OpenSearchBM25Retriever(BaseRetriever):
    """`OpenSearch` retriever that uses `BM25`."""

    client: Any
    index_name: str

    @classmethod
    def _build_index_config(cls, k1: float, b: float, similarity: str) -> dict:
        """Build the OpenSearch index settings and mappings.

        :param k1: The BM25 parameter k1.
        :param b: The BM25 parameter b.
        :param similarity: The name of the similarity method.
        :return: A dictionary with 'settings' and 'mappings'.
        """
        settings = {
            "analysis": {
                "analyzer": {
                    "default": {
                        "type": "standard",
                    },
                },
            },
            "similarity": {
                similarity: {
                    "type": "BM25",
                    "k1": k1,
                    "b": b,
                },
            },
        }

        mappings = {
            "properties": {
                "content": {
                    "type": "text",
                    "similarity": similarity,
                },
            },
        }

        return {"settings": settings, "mappings": mappings}

    @classmethod
    def create(
        cls,
        opensearch_url: str,
        index_name: str,
        k1: float = 2.0,
        b: float = 0.75,
        similarity: str = "custom_bm25",
    ) -> OpenSearchBM25Retriever:
        """Create an OpenSearchBM25Retriever from a list of texts."""
        client = OpenSearch(
            hosts=[opensearch_url],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        if not client.indices.exists(index=index_name):
            index_config = cls._build_index_config(k1, b, similarity)
            client.indices.create(
                index=index_name,
                body=index_config,
            )

        return cls(client=client, index_name=index_name)

    def add_texts(
        self,
        texts: list[Document],
    ) -> OpenSearch:
        """Add text to teh BM25 store.

        :param texts: The list of documents to add to the retriever.
        :type texts: list[Document]
        :return: The client.
        :rtype: OpenSearch
        """
        for i, doc in enumerate(texts):
            self.client.index(
                index=self.index_name,
                id=i,
                body={"content": doc.page_content, **(doc.metadata or {})},
            )
        self.client.indices.refresh(index=self.index_name)
        return self.client

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Query OpenSearch and retrieve relevant documents.

        :param query: The input query.
        :type query: str
        :param run_manager:
        :type run_manager: CallbackManagerForRetrieverRun
        :return: The list of the relevant documents.
        :rtype: list[Document]
        :raise a Value Error if the document has no content.
        """
        query_dict = {
            "query": {
                "match": {
                    "content": query,
                },
            },
        }
        res = self.client.search(index=self.index_name, body=query_dict)
        if run_manager:
            run_manager.on_text(f"Found {len(res)} results", verbose=True)
        docs = []
        for r in res["hits"]["hits"]:
            source_data = r["_source"]
            content = source_data.get("content", None)
            if content is None:
                error_msg = f"Document in index '{self.index_name}' has no 'content' field: {r}"
                raise ValueError(error_msg)
            metadata = source_data.get("metadata", {})
            docs.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                ),
            )
        return docs
