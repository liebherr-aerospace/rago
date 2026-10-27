"""Test that retrieval scores are propagated (vector, BM25 metadata) and used by the hybrid retriever."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from rago.data_objects import RetrievedContext
from rago.model.constructors.langchain.retriever_factory.base_factory import ScoredVectorStoreRetriever
from rago.model.wrapper.retriever.base import Retriever
from rago.model.wrapper.retriever.hybrid_langchain_retriever import HybridLangchainRetrieverWrapper


class _FixedRetriever(Retriever):
    def __init__(self, contexts: list[RetrievedContext]) -> None:
        self.contexts = contexts

    @classmethod
    def make(cls, config: object, ctx: object = None) -> "_FixedRetriever":
        raise NotImplementedError

    def get_retriever_output(self, query: str) -> list[RetrievedContext]:  # noqa: ARG002
        return self.contexts


@pytest.mark.parametrize(("metadata", "expected"), [({"score": 0.0}, 0.0), ({"score": 0.7}, 0.7), ({}, None)])
def test_from_langchain_node_with_score(metadata: dict[str, float], expected: float | None) -> None:
    context = RetrievedContext.from_langchain_node_with_score(Document("text", metadata=metadata))
    assert context.score == expected


@pytest.mark.parametrize("search_type", ["similarity", "similarity_score_threshold"])
def test_scored_vector_store_retriever_keeps_scores(search_type: str) -> None:
    vectorstore = MagicMock(spec=VectorStore)
    vectorstore.similarity_search_with_relevance_scores.return_value = [
        (Document("a", metadata={"source": "x"}), 0.9),
        (Document("b"), 0.4),
    ]
    search_kwargs = {"k": 2, "score_threshold": 0.1} if search_type == "similarity_score_threshold" else {"k": 2}
    retriever = ScoredVectorStoreRetriever(
        vectorstore=vectorstore,
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    docs = retriever.invoke("query")

    vectorstore.similarity_search_with_relevance_scores.assert_called_once_with("query", **search_kwargs)
    assert [doc.metadata["score"] for doc in docs] == [0.9, 0.4]
    assert docs[0].metadata["source"] == "x"


def test_hybrid_weight_changes_ranking() -> None:
    vector = _FixedRetriever([RetrievedContext("a", score=0.9), RetrievedContext("b", score=0.1)])
    bm25 = _FixedRetriever([RetrievedContext("b", score=12.0), RetrievedContext("a", score=2.0)])

    vector_first = HybridLangchainRetrieverWrapper([(vector, 0.8), (bm25, 0.2)]).get_retriever_output("q")
    bm25_first = HybridLangchainRetrieverWrapper([(vector, 0.2), (bm25, 0.8)]).get_retriever_output("q")

    assert [ctx.text for ctx in vector_first] == ["a", "b"]
    assert [ctx.text for ctx in bm25_first] == ["b", "a"]
    assert vector_first[0].score == pytest.approx(0.8)
