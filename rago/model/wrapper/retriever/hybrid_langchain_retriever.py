"""Define a hybrid retriever combining vector and BM25 retrievers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rago.data_objects import RetrievedContext
from rago.model.wrapper.context_post_processor.llama_index_post_processor import LLamaIndexContextPostProcessorWrapper
from rago.model.wrapper.retriever.base import Retriever
from rago.model.wrapper.retriever.langchain_retriever import LangchainRetrieverWrapper

if TYPE_CHECKING:
    from rago.model.configs.retriever_config.langchain import LangchainRetrieverConfig


class HybridLangchainRetrieverWrapper(Retriever):
    """A hybrid retriever that combines VectorIndex and BM25 retrievers with weighted scoring.

    The hybrid retriever runs both sub-retrievers independently, normalizes their scores,
    and combines them using a configurable weight parameter.
    """

    def __init__(
        self,
        vector_retriever: LangchainRetrieverWrapper,
        bm25_retriever: LangchainRetrieverWrapper,
        hybrid_weight: float = 0.5,
        nodes_post_processors: Optional[LLamaIndexContextPostProcessorWrapper] = None,
    ) -> None:
        """Instantiate a hybrid retriever.

        :param vector_retriever: The vector index retriever.
        :type vector_retriever: LangchainRetrieverWrapper
        :param bm25_retriever: The BM25 retriever.
        :type bm25_retriever: LangchainRetrieverWrapper
        :param hybrid_weight: Weight for the vector retriever score (1 - weight for BM25), defaults to 0.5.
        :type hybrid_weight: float
        :param nodes_post_processors: The context post-processors, defaults to None.
        :type nodes_post_processors: Optional[LLamaIndexContextPostProcessorWrapper], optional
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.hybrid_weight = hybrid_weight
        self.nodes_post_processors = nodes_post_processors

    @classmethod
    def make(
        cls,
        config: LangchainRetrieverConfig,
        inputs_chunks: list[str],
    ) -> HybridLangchainRetrieverWrapper:
        """Generate a hybrid retriever from a config and document chunks.

        :param config: The config of the hybrid retriever to generate.
        :type config: LangchainRetrieverConfig
        :param inputs_chunks: The document chunks used by the retrievers.
        :type inputs_chunks: list[str]
        :return: The hybrid retriever.
        :rtype: HybridLangchainRetrieverWrapper
        """
        if config.vector_config is None:
            error_msg = "HybridRetriever requires a vector_config."
            raise ValueError(error_msg)
        if config.bm25_config is None:
            error_msg = "HybridRetriever requires a bm25_config."
            raise ValueError(error_msg)
        if config.hybrid_weight is None:
            error_msg = "HybridRetriever requires a hybrid_weight."
            raise ValueError(error_msg)

        vector_retriever = LangchainRetrieverWrapper.make(config.vector_config, inputs_chunks)
        bm25_retriever = LangchainRetrieverWrapper.make(config.bm25_config, inputs_chunks)

        nodes_post_processors = None
        if config.node_post_processor_config is not None:
            nodes_post_processors = LLamaIndexContextPostProcessorWrapper.make(
                config.node_post_processor_config,
            )

        return cls(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            hybrid_weight=config.hybrid_weight,
            nodes_post_processors=nodes_post_processors,
        )

    @staticmethod
    def _normalize_scores(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """Normalize scores of retrieved contexts to [0, 1] range using min-max normalization.

        :param contexts: The retrieved contexts with raw scores.
        :type contexts: list[RetrievedContext]
        :return: The contexts with normalized scores.
        :rtype: list[RetrievedContext]
        """
        scores = [ctx.score for ctx in contexts if ctx.score is not None]
        if not scores:
            return contexts

        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        normalized = []
        for ctx in contexts:
            if ctx.score is not None and score_range > 0:
                normalized_score = (ctx.score - min_score) / score_range
            elif ctx.score is not None:
                normalized_score = 1.0
            else:
                normalized_score = 0.0
            normalized.append(
                RetrievedContext(text=ctx.text, embedding=ctx.embedding, score=normalized_score),
            )
        return normalized

    def get_retriever_output(self, query: str) -> list[RetrievedContext]:
        """Get the texts relevant to the query by combining vector and BM25 retriever results.

        Results are combined using weighted scoring: hybrid_weight * vector_score + (1 - hybrid_weight) * bm25_score.
        Duplicate contexts (same text) are merged by taking the combined score.

        :param query: The query.
        :type query: str
        :return: The texts relevant to the query, sorted by combined score descending.
        :rtype: list[RetrievedContext]
        """
        vector_results = self._normalize_scores(self.vector_retriever.get_retriever_output(query))
        bm25_results = self._normalize_scores(self.bm25_retriever.get_retriever_output(query))

        # Combine results using weighted scores, merging duplicates
        combined: dict[str, float] = {}
        embeddings: dict[str, list[float] | None] = {}

        for ctx in vector_results:
            score = (ctx.score or 0.0) * self.hybrid_weight
            combined[ctx.text] = combined.get(ctx.text, 0.0) + score
            if ctx.embedding is not None:
                embeddings[ctx.text] = ctx.embedding

        for ctx in bm25_results:
            score = (ctx.score or 0.0) * (1.0 - self.hybrid_weight)
            combined[ctx.text] = combined.get(ctx.text, 0.0) + score
            if ctx.text not in embeddings and ctx.embedding is not None:
                embeddings[ctx.text] = ctx.embedding

        # Build sorted results
        retrieved_contexts = [
            RetrievedContext(text=text, score=score, embedding=embeddings.get(text))
            for text, score in sorted(combined.items(), key=lambda x: x[1], reverse=True)
        ]

        if self.nodes_post_processors is not None:
            retrieved_contexts = self.nodes_post_processors.post_process_context(query, retrieved_contexts)

        return retrieved_contexts
