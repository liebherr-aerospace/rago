"""Define a hybrid retriever combining multiple retrievers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rago.data_objects import RetrievedContext
from rago.model.wrapper.context_post_processor.llama_index_post_processor import LLamaIndexContextPostProcessorWrapper
from rago.model.wrapper.retriever.base import Retriever
from rago.model.wrapper.retriever.langchain_retriever import LangchainRetrieverWrapper

if TYPE_CHECKING:
    from rago.model.configs.retriever_config.langchain import LangchainRetrieverConfig


class HybridLangchainRetrieverWrapper(Retriever):
    """A hybrid retriever that combines an arbitrary number of sub-retrievers with weighted scoring.

    Each sub-retriever's results are independently normalized then multiplied by
    its associated weight before being merged.
    """

    def __init__(
        self,
        retrievers: list[tuple[Retriever, float]],
        nodes_post_processors: Optional[LLamaIndexContextPostProcessorWrapper] = None,
    ) -> None:
        """Instantiate a hybrid retriever.

        :param retrievers: A list of ``(retriever, weight)`` pairs.
            Weights do not need to sum to 1 — they are applied as-is.
        :type retrievers: list[tuple[Retriever, float]]
        :param nodes_post_processors: The context post-processors, defaults to None.
        :type nodes_post_processors: Optional[LLamaIndexContextPostProcessorWrapper], optional
        """
        self.retrievers = retrievers
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
        if config.hybrid_weight is None:
            error_msg = "HybridRetriever requires a hybrid_weight."
            raise ValueError(error_msg)

        retrievers: list[tuple[Retriever, float]] = []

        if config.vector_config is not None:
            vector_retriever = LangchainRetrieverWrapper.make(config.vector_config, inputs_chunks)
            retrievers.append((vector_retriever, config.hybrid_weight))

        if config.bm25_config is not None:
            bm25_retriever = LangchainRetrieverWrapper.make(config.bm25_config, inputs_chunks)
            retrievers.append((bm25_retriever, 1.0 - config.hybrid_weight))

        if not retrievers:
            error_msg = "HybridRetriever requires at least one sub-retriever config (vector_config or bm25_config)."
            raise ValueError(error_msg)

        nodes_post_processors = None
        if config.node_post_processor_config is not None:
            nodes_post_processors = LLamaIndexContextPostProcessorWrapper.make(
                config.node_post_processor_config,
            )

        return cls(
            retrievers=retrievers,
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
        """Get the texts relevant to the query by combining all sub-retriever results.

        Each sub-retriever's scores are normalized to [0, 1], then multiplied by
        the associated weight.  Duplicate contexts (same text) are merged by
        summing their weighted scores.

        :param query: The query.
        :type query: str
        :return: The texts relevant to the query, sorted by combined score descending.
        :rtype: list[RetrievedContext]
        """
        combined: dict[str, float] = {}
        embeddings: dict[str, list[float] | None] = {}

        for retriever, weight in self.retrievers:
            results = self._normalize_scores(retriever.get_retriever_output(query))
            for ctx in results:
                score = (ctx.score or 0.0) * weight
                combined[ctx.text] = combined.get(ctx.text, 0.0) + score
                if ctx.embedding is not None and ctx.text not in embeddings:
                    embeddings[ctx.text] = ctx.embedding

        retrieved_contexts = [
            RetrievedContext(text=text, score=score, embedding=embeddings.get(text))
            for text, score in sorted(combined.items(), key=lambda x: x[1], reverse=True)
        ]

        if self.nodes_post_processors is not None:
            retrieved_contexts = self.nodes_post_processors.post_process_context(query, retrieved_contexts)

        return retrieved_contexts
