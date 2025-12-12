"""Defines a node post-processor builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from llama_index.core.postprocessor import (
    LLMRerank,
    SentenceEmbeddingOptimizer,
    SentenceTransformerRerank,
    SimilarityPostprocessor,
)

from rago.model.constructors.llama_index.encoder_factory import EncoderFactory
from rago.model.constructors.llama_index.llm_factory import LLamaIndexLLMFactory

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.llms import LLM
    from llama_index.core.postprocessor.types import BaseNodePostprocessor

    from rago.model.configs.post_processor_config.llama_index import NodePostProcessorConfig


class NodePostProcessorFactory:
    """A post-processor builder."""

    @staticmethod
    def make(
        config: NodePostProcessorConfig,
        encoder: Optional[BaseEmbedding] = None,
        llm: Optional[LLM] = None,
    ) -> BaseNodePostprocessor:
        """Build a llama-index post-processor.

        :param config: The configuration of the post-processor to build.
        :type config: NodePostProcessorConfig
        :param encoder: The encoder used by the post-processor to build.
        :type encoder: BaseEmbedding
        :param llm: The LLM used by the post-processor to build.
        :type llm: LLM
        :return: The built post-processor.
        :rtype: BaseNodePostprocessor
        """
        match config.post_processor_type:
            case "similarity_post_processor":
                return NodePostProcessorFactory.make_similarity_post_processor(config)
            case "sentence_embedding_optimizer":
                return NodePostProcessorFactory.make_sentence_embedding_optimizer(config, encoder)
            case "llm_rerank":
                return NodePostProcessorFactory.make_llm_rerank(config, llm)
            case "sentence_transformers_rerank":
                return NodePostProcessorFactory.make_sentence_transformers_rerank(config)
            case _:
                raise ValueError(config.post_processor_type)

    @staticmethod
    def make_sentence_transformers_rerank(config: NodePostProcessorConfig) -> SentenceTransformerRerank:
        """Make a sentence transformer rerank post-processor.

        :param config: The configuration of the post-processor to build.
        :type config: NodePostProcessorConfig
        :raises ValueError: The top_n parameter is not set in the config.
        :raises ValueError: The reranker_name parameter is not set in the config.
        :return: The created post-processor.
        :rtype: SentenceTransformerRerank
        """
        if config.top_n is None:
            raise ValueError(config.top_n)
        if config.reranker_name is None:
            raise ValueError(config.reranker_name)
        return SentenceTransformerRerank(
            top_n=config.top_n,
            model=config.reranker_name,
            keep_retrieval_score=True,
        )

    @staticmethod
    def make_llm_rerank(
        config: NodePostProcessorConfig,
        llm: Optional[LLM] = None,
    ) -> LLMRerank:
        """Make a llm rerank post-processor.

        :param config: The configuration parameters of the post-processor to create.
        :type config: NodePostProcessorConfig
        :param llm: The llm optionally used by the post-processor, defaults to None
        :type llm: Optional[LLM], optional
        :raises ValueError: The llm is None and the config does not contains a llm configuration
        :raises ValueError: The batch size is not set in the config.
        :raises ValueError: The top_n parameter is not set in the config.
        :return: The created llm rerank post-processor.
        :rtype: _type_
        """
        if llm is None:
            if config.llm is None:
                raise ValueError(config.llm)
            llm = LLamaIndexLLMFactory.make(config.llm)

        if config.choice_batch_size is None:
            raise ValueError(config.choice_batch_size)

        if config.top_n is None:
            raise ValueError(config.top_n)

        return LLMRerank(
            llm=llm,
            choice_batch_size=config.choice_batch_size,
            top_n=config.top_n,
        )

    @staticmethod
    def make_similarity_post_processor(config: NodePostProcessorConfig) -> SimilarityPostprocessor:
        """Make a similarity post-processor.

        :param config: The configuration parameters of the similarity post-processor to make.
        :type config: NodePostProcessorConfig
        :raises ValueError: if the config does not contains a similarity cutoff param.
        :return: The create similarity post-processor.
        :rtype: SimilarityPostprocessor
        """
        if config.similarity_cutoff is None:
            raise ValueError
        return SimilarityPostprocessor(
            similarity_cutoff=config.similarity_cutoff,
        )

    @staticmethod
    def make_sentence_embedding_optimizer(
        config: NodePostProcessorConfig,
        encoder: Optional[BaseEmbedding] = None,
    ) -> SentenceEmbeddingOptimizer:
        """Make a sentence embedding optimizer.

        :param config: The configuration parameters of the post-processor to create.
        :type config: NodePostProcessorConfig
        :param encoder: The encoder used by the post-processor.
        :type encoder: BaseEmbedding
        :raises ValueError: If percentile_cutoff is not set in the config.
        :raises ValueError: If threshold_cutoff is not set in the config.
        :return: The created sentence embedding optimizer.
        :rtype: SentenceEmbeddingOptimizer
        """
        if encoder is None:
            if config.encoder is None:
                raise ValueError(config.encoder)
            encoder = EncoderFactory.make(config.encoder)
        if config.percentile_cutoff is None:
            raise ValueError(config.percentile_cutoff)
        if config.threshold_cutoff is None:
            raise ValueError(config.threshold_cutoff)
        return SentenceEmbeddingOptimizer(
            embed_model=encoder,
            percentile_cutoff=config.percentile_cutoff,
            threshold_cutoff=config.threshold_cutoff,
        )
