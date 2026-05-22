"""Define retriever config spaces."""

import optuna
from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.model.configs.retriever_config.langchain import LangchainRetrieverConfig
from rago.optimization.search_space.config_space import ConfigSpace
from rago.optimization.search_space.encoder_config_space import HFEncoderConfigSpace
from rago.optimization.search_space.param_space import CategoricalParamSpace, FloatParamSpace, IntParamSpace


@dataclass
class RetrieverConfigSpace(ConfigSpace):
    """Space containing all the possible retriever configurations."""

    retriever_type_name: CategoricalParamSpace = Field(
        default=CategoricalParamSpace(choices=["VectorIndexRetriever"]),
    )
    similarity_function: CategoricalParamSpace = Field(default=CategoricalParamSpace(choices=["cosine"]))
    search_type: CategoricalParamSpace = Field(default=CategoricalParamSpace(choices=["similarity_score_threshold"]))
    top_k: IntParamSpace = Field(default=IntParamSpace(low=1, high=5))
    score_threshold: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=0.9))
    encoder: HFEncoderConfigSpace = Field(default=HFEncoderConfigSpace())

    bm25_k1: FloatParamSpace = Field(default=FloatParamSpace(low=1.2, high=2.0))
    bm25_b: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=1.0))
    bm25_similarity: CategoricalParamSpace = Field(default=CategoricalParamSpace(choices=["custom_bm25"]))

    # Hybrid retriever fields
    hybrid_weight: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=1.0))

    def sample(self, trial: optuna.trial.BaseTrial) -> LangchainRetrieverConfig:
        """Sample a retriever configuration from retriever configuration space."""
        retriever_type_name = self.retriever_type_name.sample(trial)
        if not isinstance(retriever_type_name, str):
            raise TypeError(retriever_type_name)

        if retriever_type_name == "BM25Retriever":
            return self._sample_bm25(trial)

        if retriever_type_name == "HybridRetriever":
            return self._sample_hybrid(trial)

        return self._sample_vector(trial, retriever_type_name)

    def _sample_vector(self, trial: optuna.trial.BaseTrial, retriever_type_name: str) -> LangchainRetrieverConfig:
        """Sample a VectorIndex retriever configuration.

        :param trial: Trial used to sample the configuration.
        :type trial: optuna.trial.BaseTrial
        :param retriever_type_name: The name of the retriever type.
        :type retriever_type_name: str
        :return: The sampled vector retriever config.
        :rtype: LangchainRetrieverConfig
        """
        return LangchainRetrieverConfig(
            type=retriever_type_name,
            similarity_function=self.similarity_function.sample(trial),
            search_type=self.search_type.sample(trial),
            search_kwargs={"k": self.top_k.sample(trial), "score_threshold": self.score_threshold.sample(trial)},
            encoder=self.encoder.sample(trial),
        )

    def _sample_bm25(self, trial: optuna.trial.BaseTrial) -> LangchainRetrieverConfig:
        """Sample a BM25 retriever configuration.

        :param trial: Trial used to sample the configuration.
        :type trial: optuna.trial.BaseTrial
        :return: The sampled BM25 retriever config.
        :rtype: LangchainRetrieverConfig
        """
        return LangchainRetrieverConfig(
            type="BM25Retriever",
            search_type=None,
            search_kwargs={
                "k1": self.bm25_k1.sample(trial),
                "b": self.bm25_b.sample(trial),
                "similarity": self.bm25_similarity.sample(trial),
            },
            encoder=None,
            similarity_function=None,
        )

    def _sample_hybrid(self, trial: optuna.trial.BaseTrial) -> LangchainRetrieverConfig:
        """Sample a Hybrid retriever configuration combining VectorIndex and BM25.

        Optimizes the hybrid weight as well as all parameters of both sub-retrievers.

        :param trial: Trial used to sample the configuration.
        :type trial: optuna.trial.BaseTrial
        :return: The sampled hybrid retriever config.
        :rtype: LangchainRetrieverConfig
        """
        # Sample vector retriever sub-config (all vector params are optimized)
        vector_config = LangchainRetrieverConfig(
            type="VectorIndexRetriever",
            similarity_function=self.similarity_function.sample(trial),
            search_type=self.search_type.sample(trial),
            search_kwargs={"k": self.top_k.sample(trial), "score_threshold": self.score_threshold.sample(trial)},
            encoder=self.encoder.sample(trial),
        )

        # Sample BM25 retriever sub-config (all BM25 params are optimized)
        bm25_config = LangchainRetrieverConfig(
            type="BM25Retriever",
            search_type=None,
            search_kwargs={
                "k1": self.bm25_k1.sample(trial),
                "b": self.bm25_b.sample(trial),
                "similarity": self.bm25_similarity.sample(trial),
            },
            encoder=None,
            similarity_function=None,
        )

        # Sample the hybrid weight between vector and BM25
        hybrid_weight = self.hybrid_weight.sample(trial)

        return LangchainRetrieverConfig(
            type="HybridRetriever",
            vector_config=vector_config,
            bm25_config=bm25_config,
            hybrid_weight=hybrid_weight,
        )
