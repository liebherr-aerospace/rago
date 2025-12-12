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

    def sample(self, trial: optuna.trial.BaseTrial) -> LangchainRetrieverConfig:
        """Sample a retriever configuration from retriever configuration space."""
        retriever_type_name = self.retriever_type_name.sample(trial)
        if not isinstance(retriever_type_name, str):
            raise TypeError(retriever_type_name)

        if retriever_type_name == "BM25Retriever":
            return LangchainRetrieverConfig(
                type=retriever_type_name,
                search_type=None,
                search_kwargs={
                    "k1": self.bm25_k1.sample(trial),
                    "b": self.bm25_b.sample(trial),
                    "similarity": self.bm25_similarity.sample(trial),
                },
                encoder=None,
                similarity_function=None,
            )

        return LangchainRetrieverConfig(
            type=retriever_type_name,
            similarity_function=self.similarity_function.sample(trial),
            search_type=self.search_type.sample(trial),
            search_kwargs={"k": self.top_k.sample(trial), "score_threshold": self.score_threshold.sample(trial)},
            encoder=self.encoder.sample(trial),
        )
