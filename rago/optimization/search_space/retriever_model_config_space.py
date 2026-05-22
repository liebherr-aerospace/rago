"""Define the RetrieverModel config space."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from pydantic.dataclasses import dataclass

from rago.model.wrapper.retriever_model import RetrieverModelConfig
from rago.optimization.search_space.retriever_config_space import RetrieverConfigSpace
from rago.optimization.search_space.tunable_model_config_space import TunableModelConfigSpace

if TYPE_CHECKING:
    import optuna

    from rago.model.configs.retriever_config.base import RetrieverConfig
    from rago.optimization.search_space.qdrant_retriever_config_space import QdrantRetrieverConfigSpace


@dataclass
class RetrieverModelConfigSpace(TunableModelConfigSpace):
    """Config space that samples a :class:`RetrieverModelConfig`.

    Wraps an existing :class:`RetrieverConfigSpace` or
    :class:`QdrantRetrieverConfigSpace`.
    """

    retriever_space: Optional[Union[RetrieverConfigSpace, QdrantRetrieverConfigSpace]] = None

    def sample(self, trial: optuna.trial.BaseTrial) -> RetrieverModelConfig:
        """Sample a retriever-only model configuration.

        :param trial: Optuna trial.
        :type trial: optuna.trial.BaseTrial
        :return: A retriever model config.
        :rtype: RetrieverModelConfig
        """
        if self.retriever_space is None:
            self.retriever_space = RetrieverConfigSpace()
        retriever_config: RetrieverConfig = self.retriever_space.sample(trial)
        return RetrieverModelConfig(retriever=retriever_config)
