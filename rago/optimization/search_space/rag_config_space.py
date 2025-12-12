"""Define the RAG config space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import optuna

    from rago.model.configs.reader_config.base import ReaderConfig
    from rago.model.configs.retriever_config.base import RetrieverConfig

from rago.model.wrapper.rag.base import RAGConfig
from rago.optimization.search_space.config_space import ConfigSpace
from rago.optimization.search_space.reader_config_space import LangchainReaderConfigSpace, ReaderConfigSpace
from rago.optimization.search_space.retriever_config_space import RetrieverConfigSpace


@dataclass
class RAGConfigSpace(ConfigSpace):
    """Define the RAG Config space."""

    retriever_space: Optional[RetrieverConfigSpace] = None
    reader_space: Optional[ReaderConfigSpace] = None

    def sample(self, trial: optuna.trial.BaseTrial) -> RAGConfig:
        """Sample a RAG configuration from configuration spaces.

        :param trial: Trial used to sample the configuration
        :type trial: optuna.trial.BaseTrial
        :return: The sampled RAG configuration.
        :rtype: RAGConfig
        """
        if self.reader_space is None:
            self.reader_space = LangchainReaderConfigSpace()
        reader_config: ReaderConfig = self.reader_space.sample(trial)
        if self.retriever_space is None:
            self.retriever_space = RetrieverConfigSpace()
        retriever_config: RetrieverConfig = self.retriever_space.sample(trial)
        return RAGConfig(
            reader=reader_config,
            retriever=retriever_config,
        )
