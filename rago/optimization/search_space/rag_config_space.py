"""Define the RAG config space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import optuna

    from rago.model.configs.retriever_config.base import RetrieverConfig
    from rago.optimization.search_space.qdrant_retriever_config_space import QdrantRetrieverConfigSpace

from rago.model.wrapper.rag.base import RAGConfig
from rago.optimization.search_space.reader_config_space import LangchainReaderConfigSpace, ReaderConfigSpace
from rago.optimization.search_space.retriever_config_space import RetrieverConfigSpace as _RetrieverConfigSpace
from rago.optimization.search_space.tunable_model_config_space import TunableModelConfigSpace


@dataclass
class RAGConfigSpace(TunableModelConfigSpace):
    """Config space that samples a full :class:`RAGConfig` (reader **and** retriever).

    Both ``retriever_space`` and ``reader_space`` default to sensible config
    spaces when left as ``None``.  For retriever-only or reader-only
    optimisation use :class:`RetrieverModelConfigSpace` or
    :class:`ReaderModelConfigSpace` instead.
    """

    retriever_space: Optional[Union[_RetrieverConfigSpace, QdrantRetrieverConfigSpace]] = None
    reader_space: Optional[ReaderConfigSpace] = None

    def sample(self, trial: optuna.trial.BaseTrial) -> RAGConfig:
        """Sample a RAG configuration from configuration spaces.

        :param trial: Trial used to sample the configuration
        :type trial: optuna.trial.BaseTrial
        :return: The sampled RAG configuration.
        :rtype: RAGConfig
        """
        # Reader
        if self.reader_space is None:
            self.reader_space = LangchainReaderConfigSpace()
        reader_config = self.reader_space.sample(trial)

        # Retriever
        if self.retriever_space is None:
            self.retriever_space = _RetrieverConfigSpace()
        retriever_config: RetrieverConfig = self.retriever_space.sample(trial)

        return RAGConfig(
            reader=reader_config,
            retriever=retriever_config,
        )
