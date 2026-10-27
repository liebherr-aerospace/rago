"""Test the dispatch of the retriever wrapper factory."""

import pytest

from rago.model.configs.retriever_config.base import RetrieverConfig
from rago.model.configs.retriever_config.langchain import HybridLangchainRetrieverConfig
from rago.model.wrapper.retriever.retriever_wrapper_factory import RetrieverWrapperFactory
from rago.model.wrapper.tunable_model import BuildContext


def test_hybrid_config_routed_to_hybrid_wrapper() -> None:
    """A hybrid config must reach the hybrid wrapper, not the parent Langchain one."""
    with pytest.raises(ValueError, match="hybrid_weight"):
        RetrieverWrapperFactory.make(HybridLangchainRetrieverConfig(), BuildContext(inputs_chunks=[]))


def test_unsupported_config_raises() -> None:
    """An unknown retriever config raises a TypeError."""
    with pytest.raises(TypeError):
        RetrieverWrapperFactory.make(RetrieverConfig())
