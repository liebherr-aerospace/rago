"""Test that the Ollama `reasoning` option is propagated from configs and config spaces to the built LLMs."""

import optuna
import pytest

from rago.model.configs.llm_config.langchain import LangchainOllamaConfig
from rago.model.configs.llm_config.llama_index import LlamaIndexOllamaConfig
from rago.model.constructors.langchain.llm_factory import LangchainLLMFactory
from rago.model.constructors.llama_index.llm_factory import LLamaIndexLLMFactory
from rago.optimization.search_space.llm_config_space import OllamaLlamaIndexLLMConfigSpace, OllamaLLMConfigSpace


def test_reasoning_disabled_by_default() -> None:
    assert LangchainOllamaConfig().reasoning is False
    assert LlamaIndexOllamaConfig().reasoning is False


@pytest.mark.parametrize("reasoning", [True, False, None])
def test_langchain_factory_forwards_reasoning(*, reasoning: bool | None) -> None:
    llm = LangchainLLMFactory.make_ollama_llm(LangchainOllamaConfig(reasoning=reasoning))
    assert llm.reasoning is reasoning


@pytest.mark.parametrize("reasoning", [True, False, None])
def test_llama_index_factory_forwards_reasoning(*, reasoning: bool | None) -> None:
    llm = LLamaIndexLLMFactory.make_ollama(LlamaIndexOllamaConfig(reasoning=reasoning))
    assert llm.thinking is reasoning  # type: ignore[attr-defined]


@pytest.mark.parametrize("reasoning", [True, False, None])
def test_config_spaces_forward_reasoning(*, reasoning: bool | None) -> None:
    trial = optuna.create_study().ask()
    assert OllamaLLMConfigSpace(reasoning=reasoning).sample(trial).reasoning is reasoning
    assert OllamaLlamaIndexLLMConfigSpace(reasoning=reasoning).sample(trial).reasoning is reasoning
