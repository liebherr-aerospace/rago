"""Defines LLM config space and its subspaces."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.model.configs.llm_config.langchain import LangchainOllamaConfig
from rago.model.configs.llm_config.llama_index import LlamaIndexOllamaConfig
from rago.optimization.search_space.config_space import ConfigSpace
from rago.optimization.search_space.param_space import CategoricalParamSpace, FloatParamSpace, IntParamSpace

if TYPE_CHECKING:
    import optuna

DEFAULT_LLM_CHOICE = ["smollm2:1.7b", "qwen3:4b", "gemma3:4b", "llama3.2:3b"]


@dataclass
class LLMConfigSpace(ConfigSpace):
    """The Space containing all the different llm configurations."""

    model_name: CategoricalParamSpace
    temperature: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=1.0))
    top_k: IntParamSpace = Field(default=IntParamSpace(low=0, high=10000))
    top_p: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=1.0))
    max_new_tokens: IntParamSpace = Field(default=IntParamSpace(low=64, high=1024))

    def sample(self, trial: optuna.trial.BaseTrial) -> LangchainOllamaConfig:
        """Sample a LLM configuration from LLM configuration space.

        :param trial: the trial used to sample elements.
        :type trial: optuna.trial.BaseTrial
        :return: The sampled LLM config.
        :rtype: LLMConfig
        """
        return LangchainOllamaConfig(
            model_name=self.model_name.sample(trial),
            temperature=self.temperature.sample(trial),
            top_k=self.top_k.sample(trial),
            top_p=self.top_p.sample(trial),
            max_new_tokens=self.max_new_tokens.sample(trial),
        )


@dataclass
class HFLLMConfigSpace(LLMConfigSpace):
    """The space containing all the HuggingFace LLM configurations."""

    model_name: CategoricalParamSpace = Field(
        default=CategoricalParamSpace(
            choices=["HuggingFaceTB/SmolLM2-360M-Instruct-Q8-mlx"],
        ),
    )


@dataclass
class OllamaLLMConfigSpace(LLMConfigSpace):
    """The space containing all the Ollama LLM configurations."""

    model_name: CategoricalParamSpace = Field(
        default=CategoricalParamSpace(
            choices=DEFAULT_LLM_CHOICE,
        ),
    )
    mirostat: CategoricalParamSpace = Field(
        default=CategoricalParamSpace(
            choices=[0, 1, 2],
        ),
    )
    mirostat_eta: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=1.0))
    mirostat_tau: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=1.0))
    num_ctx: IntParamSpace = Field(default=IntParamSpace(low=64, high=12800, step=64))
    repeat_last_n: IntParamSpace = Field(default=IntParamSpace(low=-1, high=256))
    base_url: str = Field(default=os.environ.get("TEST_OLLAMA_HOST", ""))
    client_kwargs: dict[str, bool] = Field(default={"verify": False})

    def sample(self, trial: optuna.trial.BaseTrial) -> LangchainOllamaConfig:
        """Sample a Ollama LLM configuration from LLM configuration space.

        :param trial: the trial used to sample elements.
        :type trial: optuna.trial.BaseTrial
        :return: The sampled LLM Langchain Ollama config.
        :rtype: LangchainOllamaConfig
        """
        mirostat = self.mirostat.sample(trial)
        mirostat_eta = None if mirostat == 0 else self.mirostat_eta.sample(trial)
        mirostat_tau = None if mirostat == 0 else self.mirostat_tau.sample(trial)
        num_ctx = self.num_ctx.sample(trial)
        repeat_last_n = self.repeat_last_n.sample(trial)
        base_config = super().sample(trial)
        return LangchainOllamaConfig(
            model_name=base_config.model_name,
            temperature=base_config.temperature,
            top_k=base_config.top_k,
            top_p=base_config.top_p,
            max_new_tokens=base_config.max_new_tokens,
            mirostat=mirostat,
            mirostat_eta=mirostat_eta,
            mirostat_tau=mirostat_tau,
            num_ctx=num_ctx,
            repeat_last_n=repeat_last_n,
            base_url=self.base_url,
            client_kwargs=self.client_kwargs,
        )


@dataclass
class LlamaIndexLLMConfigSpace(ConfigSpace):
    """The Space containing all the different llm configurations."""

    model_name: CategoricalParamSpace
    temperature: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=1.0))
    top_k: IntParamSpace = Field(default=IntParamSpace(low=0, high=10000))
    top_p: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=1.0))
    max_new_tokens: IntParamSpace = Field(default=IntParamSpace(low=64, high=256))

    def sample(self, trial: optuna.trial.BaseTrial) -> LlamaIndexOllamaConfig:
        """Sample a LLM configuration from LLM configuration space.

        :param trial: the trial used to sample elements.
        :type trial: optuna.trial.BaseTrial
        :return: The sampled LLM config.
        :rtype: LlamaIndexOllamaConfig
        """
        return LlamaIndexOllamaConfig(
            model_name=self.model_name.sample(trial),
            temperature=self.temperature.sample(trial),
            top_k=self.top_k.sample(trial),
            top_p=self.top_p.sample(trial),
            max_new_tokens=self.max_new_tokens.sample(trial),
        )


@dataclass
class OllamaLlamaIndexLLMConfigSpace(LlamaIndexLLMConfigSpace):
    """The space containing all the Ollama LLM configurations."""

    model_name: CategoricalParamSpace = Field(
        default=CategoricalParamSpace(
            choices=DEFAULT_LLM_CHOICE,
        ),
    )
    mirostat: CategoricalParamSpace = Field(
        default=CategoricalParamSpace(
            choices=[0, 1, 2],
        ),
    )
    mirostat_eta: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=1.0))
    mirostat_tau: FloatParamSpace = Field(default=FloatParamSpace(low=0.0, high=1.0))
    num_ctx: IntParamSpace = Field(default=IntParamSpace(low=64, high=4096, step=64))
    repeat_last_n: IntParamSpace = Field(default=IntParamSpace(low=-1, high=256))
    base_url: str = Field(default=os.environ.get("TEST_OLLAMA_HOST", ""))
    client_kwargs: dict[str, bool] = Field(default={"verify": False})

    def sample(self, trial: optuna.trial.BaseTrial) -> LlamaIndexOllamaConfig:
        """Sample a Ollama LLM configuration from LLM configuration space.

        :param trial: the trial used to sample elements.
        :type trial: optuna.trial.BaseTrial
        :return: The sampled LLM llamaIndex Ollama config.
        :rtype: LlamaIndexOllamaConfig
        """
        mirostat = self.mirostat.sample(trial)
        mirostat_eta = None if mirostat == 0 else self.mirostat_eta.sample(trial)
        mirostat_tau = None if mirostat == 0 else self.mirostat_tau.sample(trial)
        num_ctx = self.num_ctx.sample(trial)
        repeat_last_n = self.repeat_last_n.sample(trial)
        base_config = super().sample(trial)
        return LlamaIndexOllamaConfig(
            model_name=base_config.model_name,
            temperature=base_config.temperature,
            top_k=base_config.top_k,
            top_p=base_config.top_p,
            max_new_tokens=base_config.max_new_tokens,
            mirostat=mirostat,
            mirostat_eta=mirostat_eta,
            mirostat_tau=mirostat_tau,
            num_ctx=num_ctx,
            repeat_last_n=repeat_last_n,
            base_url=self.base_url,
            client_kwargs=self.client_kwargs,
        )
