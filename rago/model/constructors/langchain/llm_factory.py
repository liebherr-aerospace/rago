"""Define a Langchain LLM Factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from langchain_huggingface import HuggingFaceEndpoint
from langchain_ollama import OllamaLLM

from rago.model.configs.llm_config.langchain import (
    LangchainHuggingFaceConfig,
    LangchainLLMConfig,
    LangchainOllamaConfig,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseLLM


class LangchainLLMFactory:
    """A Langchain LLM Factory."""

    @staticmethod
    def make(config: Optional[LangchainLLMConfig] = None) -> BaseLLM:
        """Make a Langchain LLM.

        :param config: The configuration of the langchain LLM to make.
        :type config: LangchainLLMConfig
        :return: The built LLM.
        :rtype: BaseLLM
        """
        if config is None:
            config = LangchainOllamaConfig()
        match config:
            case LangchainHuggingFaceConfig():
                return LangchainLLMFactory.make_huggingface_llm(config)
            case LangchainOllamaConfig():
                return LangchainLLMFactory.make_ollama_llm(config)
            case _:
                raise NotImplementedError(config)

    @staticmethod
    def make_ollama_llm(
        config: LangchainOllamaConfig,
    ) -> OllamaLLM:
        """Make a langchain llm with a Ollama backend.

        :param config: Configuration parameters of the langchain ollama llm to build .
        :type config: LangchainOllamaConfig
        :return: The built ollama model.
        :rtype: OllamaLLM
        """
        return OllamaLLM(
            model=config.model_name,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            num_predict=config.max_new_tokens,
            num_ctx=config.num_ctx,
            repeat_last_n=config.repeat_last_n,
            mirostat=config.mirostat,
            mirostat_eta=config.mirostat_eta,
            mirostat_tau=config.mirostat_tau,
            base_url=config.base_url,
            client_kwargs=config.client_kwargs,
        )

    @staticmethod
    def make_huggingface_llm(
        config: LangchainHuggingFaceConfig,
    ) -> BaseLLM:
        """Make a langchain LLM with a HuggingFace Inference API backend.

        :param config: Configuration parameters for the HuggingFace LLM.
        :type config: LangchainHuggingFaceConfig
        :return: The built HuggingFace LLM.
        :rtype: BaseLLM
        """
        kwargs: dict = {
            "repo_id": config.model_name,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "max_new_tokens": config.max_new_tokens,
        }
        if config.huggingfacehub_api_token is not None:
            kwargs["huggingfacehub_api_token"] = config.huggingfacehub_api_token
        if config.endpoint_url is not None:
            kwargs["endpoint_url"] = config.endpoint_url

        return HuggingFaceEndpoint(**kwargs)
