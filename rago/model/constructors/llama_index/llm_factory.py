"""Define a LLMBuilder that build Llama_index LLMs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from llama_index.llms.ollama import Ollama
from ollama import Client

from rago.model.configs.llm_config.llama_index import LlamaIndexOllamaConfig

if TYPE_CHECKING:
    from llama_index.core.llms import LLM

    from rago.model.configs.llm_config.llama_index import LLMConfig


class LLamaIndexLLMFactory:
    """LLMBuilder that builds LlamaIndex LLMs."""

    @staticmethod
    def make(config: Optional[LLMConfig] = None) -> LLM:
        """Build LlamaIndex llm with config.

        :param config: Configuration of the LLamaIndex LLM to build.
        :type config: dict
        :return: The built llama-index LLM.
        :rtype: LLM
        """
        if config is None:
            config = LlamaIndexOllamaConfig()
        match config:
            case LlamaIndexOllamaConfig():
                return LLamaIndexLLMFactory.make_ollama(config)
            case _:
                raise TypeError(config)

    @staticmethod
    def make_ollama(
        config: LlamaIndexOllamaConfig,
        keep_alive: Optional[float] = None,
        timeout: Optional[int] = None,
    ) -> LLM:
        """Build an llamaIndex LLM with an Ollama backend.

        :param config: Configuration parameters of the llama-Index Ollama llm.
        :type config: LlamaIndexOllamaConfig
        :param keep_alive: Duration time to keep the LLM on the Ollama server without calls, defaults to None.
        :type keep_alive: Optional[float], optional
        :param timeout: Max time to wait for an answer from the LLM on the Ollama server, defaults to None.
        :type timeout: Optional[int], optional
        :return: The built LLM.
        :rtype: LLM
        """
        client = Client(host=config.base_url, timeout=timeout, verify=False)
        return Ollama(
            model=config.model_name,
            base_url=config.base_url,
            temperature=config.temperature,
            additional_kwargs={
                "num_ctx": config.num_ctx,
                "top_k": config.top_k,
                "top_p": config.top_p,
                "num_predict": config.max_new_tokens,
                "repeat_last_n": config.repeat_last_n,
                "mirostat": config.mirostat,
                "mirostat_eta": config.mirostat_eta,
                "mirostat_tau": config.mirostat_tau,
            },
            client=client,
            keep_alive=keep_alive,
        )
