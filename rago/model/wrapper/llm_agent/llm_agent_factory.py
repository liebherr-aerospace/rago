"""Define a factory of LLM agents."""
from typing import Optional

from rago.model.configs.llm_config.base import LLMConfig
from rago.model.configs.llm_config.langchain import LangchainLLMConfig
from rago.model.configs.llm_config.llama_index import LLamaIndexLLMConfig
from rago.model.wrapper.llm_agent.base import LLMAgent
from rago.model.wrapper.llm_agent.langchain import LangchainLLMAgent
from rago.model.wrapper.llm_agent.llama_index import LlamaIndexLLMAgent


class LLMAgentFactory:
    """Factory of LLM agents."""

    @staticmethod
    def make(config: LLMConfig, system_prompt: Optional[str] = None, output_prompt: Optional[str] = None) -> LLMAgent:
        """Make a LLM Agent.

        :param config: The config of the LLM agent to make.
        :type config: LLMConfig
        :raises TypeError: The LLM Agent type is unknown.
        :return: The created LLM Agent.
        :rtype: LLMAgent
        """
        match config:
            case LangchainLLMConfig():
                return LangchainLLMAgent.make(
                    config,
                    system_prompt=system_prompt,
                    output_prompt=output_prompt)
            case LLamaIndexLLMConfig():
                return LlamaIndexLLMAgent.make(config)
            case _:
                raise TypeError(config)
