"""Define a factory of LLM agents."""

from rago.model.configs.llm_config.base import LLMConfig
from rago.model.configs.llm_config.langchain import LangchainLLMConfig
from rago.model.configs.llm_config.llama_index import LLamaIndexLLMConfig
from rago.model.wrapper.llm_agent.base import LLMAgent
from rago.model.wrapper.llm_agent.langchain import LangchainLLMAgent
from rago.model.wrapper.llm_agent.llama_index import LlamaIndexLLMAgent


class LLMAgentFactory:
    """Factory of LLM agents."""

    @staticmethod
    def make(config: LLMConfig) -> LLMAgent:
        """Make a LLM Agent.

        :param config: The config of the LLM agent to make.
        :type config: LLMConfig
        :raises TypeError: The LLM Agent type is unknown.
        :return: The created LLM Agent.
        :rtype: LLMAgent
        """
        match config:
            case LangchainLLMConfig():
                return LangchainLLMAgent.make_from_backend(config)
            case LLamaIndexLLMConfig():
                return LlamaIndexLLMAgent.make_from_backend(config)
            case _:
                raise TypeError(config)
