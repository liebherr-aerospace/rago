"""Define a retriever wrapper factory."""

from rago.model.configs.retriever_config.base import RetrieverConfig
from rago.model.configs.retriever_config.langchain import LangchainRetrieverConfig
from rago.model.configs.retriever_config.llama_index import LlamaIndexRetrieverConfig
from rago.model.wrapper.retriever.base import Retriever
from rago.model.wrapper.retriever.langchain_retriever import LangchainRetrieverWrapper
from rago.model.wrapper.retriever.llama_index_retriever import LlamaIndexRetrieverWrapper


class RetrieverWrapperFactory:
    """Factory of retriever wrappers."""

    @staticmethod
    def make(config: RetrieverConfig, input_chunks: list[str]) -> Retriever:
        """Make a retriever wrapper.

        :param config: The config of the retriever wrapper to make.
        :type config: RetrieverConfig
        :param input_chunks: The input chunks used by the retriever wrapper.
        :type input_chunks: list[str]
        :return: The created retriever wrapper.
        :rtype: Retriever
        """
        match config:
            case LlamaIndexRetrieverConfig():
                return LlamaIndexRetrieverWrapper.make(config, input_chunks)
            case LangchainRetrieverConfig():
                return LangchainRetrieverWrapper.make(config, input_chunks)
            case _:
                raise TypeError(config)
