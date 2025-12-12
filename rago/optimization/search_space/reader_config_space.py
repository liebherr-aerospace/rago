"""Define a reader config space."""

from abc import abstractmethod

import optuna
from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.model.configs.reader_config.base import ReaderConfig
from rago.model.configs.reader_config.langchain import LangchainReaderConfig
from rago.model.configs.reader_config.llama_index import LLamaIndexReaderConfig
from rago.optimization.search_space.config_space import ConfigSpace
from rago.optimization.search_space.llm_config_space import OllamaLlamaIndexLLMConfigSpace, OllamaLLMConfigSpace
from rago.optimization.search_space.param_space import CategoricalParamSpace


@dataclass
class ReaderConfigSpace(ConfigSpace):
    """Space of containing all the possible reader configurations."""

    @abstractmethod
    def sample(self, trial: optuna.trial.BaseTrial) -> ReaderConfig:
        """Sample a configuration from configuration space.

        :param trial: Trial used to sample the configuration
        :type trial: optuna.trial.BaseTrial
        :return: The sampled configuration.
        :rtype: ReaderConfig
        """


@dataclass
class LangchainReaderConfigSpace(ReaderConfigSpace):
    """Space of containing all the possible reader configurations by using Langchain."""

    llm_config: OllamaLLMConfigSpace = Field(default=OllamaLLMConfigSpace())

    def sample(self, trial: optuna.trial.BaseTrial) -> LangchainReaderConfig:
        """Sample a configuration from configuration space.

        :param trial: Trial used to sample the configuration
        :type trial: optuna.trial.BaseTrial
        :return: The sampled configuration of the Langchain reader.
        :rtype: LangchainReaderConfig
        """
        return LangchainReaderConfig(llm=self.llm_config.sample(trial))


@dataclass
class LlamaIndexReaderConfigSpace(ReaderConfigSpace):
    """Space of containing all the possible reader configurations by using llama index."""

    type_config: CategoricalParamSpace = Field(
        default=CategoricalParamSpace(
            choices=["Refine", "CompactAndRefine", "TreeSummarize", "SimpleSummarize"],
        ),
    )
    llm_config: OllamaLlamaIndexLLMConfigSpace = Field(default=OllamaLlamaIndexLLMConfigSpace())

    def sample(self, trial: optuna.trial.BaseTrial) -> LLamaIndexReaderConfig:
        """Sample a configuration from configuration space.

        :param trial: Trial used to sample the configuration
        :type trial: optuna.trial.BaseTrial
        :return: The sampled configuration of the llama_index reader.
        :rtype: LLamaIndexReaderConfig
        """
        return LLamaIndexReaderConfig(type=self.type_config.sample(trial), llm=self.llm_config.sample(trial))
