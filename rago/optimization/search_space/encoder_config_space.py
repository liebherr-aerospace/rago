"""Defines Encoder config space and its subspaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field
from pydantic.dataclasses import dataclass

from rago.model.configs.encoder_config.langchain import HuggingFaceLangchainEncoderConfig, LangchainEncoderConfig
from rago.optimization.search_space.config_space import ConfigSpace
from rago.optimization.search_space.param_space import CategoricalParamSpace

if TYPE_CHECKING:
    import optuna


@dataclass
class EncoderConfigSpace(ConfigSpace):
    """The Space containing all the different encoder configurations."""

    model_name: CategoricalParamSpace

    def sample(self, trial: optuna.trial.BaseTrial) -> LangchainEncoderConfig:
        """Sample an encoder configuration.

        :param trial: the trial used to sample elements.
        :type trial: optuna.trial.BaseTrial
        :return: The sampled encoder config.
        :rtype: LangchainEncoderConfig
        """
        model_name = self.model_name.sample(trial)
        if not isinstance(model_name, str):
            raise TypeError(model_name)
        return LangchainEncoderConfig(
            model_name=model_name,
        )


@dataclass
class HFEncoderConfigSpace(EncoderConfigSpace):
    """The space containing all the HuggingFace encoder configurations."""

    model_name: CategoricalParamSpace = Field(
        default=CategoricalParamSpace(
            name="encoder_model_name",
            choices=[
                "BAAI/bge-m3",
                "intfloat/e5-large-v2",
                "Qwen/Qwen3-Embedding-0.6B",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ],
        ),
    )

    def sample(self, trial: optuna.trial.BaseTrial) -> HuggingFaceLangchainEncoderConfig:
        """Sample Hugging Face Encoder configuration from configuration space.

        :param trial: the trial used to sample elements.
        :type trial: optuna.trial.BaseTrial
        :return: The sampled Encoder Hugging Face Langchain Encoder config.
        :rtype: HuggingFaceLangchainEncoderConfig
        """
        return HuggingFaceLangchainEncoderConfig(
            model_name=self.model_name.sample(trial),
        )
