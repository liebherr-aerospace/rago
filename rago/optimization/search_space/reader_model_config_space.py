"""Define the ReaderModel config space."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic.dataclasses import dataclass

from rago.model.wrapper.reader_model import ReaderModelConfig
from rago.optimization.search_space.reader_config_space import LangchainReaderConfigSpace, ReaderConfigSpace
from rago.optimization.search_space.tunable_model_config_space import TunableModelConfigSpace

if TYPE_CHECKING:
    import optuna


@dataclass
class ReaderModelConfigSpace(TunableModelConfigSpace):
    """Config space that samples a :class:`ReaderModelConfig`.

    Wraps an existing :class:`ReaderConfigSpace`.
    """

    reader_space: Optional[ReaderConfigSpace] = None

    def sample(self, trial: optuna.trial.BaseTrial) -> ReaderModelConfig:
        """Sample a reader-only model configuration.

        :param trial: Optuna trial.
        :type trial: optuna.trial.BaseTrial
        :return: A reader model config.
        :rtype: ReaderModelConfig
        """
        if self.reader_space is None:
            self.reader_space = LangchainReaderConfigSpace()
        reader_config = self.reader_space.sample(trial)
        return ReaderModelConfig(reader=reader_config)
