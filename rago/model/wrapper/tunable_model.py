"""Define the TunableModel abstract base class."""


from __future__ import annotations

import time
from abc import ABC, abstractmethod

from rago.data_objects import ModelOutput
from rago.model.configs.tunable_model_config import TunableModelConfig


class TunableModel[OutputType: ModelOutput, ConfigType: TunableModelConfig](ABC):
    """Abstract base class for any model that can be optimised by the Optuna manager.

    Subclasses include full RAG pipelines, retriever-only models and
    reader-only models.  The optimiser interacts exclusively through the
    :meth:`get_output` interface so it is agnostic of the concrete topology.
    """

    config: ConfigType
    @classmethod
    @abstractmethod
    def make(cls, config: ConfigType, *args ,**kwargs) -> TunableModel:# noqa: ANN002, ANN003
        """Build Model from config."""

    @abstractmethod
    def _get_output(self, *args, **kwargs) -> OutputType:...  # noqa: ANN002, ANN003

    def get_output(self, *args, **kwargs) -> OutputType: # noqa: ANN002, ANN003
        """Produce an output for the given *query*.

        :param query: The user query.
        :type query: str
        :return: The model output (answer and/or retrieved context).
        :rtype: RAGOutput
        """
        start = time.time()
        output = self._get_output(*args, **kwargs)
        end = time.time()
        output.time = end -start
        return output

