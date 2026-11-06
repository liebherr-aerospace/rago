"""Define an abstract reader class."""

from __future__ import annotations

from abc import abstractmethod

from rago.data_objects import EvalSample, ModelInput, RetrievalOutput
from rago.model.configs.retriever_config.base import RetrieverConfig
from rago.model.wrapper.tunable_model import TunableModel


class Retriever[ConfigT: RetrieverConfig](TunableModel[ModelInput, RetrievalOutput, ConfigT]):
    """A retriever that retrieves information relevant to answer a query from a database (non-parametric knowledge)."""

    @classmethod
    def get_input(cls, eval_sample: EvalSample) -> ModelInput:
        """Extract the retriever from the eval sample.

        :param eval_sample: The eval sample to extract the input from.
        :type eval_sample: EvalSample
        :return: The extracted retriever input.
        :rtype: ModelInput
        """
        return ModelInput(eval_sample.query)

    @abstractmethod
    def _get_output(self, model_input: ModelInput) -> RetrievalOutput:
        """Retrieve the context relevant to model input, without generating an answer.

        :param model_input: The ModelInput containing the query to answer.
        :type query: ModelInput
        :return: The retrieval output containing the retrieved context.
        :rtype: RetrievalOutput
        """

