"""Define an abstract reader class."""

from __future__ import annotations

from abc import abstractmethod

from rago.data_objects import EvalSample, GenerationOutput, ReaderInput, RetrievedContext
from rago.model.configs.reader_config.base import ReaderConfig
from rago.model.wrapper.tunable_model import TunableModel
from rago.prompts import AMBIGQA_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT, PromptConfig


class Reader[ConfigT: ReaderConfig](TunableModel[ReaderInput, GenerationOutput, ConfigT]):
    """A Reader takes as input a query (and optionally a context) and outputs a string response."""

    @classmethod
    def get_input(cls, eval_sample: EvalSample) -> ReaderInput:
        """Extract the model from the eval sample.

        :param eval_sample: The eval sample to extract the input from.
        :type eval_sample: EvalSample
        :return: The extracted model input.
        :rtype: InputT
        """
        retrieved_context = (
            [RetrievedContext(context.text) for context in eval_sample.context]
            if eval_sample.context is not None else [])
        return ReaderInput(eval_sample.query, retrieved_context)

    @abstractmethod
    def _get_output(self, model_input: ReaderInput) -> GenerationOutput:
        """Get the reader output answer to the input query and optional retrieved context.

        :param query: The query the reader needs to answer.
        :type query: str
        :param retrieved_context: The context retrieved to help answer the query if set, defaults to None.
        :type retrieved_context: Optional[list[str]], optional
        :return: The reader output answer.
        :rtype: str
        """
        ...


    @classmethod
    def get_prompt_template_string(cls, prompt_config: PromptConfig) -> str:
        """Get the prompt template of the reader from the prompt_config.

        :param prompt_config: The configurations params of the reader's prompt template.
        :type prompt_config: PromptConfig
        :raises ValueError: If the dataset_name is unknown.
        :return: the reader prompt_template as a str.
        :rtype: str
        """
        if prompt_config.system_message is None:
            if prompt_config.dataset_name is None:
                system_message = DEFAULT_SYSTEM_PROMPT
            else:
                match prompt_config.dataset_name:
                    case "ambig_qa" | "asqa":
                        system_message = AMBIGQA_SYSTEM_PROMPT
                    case _:
                        raise ValueError(prompt_config.dataset_name)
        else:
            system_message = prompt_config.system_message
        return system_message
