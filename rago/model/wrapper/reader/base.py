"""Define an abstract reader class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from rago.prompts import AMBIGQA_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT, PromptConfig

if TYPE_CHECKING:
    from rago.data_objects import RetrievedContext


class Reader(ABC):
    """A Reader takes as input a query (and optionally a context) and outputs a string response."""

    @abstractmethod
    def get_reader_output(self, query: str, retrieved_context: Optional[list[RetrievedContext]] = None) -> str:
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
