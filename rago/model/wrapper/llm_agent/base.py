"""Define the base language model (LM) abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from rago.model.wrapper.llm_agent.message import Message, Role


class LLMAgent(ABC):
    """Define an LLM Agent that answer queries or chat discussions."""

    system_message: Optional[Message]

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        """Instantiate a Language Model with an eventual system prompt.

        :param system_prompt: The system prompt used by the model.
        System is the standard name for commands given to the LLM on how to behave.
        :type system_prompt: Optional[str]
        """
        self.system_message = Message(system_prompt, Role.SYSTEM) if system_prompt is not None else None

    @abstractmethod
    def chat(self, chat_sequence: list[Message], **kwargs: Any) -> Message:  # noqa: ANN401
        """Chat with LM by providing new and previous interaction as sequence of message.

        :param chat_sequence: Message interaction sequence from first to current message.
        :type chat_sequence: list[Message]
        :param kwargs: Generation args used by the chat model.
        :type kwargs: dict
        :return: The response of the language model.
        :rtype: Message
        """

    def batch_chat(self, batch_chat_sequence: list[list[Message]], **kwargs: Any) -> list[Message]:  # noqa: ANN401
        """Pass a batch of chat discussions to an agent simultaneously.

        :param batch_chat_sequence: Batch of interaction messages sequences.
        :type batch_chat_sequence: list[list[Message]]
        :return: Batch of answer to the batch of chat sequences.
        :rtype: list[Message]
        """
        raise NotImplementedError

    def batch_query(self, prompts: list[str], **kwargs: Any) -> list[str]:  # noqa: ANN401
        """Pass a batch of queries to answer to language model.

        :param prompts: Batch of prompts passed to the language model.
        :type prompts: list[str]
        :return: Batch of answer to the batch of prompts.
        :rtype: list[str]
        """
        batch_message_sequence = [self.get_message_sequence_from_query(prompt) for prompt in prompts]
        batch_answer_message = self.batch_chat(batch_message_sequence, **kwargs)
        return [msg.text for msg in batch_answer_message]

    def query(self, prompt: str, **kwargs: Any) -> str:  # noqa: ANN401
        """Query Language Model an input prompt.

        :param prompt: Prompt to use for the query.
        :type prompt: str
        :return: The response of the language model.
        :rtype: str
        """
        message_sequence = self.get_message_sequence_from_query(prompt)
        return self.chat(message_sequence, **kwargs).text

    def get_message_sequence_from_query(self, query: str) -> list[Message]:
        """Get the message sequence to send to the LM by adding the LM system prompt if it exists.

        :param query: The query used to create a message sequence.
        :type query: str
        :return: The sequence of message to send to the LM.
        :rtype: list[Message]
        """
        message_sequence: list[Message] = []
        if self.system_message:
            message_sequence.append(self.system_message)
        user_message = Message(query, Role.USER)
        message_sequence.append(user_message)
        return message_sequence
