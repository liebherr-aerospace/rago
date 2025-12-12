"""Define the different LM implementation using langchain as backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, cast

from rago.model.constructors.langchain.llm_factory import LangchainLLMFactory
from rago.model.wrapper.llm_agent.base import LLMAgent
from rago.model.wrapper.llm_agent.message import Message, Role

if TYPE_CHECKING:
    from collections.abc import Sequence

    import langchain_core.language_models.llms as langchain_llm
    from langchain_core.messages import BaseMessage
    from langchain_core.prompt_values import PromptValue

    from rago.model.configs.llm_config.langchain import LangchainLLMConfig


class LangchainLLMAgent(LLMAgent):
    """A Language Model to answer queries using Langchain as a backend."""

    language_model: langchain_llm.BaseLLM

    def __init__(self, language_model: langchain_llm.BaseLLM, system_prompt: Optional[str] = None) -> None:
        """Instantiate a Langchain model with the langchain model and its system Prompt.

        :param language_model: The Langchain LM to use.
        :type language_model: langchain_core.language_models.llms
        :param system_prompt: The system prompt used by the langchain model.
        System is the standard name for commands given to the LLM on how to behave.
        :type system_prompt:  Optional[str]
        """
        super().__init__(system_prompt)
        self.language_model = language_model

    @classmethod
    def make_from_backend(
        cls,
        config: Optional[LangchainLLMConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> LangchainLLMAgent:
        """Build a llm agent from its config and the config used by its backend.

        :param config: Configuration of the LLM agent to build.
        :type config: dict
        :param system_prompt: The system prompt used by the langchain model.
        System is the standard name for commands given to the LLM on how to behave.
        :type system_prompt:  Optional[str]
        :return: _description_
        :rtype: _type_
        """
        return LangchainLLMAgent(
            language_model=LangchainLLMFactory.make(config=config),
            system_prompt=system_prompt,
        )

    def chat(self, chat_sequence: list[Message], **kwargs: Any) -> Message:  # noqa: ANN401
        """Chat with LM by providing new and previous interaction as sequence of message.

        :param chat_sequence: Message interaction sequence from first to current message.
        :type chat_sequence: list[Message]
        :param kwargs: Generation args used by the LM.
        :type kwargs: dict
        :return: The response of the language model.
        :rtype: Message
        """
        langchain_messages = [Message.get_langchain_message(message) for message in chat_sequence]
        model_answer_message = Message(
            self.language_model.invoke(langchain_messages, **kwargs),
            Role.BOT,
        )
        return model_answer_message

    def batch_chat(self, batch_chat_sequence: list[list[Message]], **kwargs: Any) -> list[Message]:  # noqa: ANN401
        """Pass a batch of chat discussions to an agent simultaneously.

        :param batch_chat_sequence: Batch of interaction messages sequences.
        :type batch_chat_sequence: list[list[Message]]
        :return: Batch of answer to the batch of chat sequences.
        :rtype: list[Message]
        """
        batch_langchain_messages = [
            [Message.get_langchain_message(message) for message in chat_sequence]
            for chat_sequence in batch_chat_sequence
        ]
        if TYPE_CHECKING:
            new_batch_langchain_messages = cast(
                list[PromptValue | str | Sequence[BaseMessage | list[str] | tuple[str, str] | str | dict[str, Any]]],
                batch_langchain_messages,
            )
        batch_model_answer = self.language_model.batch(new_batch_langchain_messages, **kwargs)
        batch_model_answer_message = [Message(model_answer, Role.BOT) for model_answer in batch_model_answer]
        return batch_model_answer_message
