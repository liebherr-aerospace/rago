"""Define the different LM implementation using LlamaIndex as backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import llama_index.core.llms.llm as llama_index_llm

from rago.model.wrapper.llm_agent.base import LLMAgent
from rago.model.wrapper.llm_agent.message import Message

if TYPE_CHECKING:
    import llama_index.core.llms.llm as llama_index_llm

    from rago.model.configs.llm_config.llama_index import LLamaIndexLLMConfig

from rago.model.constructors.llama_index.llm_factory import LLamaIndexLLMFactory


class LlamaIndexLLMAgent(LLMAgent):
    """A Language Model (LM) to answer queries using LlamaIndex as a backend."""

    language_model: llama_index_llm.LLM

    def __init__(self, language_model: llama_index_llm.LLM, system_prompt: Optional[str] = None) -> None:
        """Instantiate a LlamaIndex model with the llama-index model and its system Prompt.

        :param language_model: The LlamaIndex LM to use.
        :type language_model: llama_index.core.llms.llm
        :param system_prompt: The system prompt used by the langchain model.
        System is the standard name for commands given to the LLM on how to behave.
        :type system_prompt:  Optional[str]
        """
        super().__init__(system_prompt)
        self.language_model = language_model

    @classmethod
    def make_from_backend(
        cls,
        llm_config: Optional[LLamaIndexLLMConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> LlamaIndexLLMAgent:
        """Build a llm agent from its config and the config used by its backend.

        :param config: Configuration of the LLM agent to build.
        :type config: dict
        :param system_prompt: The system prompt used by the langchain model.
        System is the standard name for commands given to the LLM on how to behave.
        :type system_prompt: Optional[str]
        :return: the LLM agent
        :rtype: LlamaIndexLLMAgent
        """
        language_model = LLamaIndexLLMFactory.make(config=llm_config)
        return cls(language_model=language_model, system_prompt=system_prompt)

    def chat(self, chat_sequence: list[Message], **kwargs: Any) -> Message:  # noqa: ANN401
        """Chat with LM by providing new and previous interaction as sequence of message.

        :param chat_sequence: Message interaction sequence from first to current message.
        :type chat_sequence: list[Message]
        :param kwargs: Generation args used by the LM.
        :type kwargs: dict
        :return: The response of the language model.
        :rtype: Message
        """
        llama_index_messages = [Message.get_llama_index_message(message) for message in chat_sequence]
        model_answer_message = Message.from_llama_index_message(
            self.language_model.chat(llama_index_messages, **kwargs).message,
        )
        return model_answer_message
