"""Message used to interact with chat models."""

from __future__ import annotations

from enum import StrEnum
from typing import Self, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from llama_index.core.base.llms.types import ChatMessage as LLamaIndexChatMessage
from llama_index.core.base.llms.types import MessageRole


class Role(StrEnum):
    """Role of the sender of the message. Used by the Message class to define the role of the message.

    Role is an Enum in case new roles are added in the future.
    """

    USER = "user"
    """
    User is when the message is from user.
    """
    SYSTEM = "system"
    """
    Bot is when the message is from the Bot.
    """
    BOT = "bot"
    """
    System corresponds to system prompt message.
    """

    @classmethod
    def from_llama_index_role(cls, role: MessageRole) -> Role:
        """Get the Role from the LLamaIndex role.

        :param role: The LlamaIndex role.
        :type role: MessageRole
        :return: The role.
        :rtype: Role.
        """
        match role:
            case MessageRole.SYSTEM:
                return cls.SYSTEM
            case MessageRole.USER:
                return cls.USER
            case MessageRole.ASSISTANT:
                return cls.BOT
            case _:
                raise TypeError(role)

    @staticmethod
    def get_llama_index_role(role: Role) -> MessageRole:
        """Get LLamaIndex role from the role.

        :param role: The role of the message.
        :type role: Role
        :return: The llamaIndex role.
        :rtype: MessageRole
        """
        match role:
            case Role.SYSTEM:
                return MessageRole.SYSTEM
            case Role.USER:
                return MessageRole.USER
            case Role.BOT:
                return MessageRole.ASSISTANT

    @classmethod
    def from_langchain_role(cls, role: str) -> Role:
        """Get the Role from the Langchain role.

        :param role: The Langchain role.
        :type role: str
        :return: The Role.
        :rtype: Role
        """
        match role:
            case "human":
                return cls.USER
            case "ai":
                return cls.BOT
            case "system":
                return cls.SYSTEM
            case _:
                raise TypeError(role)


class Message:
    """A message is a information unit (query, answer etc) sent from a sender to a receiver.

    Message sent in a chat interaction either by the bot, the user or the system.
    Used by the LM as input and output of chat.
    """

    text: str
    role: Role

    def __init__(self, text: str, role: Role) -> None:
        """Instantiate a message from its text and role.

        :param text: The text of the message.
        :type text: str
        :param role: the role of the sender.
        :type role: Role
        """
        self.text = text
        self.role = role

    @classmethod
    def from_llama_index_message(cls: type[Self], message: LLamaIndexChatMessage) -> Message:
        """Get message from the LlamaIndex message.

        :param message: The LlamaIndex message.
        :type message: ChatMessage
        :return: The Message corresponding to the LlamaIndex message.
        :rtype: Message
        """
        return cls(message.content if message.content is not None else "", Role.from_llama_index_role(message.role))

    def get_llama_index_message(self) -> LLamaIndexChatMessage:
        """Get the LlamaIndex message from the message.

        :return: The llamaIndex message.
        :rtype: ChatMessage
        """
        return LLamaIndexChatMessage(self.text, role=Role.get_llama_index_role(self.role))

    @classmethod
    def from_langchain_message(cls: type[Self], message: BaseMessage) -> Message:
        """Get message from the Langchain message.

        :param message: The Langchain message.
        :type message: BaseMessage
        :return: The Message corresponding to the Langchain message.
        :rtype: Message
        """
        content = message.content
        if isinstance(content, str):
            return cls(content, Role.from_langchain_role(message.type))
        if all(isinstance(msg, str) for msg in content):
            content_str: list[str] = cast(list[str], content)
            return cls("\n".join(content_str), Role.from_langchain_role(message.type))
        raise TypeError(content)

    def get_langchain_message(self) -> BaseMessage:
        """Get the Langchain message from the message.

        :return: The Langchain message.
        :rtype: BaseMessage
        """
        match self.role:
            case Role.SYSTEM:
                return SystemMessage(self.text)
            case Role.USER:
                return HumanMessage(self.text)
            case Role.BOT:
                return AIMessage(self.text)
