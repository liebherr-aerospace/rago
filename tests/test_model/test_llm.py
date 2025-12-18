"""Test that all the different subclass of BaseLanguageModel have the same properties."""

import uuid

import pytest
from langchain_core.language_models import BaseLLM

from rago.model.wrapper.llm_agent.langchain import LangchainLLMAgent
from rago.model.wrapper.llm_agent.llama_index import LlamaIndexLLMAgent
from rago.model.wrapper.llm_agent.message import Message, Role


@pytest.mark.skip(reason="Skipping this test for now as Ollama is not yet fully deterministic.")
def test_deterministic(langchain_llm_agent: LangchainLLMAgent) -> None:
    """Test that subsequent calls to the same llm produce the same output."""
    query = str(uuid.uuid4())

    assert langchain_llm_agent.query(query) == langchain_llm_agent.query(query)


def test_same_answer_llm(langchain_llm_agent: LangchainLLMAgent, llama_index_llm_agent: LlamaIndexLLMAgent) -> None:
    """Test that when temperature is 0 (i.e. deterministic case) llama_index and langchain have the same input."""
    query = "Hello"

    assert langchain_llm_agent.query(query) == llama_index_llm_agent.query(query)


def test_system_prompt_unused_type(langchain_llm: BaseLLM) -> None:
    """Test that when system is not set when instantiating LM the system_message attribute is set to None."""
    langchain_model = LangchainLLMAgent(
        language_model=langchain_llm,
    )
    assert langchain_model.system_message is None


def test_system_prompt_used_type_msg(langchain_llm: BaseLLM) -> None:
    """Test that when it is the system_message attribute is a Message object."""
    langchain_model = LangchainLLMAgent(
        language_model=langchain_llm,
        system_prompt="Hello",
    )
    assert isinstance(langchain_model.system_message, Message)


def test_system_prompt_used_role(langchain_llm: BaseLLM) -> None:
    """Test that when it is the system_message's Role is Role.SYSTEM."""
    langchain_model = LangchainLLMAgent(
        language_model=langchain_llm,
        system_prompt="Hello",
    )
    assert isinstance(langchain_model.system_message, Message)
    assert langchain_model.system_message.role == Role.SYSTEM


def test_output_type_get_message_sequence_from_query_without_system_prompt(
    langchain_llm: BaseLLM,
) -> None:
    """Test that without system_prompt the output list of get_message_sequence_from_query only contains messages."""
    query = str(uuid.uuid4())
    langchain_model = LangchainLLMAgent(
        language_model=langchain_llm,
    )
    assert all(isinstance(msg, Message) for msg in langchain_model.get_message_sequence_from_query(query))


def test_output_len_get_message_sequence_from_query_without_system_prompt(langchain_llm: BaseLLM) -> None:
    """Test that with system_prompt the output list of get_message_sequence_from_query contains only one message."""
    query = str(uuid.uuid4())
    langchain_model = LangchainLLMAgent(
        language_model=langchain_llm,
    )
    assert len(langchain_model.get_message_sequence_from_query(query)) == 1


def test_output_type_get_message_sequence_from_query_with_system_prompt(langchain_llm: BaseLLM) -> None:
    """Test that with system_prompt the output list of get_message_sequence_from_query contains only message object."""
    query = str(uuid.uuid4())
    langchain_model = LangchainLLMAgent(
        language_model=langchain_llm,
        system_prompt="Hello",
    )
    assert all(isinstance(msg, Message) for msg in langchain_model.get_message_sequence_from_query(query))


def test_output_len_get_message_sequence_from_query_with_system_prompt(
    langchain_llm: BaseLLM,
) -> None:
    """Test that without system_prompt the output list of get_message_sequence_from_query is of len 2."""
    query = str(uuid.uuid4())
    langchain_model = LangchainLLMAgent(
        language_model=langchain_llm,
        system_prompt="Hello",
    )
    assert len(langchain_model.get_message_sequence_from_query(query)) == 2  # noqa: PLR2004


def test_chat(langchain_llm_agent: LangchainLLMAgent) -> None:
    """Test that the output of chat is a Message Object."""
    query = "Hello"
    message_sequence = langchain_llm_agent.get_message_sequence_from_query(query)
    assert isinstance(langchain_llm_agent.chat(message_sequence), Message)


def test_empty_message(langchain_llm_agent: LangchainLLMAgent) -> None:
    """Test that with an empty list as input the output message of chat has empty text."""
    assert langchain_llm_agent.chat([]).text == ""
