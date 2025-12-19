"""Test Message class."""

from __future__ import annotations

from typing import Any, cast

import pytest
from langchain_core.messages import HumanMessage

from rago.model.wrapper.llm_agent.message import Message


def test_langchain_message_with_string_content() -> None:
    """Test that the output for fixed query hello has not changed in deterministic case with Langchain."""
    content = "Hello"
    msg = Message.from_langchain_message(HumanMessage(content))
    assert msg.text == content


def test_langchain_message_with_list_string_content() -> None:
    content: list[str] = ["Hello", "Hello"]
    msg = Message.from_langchain_message(HumanMessage(cast(str | list[str | dict[Any, Any]], content)))
    assert msg.text == "\n".join(content)


def test_langchain_message_with_list_dict_content() -> None:
    content: list[dict[str, str]] = [{"str": "str"}]
    with pytest.raises(TypeError) as excinfo:
        Message.from_langchain_message(HumanMessage(cast(str | list[str | dict[Any, Any]], content)))
    assert str(excinfo.value) == "[{'str': 'str'}]"
