"""Test the Langchain class."""

from rago.model.wrapper.llm_agent.langchain import LangchainLLMAgent


def test_query(langchain_llm_agent: LangchainLLMAgent) -> None:
    """Test that the output for fixed query hello has not changed in deterministic case with Langchain."""
    query = "Hello"
    assert langchain_llm_agent.query(query) == "Hello! How can I help you today?"
