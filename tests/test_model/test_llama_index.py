"""Test the LlamaIndex class."""

from rago.model.wrapper.llm_agent.llama_index import LlamaIndexLLMAgent


def test_query(llama_index_llm_agent: LlamaIndexLLMAgent) -> None:
    """Test that the output for fixed query hello has not changed in deterministic case with LlamaIndex."""
    query = "Hello"
    assert llama_index_llm_agent.query(query) == "Hello! How can I help you today?"
