"""Integration Test of the dataset generator."""

import os

import pytest

from rago.dataset import RAGDataset
from rago.dataset.generator import SimpleDatasetGenerator
from rago.model.wrapper.llm_agent.langchain import LangchainLLMAgent

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test ignored in GitHub Actions")
def test_generated_dataset_correct_test_samples_number(
    langchain_llm_agent: LangchainLLMAgent,
    hotpot_qa_sample: RAGDataset,
    expected_dataset: RAGDataset,
) -> None:
    """Test that the Simple generator actually generates the expected dataset in the deterministic case."""
    dataset_generator = SimpleDatasetGenerator(langchain_llm_agent)

    generated_dataset = dataset_generator.generate_dataset(hotpot_qa_sample)
    assert generated_dataset == expected_dataset
