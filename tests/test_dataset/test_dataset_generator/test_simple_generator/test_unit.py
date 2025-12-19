"""Unit Tests of the DatasetGenerator."""

import uuid
from unittest.mock import MagicMock, patch

from rago.data_objects import Document, EvalSample
from rago.dataset import RAGDataset
from rago.dataset.generator import SimpleDatasetGenerator


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_parse_generation_output_type(lm: MagicMock) -> None:
    """Test the type of the parse_generation's output is correct."""
    generation = str(uuid.uuid4())
    dataset_generator = SimpleDatasetGenerator(lm)
    document = Document(str(uuid.uuid4()))
    parsed_generation = dataset_generator.parse_generation(generation=generation, document=document)
    assert isinstance(parsed_generation, list)
    assert all(isinstance(query, EvalSample) for query in parsed_generation)


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_parse_generation_output_query(lm: MagicMock) -> None:
    """Test the query value of the parse_generation's output is correct."""
    generation = str(uuid.uuid4())
    dataset_generator = SimpleDatasetGenerator(lm)
    document = Document(str(uuid.uuid4()))
    expected_output = [EvalSample(generation_line, [document]) for generation_line in generation.split("\n")]
    assert dataset_generator.parse_generation(generation, document=document) == expected_output


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_parse_generation_output_context(lm: MagicMock) -> None:
    """Test the context value of the parse_generation's output is correct."""
    generation = str(uuid.uuid4())
    dataset_generator = SimpleDatasetGenerator(lm)
    document = Document(str(uuid.uuid4()))
    parsed_generation = dataset_generator.parse_generation(generation=generation, document=document)
    assert isinstance(parsed_generation, list)
    assert all(eval_sample.context == [document] for eval_sample in parsed_generation)


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_generate_dataset_output_is_correct(
    lm: MagicMock,
    hotpot_qa_sample: RAGDataset,
) -> None:
    """Test the dataset is correctly generated."""
    lm.query.return_value = str(uuid.uuid4())
    dataset_generator = SimpleDatasetGenerator(lm)
    generated_dataset = dataset_generator.generate_dataset(hotpot_qa_sample)

    expected_samples = hotpot_qa_sample.samples + [
        EvalSample(lm.query.return_value, [document]) for document in hotpot_qa_sample.corpus.values()
    ]
    assert generated_dataset == RAGDataset(expected_samples, hotpot_qa_sample.corpus)
