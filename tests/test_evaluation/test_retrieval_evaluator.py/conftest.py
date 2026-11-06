"""Define the pytest.fixture used by the different evaluation test."""
import pytest

from rago.data_objects import Document, EvalSample, RetrievalOutput, RetrievedContext
from rago.eval import EvaluatorReference


@pytest.fixture
def query() -> str:
    """Return a fake query used through out the evaluation tests."""
    return "How old is Thomas?"

@pytest.fixture
def context_1()-> str:
    """Return a fake context used through out the evaluation tests."""
    return "Thomas is 12."

@pytest.fixture
def context_2() -> str:
    """Return a fake context used through out the evaluation tests."""
    return "Thomas was born 12 years ago."

@pytest.fixture
def context_3() -> str:
    """Return a fake context used through out the evaluation tests."""
    return "Paris is in France."

@pytest.fixture
def sample_wo_context(query: str) -> EvalSample:
    """Return a fake eval sample with query and no context doc used through out the evaluation tests."""
    return EvalSample(query)

@pytest.fixture
def sample_w_1_context(query: str, context_1: str) -> EvalSample:
    """Return a fake eval sample with query and one context doc used through out the evaluation tests."""
    return EvalSample(query, [Document(context_1)])

@pytest.fixture
def sample_w_2_contexts(query: str, context_1: str, context_2: str) -> EvalSample:
    """Return a fake eval sample with query and two context doc used through out the evaluation tests."""
    return EvalSample(query, [Document(context_1), Document(context_2)])

@pytest.fixture
def ref_w_1_context(sample_w_1_context: EvalSample) -> EvaluatorReference:
    """Return a fake ref with query and 1 context doc used through out the evaluation tests."""
    return EvaluatorReference(sample_w_1_context)

@pytest.fixture
def ref_w_2_contexts(sample_w_2_contexts: EvalSample) -> EvaluatorReference:
    """Return a fake ref with query and two contexts doc used through out the evaluation tests."""
    return EvaluatorReference(sample_w_2_contexts)

@pytest.fixture
def retrieved_output_0() -> RetrievalOutput:
    """Return a fake ref with query and two contexts doc used through out the evaluation tests."""
    return RetrievalOutput([])

@pytest.fixture
def retrieved_output_1c_1f(
    context_1: str,
    context_3:str,
    ) -> RetrievalOutput:
    """Return a fake retrieved_output with 1 correct and 1 false contexts used through out the evaluation tests."""
    return RetrievalOutput([RetrievedContext(context_1), RetrievedContext(context_3)])

@pytest.fixture
def retrieved_output_2c(
    context_1: str,
    context_2:str,
    ) -> RetrievalOutput:
    """Return a fake retrieved_output with 2 correct contexts used through out the evaluation tests."""
    return RetrievalOutput([RetrievedContext(context_1), RetrievedContext(context_2)])
@pytest.fixture
def retrieved_output_2c_1f(
    context_1: str,
    context_2:str,
    context_3: str,
    ) -> RetrievalOutput:
    """Return a fake retrieved_output with 2 correct and 1 false contexts used through out the evaluation tests."""
    return RetrievalOutput([RetrievedContext(context_1), RetrievedContext(context_2), RetrievedContext(context_3)])
