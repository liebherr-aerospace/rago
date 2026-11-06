"""Define the pytest.fixture used by the different evaluation test."""
import pytest

from rago.data_objects import EvalSample
from rago.eval import DependentEvaluatorReference


@pytest.fixture
def dependent_reference() -> DependentEvaluatorReference:
    """Make a fake eval reference used throughout the judge evaluation tests."""
    return DependentEvaluatorReference(
        eval_sample = EvalSample("How old is thomas", reference_answer = "Thomas is 12."),
        )
