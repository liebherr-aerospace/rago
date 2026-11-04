"""Test the context retrieval evaluator (relevancy, precision, context_f1)."""

import pytest

from rago.data_objects import Document, EvalSample, RetrievalOutput, RetrievedContext
from rago.eval import EvaluatorReference, RetrievalF1Evaluator, RetrievalPrecisionEvaluator, RetrievalRecallEvaluator


@pytest.fixture
def reference() -> EvaluatorReference:
    """Sample with two expected chunks."""
    return EvaluatorReference(EvalSample(
        "How old is thomas",
        [Document("Thomas is 12."), Document("Thomas was born 12 years ago.")],
    ))


def test_all_metrics_returned(reference: EvaluatorReference) -> None:
    """One relevant chunk out of two retrieved, one expected chunk out of two found."""
    output = RetrievalOutput(retrieved_context=[RetrievedContext("Thomas is 12."), RetrievedContext("Paris is in France.")])
    result = RetrievalF1Evaluator().evaluate(output, reference)
    assert result.recall == 1 / 2
    assert result.precision == 1 / 2
    assert result.score == pytest.approx(1 / 2)


def test_perfect_retrieval(reference: EvaluatorReference) -> None:
    """Retrieving exactly the expected chunks gives 1 everywhere."""
    output = RetrievalOutput(
        retrieved_context=[RetrievedContext("Thomas is 12."), RetrievedContext("Thomas was born 12 years ago.")],
    )
    result = RetrievalF1Evaluator().evaluate(output, reference)
    assert result.recall == 1
    assert result.precision == 1
    assert result.score == 1


def test_noise_lowers_precision_only(reference: EvaluatorReference) -> None:
    """Adding irrelevant chunks keeps relevancy but lowers precision."""
    output = RetrievalOutput(
        retrieved_context=[
            RetrievedContext("Thomas is 12."),
            RetrievedContext("Thomas was born 12 years ago."),
            RetrievedContext("Paris is in France."),
            RetrievedContext("The sky is blue."),
        ],
    )
    result = RetrievalF1Evaluator().evaluate(output, reference)
    assert result.recall == 1
    assert result.precision == 1 / 2
    assert result.score == pytest.approx(2 / 3)


def test_precision_matches_both_inclusion_directions() -> None:
    """A retrieved chunk counts as relevant if it contains or is contained in an expected chunk."""
    ref = EvaluatorReference(EvalSample("q", [Document("Thomas is 12 years old and lives in Paris.")]))
    output = RetrievalOutput(retrieved_context=[RetrievedContext("Thomas is 12 years old")])
    assert RetrievalF1Evaluator().evaluate(output, ref).score == 1


def test_no_retrieved_context(reference: EvaluatorReference) -> None:
    """Expected context but nothing retrieved gives 0 everywhere."""
    result = RetrievalF1Evaluator().evaluate(RetrievalOutput([]), reference)
    assert result.recall == 0
    assert result.precision == 0
    assert result.score == 0


def test_no_expected_context() -> None:
    """No expected context gives 1 everywhere."""
    output = RetrievalOutput([])
    result = RetrievalF1Evaluator().evaluate(output, EvaluatorReference(EvalSample("Hello!")))
    assert result.recall == 1
    assert result.precision == 1
    assert result.score == 1


def test_n_wise_equals_individual(reference: EvaluatorReference) -> None:
    """The evaluator is independent: n-wise results equal individual results."""
    evaluator = RetrievalF1Evaluator()
    outputs = [
        RetrievalOutput(retrieved_context=[RetrievedContext("Thomas is 12.")]),
        RetrievalOutput(retrieved_context=[RetrievedContext("Thomas is 13.")]),
    ]
    results = [evaluator.evaluate(o, reference) for o in outputs]
    n_wise_results = evaluator.evaluate_n_wise(outputs, reference)
    for r, n_r in zip(results, n_wise_results, strict=True):
        assert r == n_r


def test_deprecated_evaluators_match_new_one(reference: EvaluatorReference) -> None:
    """Deprecated evaluators warn and return the same score under their legacy key only."""
    output = RetrievalOutput(retrieved_context=[RetrievedContext("Thomas is 12."), RetrievedContext("Paris is in France.")])
    expected = RetrievalF1Evaluator().evaluate(output, reference)

    with pytest.warns(DeprecationWarning, match="RelevancyEvaluator is deprecated"):
        recall = RetrievalRecallEvaluator().evaluate(output, reference)
    with pytest.warns(DeprecationWarning, match="PrecisionEvaluator is deprecated"):
        precision = RetrievalPrecisionEvaluator().evaluate(output, reference)

    assert recall.score == expected.recall
    assert precision.score == expected.precision
