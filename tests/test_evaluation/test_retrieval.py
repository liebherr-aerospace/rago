"""Test the context retrieval evaluator (relevancy, precision, context_f1)."""

import pytest

from rago.data_objects import Document, EvalSample, RAGOutput, RetrievedContext
from rago.eval import ContextRetrievalEvaluator, PrecisionEvaluator, RelevancyEvaluator


@pytest.fixture
def eval_sample() -> EvalSample:
    """Sample with two expected chunks."""
    return EvalSample(
        "How old is thomas",
        [Document("Thomas is 12."), Document("Thomas was born 12 years ago.")],
    )


def test_all_metrics_returned(eval_sample: EvalSample) -> None:
    """One relevant chunk out of two retrieved, one expected chunk out of two found."""
    output = RAGOutput(retrieved_context=[RetrievedContext("Thomas is 12."), RetrievedContext("Paris is in France.")])
    result = ContextRetrievalEvaluator().evaluate(output, eval_sample)
    assert result["relevancy"].score == 1 / 2
    assert result["precision"].score == 1 / 2
    assert result["context_f1"].score == pytest.approx(1 / 2)


def test_perfect_retrieval(eval_sample: EvalSample) -> None:
    """Retrieving exactly the expected chunks gives 1 everywhere."""
    output = RAGOutput(
        retrieved_context=[RetrievedContext("Thomas is 12."), RetrievedContext("Thomas was born 12 years ago.")],
    )
    result = ContextRetrievalEvaluator().evaluate(output, eval_sample)
    assert result["relevancy"].score == 1
    assert result["precision"].score == 1
    assert result["context_f1"].score == 1


def test_noise_lowers_precision_only(eval_sample: EvalSample) -> None:
    """Adding irrelevant chunks keeps relevancy but lowers precision."""
    output = RAGOutput(
        retrieved_context=[
            RetrievedContext("Thomas is 12."),
            RetrievedContext("Thomas was born 12 years ago."),
            RetrievedContext("Paris is in France."),
            RetrievedContext("The sky is blue."),
        ],
    )
    result = ContextRetrievalEvaluator().evaluate(output, eval_sample)
    assert result["relevancy"].score == 1
    assert result["precision"].score == 1 / 2
    assert result["context_f1"].score == pytest.approx(2 / 3)


def test_precision_matches_both_inclusion_directions() -> None:
    """A retrieved chunk counts as relevant if it contains or is contained in an expected chunk."""
    sample = EvalSample("q", [Document("Thomas is 12 years old and lives in Paris.")])
    output = RAGOutput(retrieved_context=[RetrievedContext("Thomas is 12 years old")])
    assert ContextRetrievalEvaluator().evaluate(output, sample)["precision"].score == 1


def test_no_retrieved_context(eval_sample: EvalSample) -> None:
    """Expected context but nothing retrieved gives 0 everywhere."""
    result = ContextRetrievalEvaluator().evaluate(RAGOutput("Thomas is 12."), eval_sample)
    assert result["relevancy"].score == 0
    assert result["precision"].score == 0
    assert result["context_f1"].score == 0


def test_no_expected_context() -> None:
    """No expected context gives 1 everywhere."""
    output = RAGOutput(retrieved_context=[RetrievedContext("Thomas is 12.")])
    result = ContextRetrievalEvaluator().evaluate(output, EvalSample("Hello!"))
    assert result["relevancy"].score == 1
    assert result["precision"].score == 1
    assert result["context_f1"].score == 1


def test_n_wise_equals_individual(eval_sample: EvalSample) -> None:
    """The evaluator is independent: n-wise results equal individual results."""
    evaluator = ContextRetrievalEvaluator()
    outputs = [
        RAGOutput(retrieved_context=[RetrievedContext("Thomas is 12.")]),
        RAGOutput(retrieved_context=[RetrievedContext("Thomas is 13.")]),
    ]
    results = [evaluator.evaluate(o, eval_sample) for o in outputs]
    n_wise_results = evaluator.evaluate_n_wise(outputs, eval_sample)
    for r, n_r in zip(results, n_wise_results, strict=True):
        assert {k: m.score for k, m in r.items()} == {k: m.score for k, m in n_r.items()}


def test_deprecated_evaluators_match_new_one(eval_sample: EvalSample) -> None:
    """Deprecated evaluators warn and return the same score under their legacy key only."""
    output = RAGOutput(retrieved_context=[RetrievedContext("Thomas is 12."), RetrievedContext("Paris is in France.")])
    expected = ContextRetrievalEvaluator().evaluate(output, eval_sample)

    with pytest.warns(DeprecationWarning, match="RelevancyEvaluator is deprecated"):
        relevancy = RelevancyEvaluator().evaluate(output, eval_sample)
    with pytest.warns(DeprecationWarning, match="PrecisionEvaluator is deprecated"):
        precision = PrecisionEvaluator().evaluate(output, eval_sample)

    assert relevancy.keys() == {"relevancy"}
    assert precision.keys() == {"precision"}
    assert relevancy["relevancy"].score == expected["relevancy"].score
    assert precision["precision"].score == expected["precision"].score
