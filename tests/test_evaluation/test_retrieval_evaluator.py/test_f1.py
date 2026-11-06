"""Test the context retrieval evaluator (relevancy, precision, context_f1)."""

import math
import random

import pytest

from rago.data_objects import EvalSample, RetrievalOutput, RetrievedContext
from rago.eval import EvaluatorReference, RetrievalF1Evaluator, RetrievalPrecisionEvaluator, RetrievalRecallEvaluator


@pytest.fixture
def f1_evaluator() -> RetrievalF1Evaluator:
    return RetrievalF1Evaluator()

def test_all_metrics_returned(
        f1_evaluator: RetrievalF1Evaluator,
        ref_w_2_contexts: EvaluatorReference,
        retrieved_output_1c_1f: RetrievalOutput) -> None:
    """One relevant chunk out of two retrieved, one expected chunk out of two found."""
    result = f1_evaluator.evaluate(retrieved_output_1c_1f, ref_w_2_contexts)
    assert result.recall == 1 / 2
    assert result.precision == 1 / 2
    assert result.score == pytest.approx(1 / 2)


def test_perfect_retrieval(
        f1_evaluator: RetrievalF1Evaluator,
        ref_w_2_contexts: EvaluatorReference,
        retrieved_output_2c: RetrievalOutput,
        ) -> None:
    """Retrieving exactly the expected chunks gives 1 everywhere."""
    result = f1_evaluator.evaluate(retrieved_output_2c, ref_w_2_contexts)
    assert result.recall == 1
    assert result.precision == 1
    assert result.score == 1


def test_noise_lowers_precision_only(
        f1_evaluator: RetrievalF1Evaluator,
        ref_w_2_contexts: EvaluatorReference,
        retrieved_output_2c_1f: RetrievalOutput) -> None:
    """Adding irrelevant chunks keeps relevancy but lowers precision."""
    result = f1_evaluator.evaluate(retrieved_output_2c_1f, ref_w_2_contexts)
    assert result.recall == 1
    assert result.precision == 2 / 3
    assert result.score == 4 / 5


def test_precision_inclusion_lower_max_ratio(
        context_1: str,
        ref_w_1_context: EvaluatorReference,
        ) -> None:
    """A retrieved chunk counts as relevant if it contains or is contained in an expected chunk."""
    num_random_evals = 10**2
    scale = 10
    for _ in range(num_random_evals):
        max_ratio = 1 + scale* random.random()
        f1_evaluator = RetrievalF1Evaluator(max_ratio)
        effective_ratio = 1 + (max_ratio-1)* random.random()
        text = context_1 + math.floor(len(context_1)*(effective_ratio -1))*"a"
        assert len(text)/len(context_1) < max_ratio
        output = RetrievalOutput(retrieved_context=[RetrievedContext(text)])
        assert f1_evaluator.evaluate(output, ref_w_1_context).score == 1

def test_precision_inclusion_equal_max_ratio(
        context_1: str,
        ref_w_1_context: EvaluatorReference,
        f1_evaluator: RetrievalF1Evaluator,
        ) -> None:
    """A retrieved chunk counts as relevant if it contains or is contained in an expected chunk."""
    num_random_evals = 10**2
    scale = 10
    for _ in range(num_random_evals):
        max_ratio = scale* random.randint(1, scale)
        f1_evaluator = RetrievalF1Evaluator(max_ratio)
        text = context_1 + math.floor(len(context_1)*(max_ratio -1))*"a"
        output = RetrievalOutput(retrieved_context=[RetrievedContext(text)])
        assert f1_evaluator.evaluate(output, ref_w_1_context).score == 1

def test_precision_inclusion_greater_max_ratio(
        context_1: str,
        ref_w_1_context: EvaluatorReference,
        ) -> None:
    """A retrieved chunk counts as relevant if it contains or is contained in an expected chunk."""
    num_random_evals = 10**3
    scale = 10
    for _ in range(num_random_evals):
        max_ratio = 1 + scale* random.random()
        f1_evaluator = RetrievalF1Evaluator(max_ratio)
        effective_ratio = max_ratio + scale* random.random()
        text = context_1 + math.ceil(len(context_1)*(effective_ratio -1))*"a"
        assert len(text)/len(context_1) > max_ratio
        output = RetrievalOutput(retrieved_context=[RetrievedContext(text)])
        assert f1_evaluator.evaluate(output, ref_w_1_context).score == 0


def test_no_retrieved_context(
        f1_evaluator: RetrievalF1Evaluator,
        ref_w_2_contexts: EvaluatorReference,
        retrieved_output_0: RetrievalOutput) -> None:
    """Expected context but nothing retrieved gives 0 everywhere."""
    result = f1_evaluator.evaluate(retrieved_output_0, ref_w_2_contexts)
    assert result.recall == 0
    assert result.precision == 0
    assert result.score == 0


def test_no_expected_context(
        f1_evaluator: RetrievalF1Evaluator,
        sample_wo_context: EvalSample,
        retrieved_output_0: RetrievalOutput) -> None:
    """No expected context gives 1 everywhere."""
    result = f1_evaluator.evaluate(retrieved_output_0, EvaluatorReference(sample_wo_context))
    assert result.recall == 1
    assert result.precision == 1
    assert result.score == 1


def test_n_wise_equals_individual(f1_evaluator: RetrievalF1Evaluator, ref_w_2_contexts: EvaluatorReference) -> None:
    """The evaluator is independent: n-wise results equal individual results."""
    outputs = [
        RetrievalOutput(retrieved_context=[RetrievedContext("Thomas is 12.")]),
        RetrievalOutput(retrieved_context=[RetrievedContext("Thomas is 13.")]),
    ]
    results = [f1_evaluator.evaluate(o, ref_w_2_contexts) for o in outputs]
    n_wise_results = f1_evaluator.evaluate_n_wise(outputs, ref_w_2_contexts)
    for r, n_r in zip(results, n_wise_results, strict=True):
        assert r == n_r


def test_recall_precision_evaluators_match_recall_precision(
        f1_evaluator: RetrievalF1Evaluator, ref_w_2_contexts: EvaluatorReference) -> None:
    """Deprecated evaluators warn and return the same score under their legacy key only."""
    output = RetrievalOutput(
        retrieved_context=[RetrievedContext("Thomas is 12."), RetrievedContext("Paris is in France.")])
    expected = f1_evaluator.evaluate(output, ref_w_2_contexts)

    recall = RetrievalRecallEvaluator().evaluate(output, ref_w_2_contexts)
    precision = RetrievalPrecisionEvaluator().evaluate(output, ref_w_2_contexts)

    assert recall.score == expected.recall
    assert precision.score == expected.precision
