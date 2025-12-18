"""Test the evaluator in isolation."""

import math
from unittest.mock import MagicMock, patch

import pytest

from rago.data_objects import Document, EvalSample, RAGOutput
from rago.eval import JudgeError, PolicyOnError, SimpleLLMEvaluator


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_evaluation_in_interval_score_with_min_score_policy(lm: MagicMock) -> None:
    """Test that with a score inside the authorized interval the output score is the same score with min score policy.

    i.e test that it is correctly parsed.
    """
    min_score = 0
    max_score = 5
    lm.query.return_value = "0"
    evaluator = SimpleLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    eval_sample = EvalSample(query="How old is thomas")
    eval_result = evaluator.evaluate(RAGOutput("Thomas is 12."), eval_sample)
    assert eval_result["correctness"].score == 0


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_evaluation_output_out_interval_score_with_min_score_policy_2(lm: MagicMock) -> None:
    """Test that with a score outside the authorized interval, output score is the min score with min score policy."""
    min_score = 0
    max_score = 5
    lm.query.return_value = "6"
    evaluator = SimpleLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    eval_sample = EvalSample("How old is thomas", [Document("Thomas is 12.")])
    eval_result = evaluator.evaluate(RAGOutput("Thomas is 12."), eval_sample)
    assert eval_result["correctness"].score == 0


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_evaluation_output_non_digit_score_with_min_score_policy_3(lm: MagicMock) -> None:
    """Test that with a non digit score the output score is the min score with the min score policy."""
    min_score = 0
    max_score = 5
    lm.query.return_value = "e"
    evaluator = SimpleLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    eval_sample = EvalSample("How old is thomas", [Document("Thomas is 12.")])
    eval_result = evaluator.evaluate(RAGOutput("Thomas is 12."), eval_sample)
    assert eval_result["correctness"].score == 0


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_evaluation_in_interval_score_with_nan_score_policy(lm: MagicMock) -> None:
    """Test that with a score inside the authorized interval the output score is the same score with nan score policy.

    i.e test that it is correctly parsed.
    """
    min_score = 0
    max_score = 5
    lm.query.return_value = "3"
    evaluator = SimpleLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.NAN_SCORE,
    )
    eval_sample = EvalSample("How old is thomas", [Document("Thomas is 12.")])
    eval_result = evaluator.evaluate(RAGOutput("Thomas is 12."), eval_sample)
    assert eval_result["correctness"].score == int(lm.query.return_value)


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_evaluation_out_interval_score_with_nan_score_policy(lm: MagicMock) -> None:
    """Test that with a score outside the authorized interval the output score is nan with the nan score policy."""
    min_score = 0
    max_score = 5
    lm.query.return_value = "6"
    evaluator = SimpleLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.NAN_SCORE,
    )
    eval_sample = EvalSample("How old is thomas", [Document("Thomas is 12.")])
    eval_result = evaluator.evaluate(RAGOutput("Thomas is 12."), eval_sample)
    assert eval_result["correctness"].score is not None
    assert math.isnan(eval_result["correctness"].score)


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_evaluation_string_score_with_nan_score_policy(lm: MagicMock) -> None:
    """Test that with a non digit score the output score is nan with the nan score policy."""
    min_score = 0
    max_score = 5
    lm.query.return_value = "e"
    evaluator = SimpleLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.NAN_SCORE,
    )
    eval_sample = EvalSample("How old is thomas", [Document("Thomas is 12.")])
    eval_result = evaluator.evaluate(RAGOutput("Thomas is 12."), eval_sample)
    assert eval_result["correctness"].score is not None
    assert math.isnan(eval_result["correctness"].score)


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_evaluation_in_interval_score_with_trow_policy(lm: MagicMock) -> None:
    """Test that with a score inside the authorized interval the output score is the same."""
    min_score = 0
    max_score = 5
    lm.query.return_value = "3"
    evaluator = SimpleLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.THROW_ERROR,
    )
    eval_sample = EvalSample("How old is thomas", [Document("Thomas is 12.")])
    eval_result = evaluator.evaluate(RAGOutput("Thomas is 12."), eval_sample)
    assert eval_result["correctness"].score is not None
    assert eval_result["correctness"].score == int(lm.query.return_value)


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_evaluation_out_interval_score_with_trow_policy(lm: MagicMock) -> None:
    """Test that with a score outside the authorized interval the evaluation throws a JudgeError."""
    min_score = 0
    max_score = 5
    lm.query.return_value = "6"
    evaluator = SimpleLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.THROW_ERROR,
    )
    eval_sample = EvalSample("How old is thomas", [Document("Thomas is 12.")])
    with pytest.raises(JudgeError):
        evaluator.evaluate(RAGOutput("Thomas is 12."), eval_sample)


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_pairwise_evaluation_in_interval_scores_with_min_score_policy(lm: MagicMock) -> None:
    """Test that with scores inside the authorized interval the output scores stay the same with min score policy."""
    min_score = 0
    max_score = 5
    lm.query.return_value = "1\n1"
    evaluator = SimpleLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    eval_sample = EvalSample(query="How old is thomas")
    rag_output_1 = RAGOutput("Thomas is 12.")
    rag_output_2 = RAGOutput("Thomas is 12.")
    eval_result_1, eval_result_2 = evaluator.evaluate_pairwise(rag_output_1, rag_output_2, eval_sample)
    assert eval_result_1["correctness"].score == 1
    assert eval_result_2["correctness"].score == 1


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_pairwise_evaluation_out_interval_scores_with_min_score_policy(lm: MagicMock) -> None:
    """Test that with a score outside the authorized interval the output score is the min score with min score policy.

    i.e test the score is correctly parsed and that the check interval check works.
    """
    min_score = 0
    max_score = 5
    lm.query.return_value = "6\n-1"
    evaluator = SimpleLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    eval_sample = EvalSample(query="How old is thomas")
    rag_output_1 = RAGOutput("Thomas is 12.")
    rag_output_2 = RAGOutput("Thomas is 12.")
    eval_result_1, eval_result_2 = evaluator.evaluate_pairwise(rag_output_1, rag_output_2, eval_sample)
    assert eval_result_1["correctness"].score == 0
    assert eval_result_2["correctness"].score == 0


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_pairwise_evaluation_float_scores_with_min_score_policy(lm: MagicMock) -> None:
    """Test that with non-digit scores the output scores are the min scores with min score policy.

    i.e test the score is correctly parsed and that the check interval check works.
    """
    min_score = 0
    max_score = 5
    lm.query.return_value = "Je m'appelle\nAntoine"
    evaluator = SimpleLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    eval_sample = EvalSample(query="How old is thomas")
    rag_output_1 = RAGOutput("Thomas is 12.")
    rag_output_2 = RAGOutput("Thomas is 12.")
    eval_result_1, eval_result_2 = evaluator.evaluate_pairwise(rag_output_1, rag_output_2, eval_sample)
    assert eval_result_1
    assert eval_result_2["correctness"].score == 0


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_pairwise_evaluation_too_many_scores_with_min_score_policy(lm: MagicMock) -> None:
    """Test that with too many score the output scores are the min scores with min score policy.

    i.e test the score is correctly parsed and that the check interval check works.
    """
    min_score = 0
    max_score = 5
    lm.query.return_value = "3\n4\n5"
    evaluator = SimpleLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    eval_sample = EvalSample(query="How old is thomas")
    rag_output_1 = RAGOutput("Thomas is 12.")
    rag_output_2 = RAGOutput("Thomas is 12.")
    eval_result_1, eval_result_2 = evaluator.evaluate_pairwise(rag_output_1, rag_output_2, eval_sample)
    assert eval_result_1["correctness"].score == 0
    assert eval_result_2["correctness"].score == 0


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_pairwise_evaluation_not_enough_scores_with_min_score_policy(lm: MagicMock) -> None:
    """Test that with not enough scores the output scores are the min scores with min score policy.

    i.e test the score is correctly parsed and that the check interval check works.
    """
    min_score = 0
    max_score = 5
    lm.query.return_value = "3"
    evaluator = SimpleLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    eval_sample = EvalSample(query="How old is thomas")
    rag_output_1 = RAGOutput("Thomas is 12.")
    rag_output_2 = RAGOutput("Thomas is 12.")
    eval_result_1, eval_result_2 = evaluator.evaluate_pairwise(rag_output_1, rag_output_2, eval_sample)
    assert eval_result_1["correctness"].score == 0
    assert eval_result_2["correctness"].score == 0


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_pairwise_evaluation_too_many_score_with_min_score_policy(lm: MagicMock) -> None:
    """Test that with only the second score incorrect both are the min score with min score policy.

    i.e test the score is correctly parsed and that the check interval check works.
    """
    min_score = 0
    max_score = 5
    lm.query.return_value = "3\nAntoine"
    evaluator = SimpleLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    eval_sample = EvalSample(query="How old is thomas")
    rag_output_1 = RAGOutput("Thomas is 12.")
    rag_output_2 = RAGOutput("Thomas is 12.")
    eval_result_1, eval_result_2 = evaluator.evaluate_pairwise(rag_output_1, rag_output_2, eval_sample)
    assert eval_result_1["correctness"].score == 0
    assert eval_result_2["correctness"].score == 0
