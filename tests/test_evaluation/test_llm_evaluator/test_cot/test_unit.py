"""Test the CoT llm evaluator."""

from unittest.mock import MagicMock, patch

from rago.eval import CoTLLMEvaluator, PolicyOnError


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_evaluation_in_interval_score_with_min_score_policy(lm: MagicMock) -> None:
    """Test that with a score inside the interval and correct template the result is correct with min score policy.

    i.e test that it is correctly parsed.
    """
    min_score = 0
    max_score = 5

    evaluator = CoTLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    judge_reasoning = "Judge's reasoning."
    judge_score = "1"
    judge_evaluation_string = f"""
{evaluator.explanation_tag}
{judge_reasoning}
{evaluator.score_tag}
{judge_score}
"""
    eval_result = evaluator.get_evaluation_result_from_evaluation(judge_evaluation_string)
    correctness = eval_result["correctness"]
    assert correctness.score == int(judge_score)
    assert correctness.explanation == judge_reasoning


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_evaluation_out_interval_score_with_min_score_policy(lm: MagicMock) -> None:
    """Test that with a score outside the interval and correct template the result is correct with min score policy.

    i.e test that it is correctly parsed.
    """
    min_score = 0
    max_score = 5

    evaluator = CoTLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    judge_reasoning = "Judge's reasoning."
    judge_score = "7"
    judge_evaluation_string = f"""
{evaluator.explanation_tag}
{judge_reasoning}
{evaluator.score_tag}
{judge_score}
"""
    eval_result = evaluator.get_evaluation_result_from_evaluation(judge_evaluation_string)
    correctness = eval_result["correctness"]
    assert correctness.score == min_score


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_evaluation_missing_reasoning_with_min_score_policy(lm: MagicMock) -> None:
    """Test that if the reasoning is missing the result score is equal to the min score with the min score policy.

    i.e test that it is correctly parsed.
    """
    min_score = 0
    max_score = 5

    evaluator = CoTLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    judge_reasoning = "Judge's reasoning."
    judge_score = "3"
    judge_evaluation_string = f"""
{judge_reasoning}
{evaluator.score_tag}
{judge_score}
"""
    eval_result = evaluator.get_evaluation_result_from_evaluation(judge_evaluation_string)
    correctness = eval_result["correctness"]
    assert correctness.score == min_score


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_evaluation_incorrect_template_with_min_score_policy(lm: MagicMock) -> None:
    """Test that with too many score, the result score is equal to the min score with the min score policy.

    i.e test that it is correctly parsed.
    """
    min_score = 0
    max_score = 5

    evaluator = CoTLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    judge_reasoning = "Judge's reasoning."
    judge_score = "3"
    judge_evaluation_string = f"""
{evaluator.explanation_tag}
{judge_reasoning}
{evaluator.score_tag}
{judge_score}
{evaluator.score_tag}
{judge_score}
"""
    eval_result = evaluator.get_evaluation_result_from_evaluation(judge_evaluation_string)
    correctness = eval_result["correctness"]
    assert correctness.score == min_score


@patch("rago.model.wrapper.llm_agent.base.LLMAgent")
def test_get_pairwise_evaluation_result_with_correct_judge_output(lm: MagicMock) -> None:
    """Test that with too many score, the result score is equal to the min score with the min score policy.

    i.e test that it is correctly parsed.
    """
    min_score = 0
    max_score = 5
    judge_score_template = """
    {score_tag}
    {judge_score}
    """
    evaluator = CoTLLMEvaluator(
        judge=lm,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    judge_reasoning = "Judge's reasoning."
    judge_scores = ["3", "2"]
    judge_evaluation_string = f"""
{evaluator.explanation_tag}
{judge_reasoning}
{
        "\n".join(
            judge_score_template.format(
                score_tag=evaluator.score_i_tag_template.format(i=i),
                judge_score=judge_scores[i - 1],
            )
            for i in range(1, 3)
        )
    }
"""
    eval_results = evaluator.get_pairwise_evaluation_result_from_evaluation(judge_evaluation_string, num_candidates=2)
    assert [e["correctness"].score == int(judge_scores[idx]) for idx, e in enumerate(eval_results)]
