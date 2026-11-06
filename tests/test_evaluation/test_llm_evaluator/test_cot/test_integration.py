"""Test the CoT llm evaluator."""

from rago.data_objects import GenerationOutput
from rago.eval import CoTLLMEvaluator, DependentEvaluatorReference, PolicyOnError
from rago.model.wrapper.llm_agent.langchain import LangchainLLMAgent


def test_llm_judge_score_is_valid(
        langchain_llm_agent: LangchainLLMAgent,
        dependent_reference: DependentEvaluatorReference) -> None:
    """Test that the output score of evaluation is in the interval with min score policy."""
    min_score = 0
    max_score = 5

    evaluator = CoTLLMEvaluator(
        judge=langchain_llm_agent,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    eval_result = evaluator.evaluate(GenerationOutput("Thomas is 12."), dependent_reference)
    assert eval_result.score is not None
    assert min_score <= eval_result.score <= max_score


def test_n_wise_evaluation_in_interval_scores_with_min_score_policy(
        langchain_llm_agent: LangchainLLMAgent,
        dependent_reference: DependentEvaluatorReference) -> None:
    """Test that the output scores of n wise evaluation are in the interval with min score policy."""
    min_score = 0
    max_score = 5
    evaluator = CoTLLMEvaluator(
        judge=langchain_llm_agent,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    rag_outputs = [GenerationOutput("Thomas is 12."), GenerationOutput("Thomas is 13.")]
    eval_results = evaluator.evaluate_n_wise(rag_outputs, dependent_reference)
    assert all(e.score is not None for e in eval_results)
    assert all(min_score <= e.score <= max_score for e in eval_results)
