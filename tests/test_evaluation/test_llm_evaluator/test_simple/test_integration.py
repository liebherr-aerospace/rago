"""Test the llm evaluator."""

from rago.data_objects import Document, EvalSample, RAGOutput
from rago.eval import PolicyOnError, SimpleLLMEvaluator
from rago.model.wrapper.llm_agent.langchain import LangchainLLMAgent


def test_llm_judge_score_is_valid(langchain_llm_agent: LangchainLLMAgent) -> None:
    """Test that the output score of evaluation is in the interval with min score policy."""
    min_score = 0
    max_score = 5

    evaluator = SimpleLLMEvaluator(
        judge=langchain_llm_agent,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    eval_sample = EvalSample("How old is thomas", [Document("Thomas is 12.")])
    eval_result = evaluator.evaluate(RAGOutput("Thomas is 12."), eval_sample)
    assert eval_result["correctness"].score is not None
    assert min_score <= eval_result["correctness"].score <= max_score


def test_pairwise_evaluation_in_interval_scores_with_min_score_policy(langchain_llm_agent: LangchainLLMAgent) -> None:
    """Test that the output scores of pairwise evaluation are in the interval with min score policy."""
    min_score = 0
    max_score = 5
    evaluator = SimpleLLMEvaluator(
        judge=langchain_llm_agent,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    eval_sample = EvalSample(query="How old is thomas")
    rag_outputs = [RAGOutput("Thomas is 12."), RAGOutput("Thomas is 13.")]
    eval_results = evaluator.evaluate_n_wise(rag_outputs, eval_sample)
    assert all(e["correctness"].score is not None for e in eval_results)
    assert all(min_score <= e["correctness"].score <= max_score for e in eval_results)
