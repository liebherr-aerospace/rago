"""Test the CoT llm evaluator."""

from rago.data_objects import Document, EvalSample, RAGOutput
from rago.eval import CoTLLMEvaluator, PolicyOnError
from rago.model.wrapper.llm_agent.langchain import LangchainLLMAgent


def test_llm_judge_score_is_valid(langchain_llm_agent: LangchainLLMAgent) -> None:
    """Test that the output score of evaluation is in the interval with min score policy."""
    min_score = 0
    max_score = 5

    evaluator = CoTLLMEvaluator(
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
    evaluator = CoTLLMEvaluator(
        judge=langchain_llm_agent,
        min_score=min_score,
        max_score=max_score,
        policy_on_errors=PolicyOnError.MIN_SCORE,
    )
    eval_sample = EvalSample(query="How old is thomas")
    rag_output_1 = RAGOutput("Thomas is 12.")
    rag_output_2 = RAGOutput("Thomas is 13.")
    eval_result_1, eval_result_2 = evaluator.evaluate_pairwise(rag_output_1, rag_output_2, eval_sample)
    assert eval_result_1["correctness"].score is not None
    assert eval_result_2["correctness"].score is not None
    assert min_score <= eval_result_1["correctness"].score <= max_score
    assert min_score <= eval_result_2["correctness"].score <= max_score
