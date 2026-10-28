
from rago.eval import CoTLLMEvaluator
from rago.eval.llm_evaluator import EvalPrompts, PolicyOnError
from rago.eval.llm_evaluator.cot import EvalOutputTags
from rago.model.configs.llm_config.langchain import LangchainOllamaConfig
from rago.model.wrapper.llm_agent.llm_agent_factory import LLMAgent, LLMAgentFactory
from scripts.prompts import (
    DEFAULT_COT_SCORE_TAG,
    DEFAULT_EXPLANATION_TAG,
    DEFAULT_SCORE_I_TAG_TEMPLATE,
    PROMPT_CONTEXT_EVAL,
    PROMPT_DIRECT_EVAL,
    PROMPT_DUAL_EVAL,
    PROMPT_SYSTEME,
)


def get_llm(seed: int, name: str = "qwen3.5:2b") -> LLMAgent:
    llm_config = LangchainOllamaConfig(
            model_name=name,
            temperature=1.0,
            top_p=1.0,
            top_k=20,
            num_ctx = 10000,
            repeat_penalty = 1.0,
            max_new_tokens = 256,
            seed = seed,
            )
    prompt = PROMPT_SYSTEME.format(
        context_str = "",
        query_str = "{query}",
    )
    return LLMAgentFactory.make(llm_config, output_prompt=prompt)

def get_judge(
        seed: int,
        model_name: str = "qwen3.5:9b",
        policy_on_error: PolicyOnError = PolicyOnError.NAN_SCORE,
        evaluation_prompt: str = PROMPT_DIRECT_EVAL,
        pairwise_evaluation_prompt: str = PROMPT_DUAL_EVAL,
        source_context_prompt: str = PROMPT_CONTEXT_EVAL,
        ) -> CoTLLMEvaluator:

    llm_agent = get_llm(seed, model_name)
    eval_prompts = EvalPrompts(
            evaluation_prompt=evaluation_prompt,
            pairwise_evaluation_prompt=pairwise_evaluation_prompt,
            source_context_prompt=source_context_prompt,
        )
    tags = EvalOutputTags(
        score_tag=DEFAULT_COT_SCORE_TAG,
        explanation_tag=DEFAULT_EXPLANATION_TAG,
        score_i_tag_template=DEFAULT_SCORE_I_TAG_TEMPLATE,
    )
    return CoTLLMEvaluator(
        llm_agent,
        min_score=0,
        max_score = 1,
        policy_on_errors = policy_on_error,
        eval_prompts=eval_prompts,
        eval_output_tags=tags,
    )
