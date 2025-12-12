"""Define all the default prompts used throughout the optimization cycle and the prompt config dataclass.

i.e. In Dataset Generation, Judge, RAG.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic.dataclasses import dataclass

## Dataset Generation
DEFAULT_DATASET_GENERATION_PROMPT = """Ask {number_of_question} question(s) about the following document:
{documents}
You must put one question per line.
Your questions:"""

#### QA Prompts ####
CONTEXT_TEMPLATE = """
To answer the question you may use the following context information:
{retrieved_documents}
"""

LEGACY_AMBIGQA_SYSTEM_PROMPT = """
The following question admits multiple short answers (keywords) depending on the interpretation:
{query_str}
{context_str}
Give all the possible answers to the question. Each answer must be keyword only on a new line.
"""

DEFAULT_SYSTEM_PROMPT = """
Answer the following user question:
{query_str}
{context_str}
"""

AMBIGQA_SYSTEM_PROMPT = """
You need to answer the following question:
{query_str}
This question is ambiguous this means it has multiple interpretations.
{context_str}
Since the question is ambiguous you need to answer all its possible interpretations.
Each answer must be  very short (keywords only) on a separate line.
"""

#### Evaluation ####
SOURCE_CONTEXT_PROMPT = """
The correct answers are in the following set of documents:
{source_context}
"""
## Simple ##
DEFAULT_EVAL_PROMPT = """
You need to evaluate the answer to the following query:
{query}
{source_context_prompt}
The candidate answer is:
{candidate_answer}
Your evaluation must be a score between {min_score} and {max_score}.
Only output a score without comments before or after.
"""
DEFAULT_PAIRWISE_EVAL_PROMPT = """
You need to evaluate two answers to the following query:
{query}
{source_context_prompt}
The candidate answer 1 is:
{candidate_answer_1}
The candidate answer 2 is:
{candidate_answer_2}
Your evaluation must be a score between {min_score} and {max_score}.
Only output the two scores on separate lines without comments before or after.
Generate the score in the same order as the queries.
"""
## CoT ##
DEFAULT_EXPLANATION_TAG = """Reasoning:"""

# Direct
DEFAULT_COT_SCORE_TAG = """Correctness:"""

DEFAULT_COT_EVAL_PROMPT = """
You need to evaluate an answer to the following query:
{query}
This question is ambiguous this means it has multiple interpretations.
You need to evaluate out of all the answers to the different interpretations
given in the answer the amount that are correct.
If the answer is out of scope or bad give it the bad score.
{source_context_prompt}
The candidate answer you need to evaluate is:
{candidate_answer}
Your evaluation must be a score between {min_score} and {max_score}.
Give your reasoning before scoring the answer using the following template:
{explanation_tag}
Your reasoning here.
{score_tag}
A single digit score between {min_score} (worst) and {max_score} (best).
It is important to only put a single digit here without any comments.
Finally it is important to use the exact same template as it will be parsed automatically
"""

# Pairwise
DEFAULT_SCORE_1_TAG = """Correctness answer 1:"""

DEFAULT_SCORE_2_TAG = """Correctness answer 2:"""

DEFAULT_COT_PAIRWISE_EVAL_PROMPT = """
You need to Evaluate two answers to the following query:
{query}
This question is ambiguous this means it has multiple interpretations.
You need to evaluate out of all the answers to the different interpretations
the amount that are correct in each answers.
If the answer is out of scope or bad it should receive a bad score.
{source_context_prompt}
The candidate answer 1 is:
{candidate_answer_1}
The candidate answer 2 is:
{candidate_answer_2}
Your evaluation must be a score between {min_score} and {max_score}.
Give your reasoning before answering.
Use the following template to answer:
{explanation_tag}
Your reasoning here.
{score_1_tag}
A single digit score between for answer 1 between {min_score} and {max_score}.
{score_2_tag}
A single digit score between for answer 2 between {min_score} and {max_score}.
It is important to only write a single digit in answers'score section without any comments.
Both answers should always be scored.
Finally it is important to use the exact same template as it will be parsed automatically.
"""


@dataclass
class PromptConfig:
    """Define the configuration to use for a prompt."""

    system_message: Optional[str] = Field(default=DEFAULT_SYSTEM_PROMPT)
    dataset_name: Optional[str] = Field(default=None)
    context_template: str = Field(default=SOURCE_CONTEXT_PROMPT)

    def __post_init__(self) -> None:
        """Modify the parameters."""
        if self.system_message is None and self.dataset_name is None:
            self.system_message = DEFAULT_SYSTEM_PROMPT
