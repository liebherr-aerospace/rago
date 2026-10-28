########################################################################################
# RAG
########################################################################################
PROMPT_CONTEXTE = """
Utilise le uniquement contexte suivant pour répondre à la question:
{source_context}
"""

PROMPT_SYSTEME = """
{context_str}
Réponds en francais de manière concise (1 phrase) à la question suivante:
{query_str}
"""
########################################################################################
# Eval
########################################################################################
DEFAULT_EXPLANATION_TAG = """RAISONNEMENT:"""


DEFAULT_COT_SCORE_TAG = """DECISION:"""

PROMPT_CONTEXT_EVAL = """
Voici le contexte nécessaire pour répondre à la requête :
{source_context}
"""
DEFAULT_SCORE_I_TAG_TEMPLATE = """DECISION REPONSE {i}:"""

################################### Direct #############################################
PROMPT_DIRECT_EVAL = """
Tu dois évaluer la réponse candidate à la requête suivante :
{query}

{source_context_prompt}

La réponse candidate est :
{candidate_answer}

Ton évaluation doit être 0 si la réponse est incorrecte et 1 si elle est correcte.
Une réponse est correcte si elle répond à la requête, est cohérente avec le contexte fourni, et ne contient pas d'information contradictoire ou hallucinée.

Donne ton raisonnement avant de répondre.

Utilise le format suivant pour répondre :
{explanation_tag}
Ton raisonnement concis ici en une phrase.
{score_tag}
Ton score pour la réponse (0 ou 1).

Il est important d'écrire uniquement un seul chiffre dans la sections de score de la réponse, sans aucun commentaire.
Enfin, il est important d'utiliser exactement le même format, car la réponse sera analysée automatiquement.
"""

################################### DUAL #############################################
PROMPT_DUAL_EVAL = """
Tu dois évaluer deux réponses à la requête suivante :
{query}

{source_context_prompt}

La réponse candidate 1 est :
{candidate_answer_1}

La réponse candidate 2 est :
{candidate_answer_2}

Ton évaluation doit être 0 si la réponse est incorrecte et 1 si elle est correcte.
Une réponse est correcte si elle répond à la requête, est cohérente avec le contexte fourni, et ne contient pas d'information contradictoire ou hallucinée.

Donne ton raisonnement avant de répondre.

Utilise le format suivant pour répondre :
{explanation_tag}
Ton raisonnement concis ici en une phrase.
{score_1_tag}
Ton score pour la réponse 1 (0 ou 1).
{score_2_tag}
Ton score pour la réponse 2 (0 ou 1).

Il est important d'écrire uniquement un seul chiffre dans les sections de score des réponses, sans aucun commentaire.
Les deux réponses doivent toujours être notées.
Enfin, il est important d'utiliser exactement le même format, car la réponse sera analysée automatiquement.
"""
################################### PARTIEL/REF #############################################
PROMPT_EVAL_PARTIEL = """
Tu dois évaluer la réponse candidate 2 à la requête suivante :
{query}

{source_context_prompt}

La réponse candidate 1 est :
{reference_answer}

La réponse candidate 2 est :
{candidate_answer}

Le score de la réponse candidate 1 est :
{reference_score}

Ton évaluation doit être 0 si la réponse candidate 2 est incorrecte et 1 si elle est correcte.
Une réponse candidate est correcte si elle répond à la requête, est cohérente avec le contexte fourni, et ne contient pas d'information contradictoire ou hallucinée.
Evalue uniquement la réponse candidate 2.
Donne ton raisonnement concis en une phrase avant de répondre.

Utilise le format suivant pour répondre :
{explanation_tag}
Ton raisonnement concis ici en une phrase.
{score_tag}
Ton score pour la réponse candidate 2 (0 ou 1).

Il est important d'écrire uniquement un seul chiffre dans la section de score, sans aucun commentaire.
Donne uniquement le score de la réponse candidate 2, car la réponse candidate 1 est déjà notée.
Enfin, il est important d'utiliser exactement le même format de réponse, car la réponse sera analysée automatiquement.
"""

DEFAULT_COT_PAIRWISE_EVAL_PROMPT = """
You need to Evaluate two answers to the following query:
{query}
Below is the context necessary to answer the query:
{source_context_prompt}
The candidate answer 1 is:
{candidate_answer_1}
The candidate answer 2 is:
{candidate_answer_2}
Your evaluation must be 0 if the answer is incorrect and 1 if it correct.
Give your reasoning before answering.
Use the following template to answer:
{explanation_tag}
Your reasoning here.
{score_1_tag}
Your score for answer 1 (0 or 1).
{score_2_tag}
Your score for answer 2 (0 or 1).
It is important to only write a single digit in answers'score sections without any comments.
Both answers should always be scored.
Finally it is important to use the exact same template as it will be parsed automatically.
"""

DEFAULT_COT_REFERENCE_EVAL_PROMPT = """
You need to Evaluate two answers to the following query:
{query}
Below is the context necessary to answer the query:
{source_context_prompt}
The candidate answer 1 is:
{reference_answer}
The candidate answer 2 is:
{candidate_answer}
The score of answer 1 is:
{reference_score}
Your evaluation must be 0 if the answer is incorrect and 1 if it correct.
Give your reasoning before answering.
Use the following template to answer:
{explanation_tag}
Your reasoning here.
{score_2_tag}
Your score for answer 2 (0 or 1).
It is important to only write a single digit in answers'score section without any comments.
Only give score for answer 2 as answer is already scored
Finally it is important to use the exact same template as it will be parsed automatically.
"""
