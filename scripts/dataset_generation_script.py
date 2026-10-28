from pathlib import Path
from typing import Optional

import langchain_core.documents
from transformers import set_seed

from rago.data_objects import Document, EvalSample
from rago.dataset import RAGDataset
from rago.dataset.generator import SimpleDatasetGenerator
from rago.model.configs.llm_config.langchain import LangchainOllamaConfig
from rago.model.wrapper.llm_agent.llm_agent_factory import LLMAgentFactory
from rago.utils import PATH_DEFAULT_DATA_DIR, get_document_hash
from scripts.seeds import load_seeds
from scripts.utils import load_docs_from_jsonl, save_docs_to_jsonl

path_data = Path(PATH_DEFAULT_DATA_DIR)

ANSWERABLE_DATASET_GENERATION_PROMPT ="""
Ask {number_of_question} question about that can be answered only by reading the following documents:
{documents}
You must put one question per line.
Your questions:"""

PROMPT_REPONSE_REPONDABLE ="""
Si le document ne contient pas assez d'information pour poser une question factuelle intéressante dont la réponse est dans le document, écris uniquement NaN.
Sinon tu dois poser une question courte pour laquelle la reponse est dans le document.
Cette question sera utilisé pour evaluer un systeme de rag et ne doit donc pas contenir de reference implicites.
Le document est:
{documents}

Contraintes :
- La question doit être indépendante du document : elle doit rester compréhensible même sans voir le document.
- La question doit porter sur le sujet principal du document (e.g armement, drone, industry), pas sur sa structure, son titre, sa page, sa section ou sa traduction.
- La question ne doit pas contenir de reference implicites au document comme: le document, le texte, le rapport, ce tableau etc.
Ta question ou NaN:
"""

def augment(
        path_documents: Path,
        path_augmentation: Path,
        path_filtered_chunks: Path,
        model_name: str = "qwen3.5:9b",
        seed: Optional[int] = None,
        ) -> None:
    """Create a synthetic question-corpus dataset by generated questions from a random subset of documents."""
    if seed is not None:
        set_seed(seed)
    chunks = load_docs_from_jsonl(path_documents)
    rago_chunks =  [Document(c.page_content, get_document_hash(c.page_content)) for c in chunks]
    print("Number of Chunks: ", len(rago_chunks))
    llm_config = LangchainOllamaConfig(
        model_name=model_name,
        temperature=1.0,
        top_p=1.0,
        top_k=20,
        repeat_penalty = 1.0,
        max_new_tokens = 256,
        seed = seed,
        )
    llm_agent = LLMAgentFactory.make(llm_config)
    dataset_generator = SimpleDatasetGenerator(
        generator=llm_agent,
        generation_prompt=PROMPT_REPONSE_REPONDABLE,
    )
    synthetic_dataset = dataset_generator.generate_dataset(rago_chunks)
    filtered_chunks: list[langchain_core.documents.Document] = []
    filtered_samples: list[EvalSample] = []
    corpus: dict[str, Document] = {}
    for s, chunk in zip(synthetic_dataset.samples, chunks):
        if s.query.strip().strip('"').strip("'").lower() == "nan":
            continue
        filtered_samples.append(s)
        corpus[s.context[0].id] = s.context[0]
        filtered_chunks.append(chunk)

    filter_dataset= RAGDataset(filtered_samples, corpus)
    RAGDataset.save_to_disk(filter_dataset, str(path_augmentation))
    save_docs_to_jsonl(filtered_chunks, path_filtered_chunks)


if __name__ == "__main__":

    seed = load_seeds(num_seeds=1)[0]
    path_with_pages = path_data / "eval_llm_2026_with_pages"
    path_chunks = path_with_pages/ "chunks"
    path_filtered_chunks = path_with_pages / "filtered_chunks"
    path_filtered_chunks.mkdir(exist_ok=True)
    path_augmented = path_with_pages / "augmented"
    path_augmented.mkdir(exist_ok=True)
    for fpath in path_chunks.iterdir():
        print(f"Augmenting {fpath.stem} split...")
        augment(
            fpath,
            path_augmented/ fpath.stem,
            path_filtered_chunks/ fpath.name,
            seed = seed,
        )

