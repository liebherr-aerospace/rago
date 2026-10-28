import json
from pathlib import Path
from typing import Any

import langchain_core.documents
import optuna
from tqdm import tqdm

from rago.data_objects import RAGOutput, RetrievedContext
from rago.model.wrapper.rag.base import RAG, RAGConfig
from rago.optimization.manager import SimplePairWiseOptunaManager
from rago.utils import PATH_DEFAULT_DATA_DIR, get_document_hash
from scripts.optim import get_optimizer
from scripts.seeds import load_seeds
from scripts.utils import load_docs_from_jsonl

TEAM_NAME = "B4-RAG"

def get_retrieved(rank: int, retrieved: RetrievedContext, corpus:  dict[str, langchain_core.documents.Document]) -> dict[str, Any]:
    doc_id = get_document_hash(retrieved.text)
    document = corpus[doc_id]
    return {
        "rank":rank,
        "doc_name": document.metadata["title"],
        "page": document.metadata["page"],
        "metadata":{

        },

    }
def get_retrieved_results(out: RAGOutput, corpus: dict[str, langchain_core.documents.Document]) -> list[dict[str, Any]]:
    return [get_retrieved(idx+1, c, corpus) for idx, c in enumerate(out.retrieved_context)]

def get_submission_results(
        outputs: list[RAGOutput],
        queries: list[dict[str, Any]],
        corpus: dict[str, langchain_core.documents.Document]) -> list:

    return [
        {
            "qid": q["qid"],
            "question": q["question"],
            "retrieved": get_retrieved_results(out, corpus),
            "answer": out.answer,
            "metadata": q["metadata"],
        }
        for out, q in zip(outputs, queries)
    ]

def generate_submission(
        outputs: list[RAGOutput],
        queries: list[dict[str, Any]],
        corpus: dict[str, langchain_core.documents.Document],
        model_name: str,
        dir_path: Path,
        task: int = 1,
        run: int = 1,
        ) -> None:
    submission = {
        "run_id": run,
        "parameters":{
            "model_name": model_name,
        },
        "results": get_submission_results(outputs, queries, corpus),
    }
    submission_name = f"{TEAM_NAME}_tache{task}_run{run}.json"
    submission_path = dir_path / submission_name
    with submission_path.open("w",encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

def load_queries(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        queries  = json.load(f)
    return queries["results"]

def get_corpus(path: Path) -> dict[str, langchain_core.documents.Document]:
    corpus: dict[str, langchain_core.documents.Document] = {}
    for fpath in path.iterdir():
        docs = load_docs_from_jsonl(fpath)
        for d in docs:
            doc_id = get_document_hash(d.page_content)
            if doc_id not in corpus:
                corpus[doc_id] = d
    return corpus

def run_submission(
        trial: optuna.trial.FrozenTrial,
        optimizer: SimplePairWiseOptunaManager,
        queries: list[dict[str, Any]],
        corpus: dict[str, langchain_core.documents.Document],
        model_name: str,
        submission_dir: Path,
        task: int,
        run: int,
)-> None:

    config: RAGConfig = optimizer.config_space.sample(trial)
    inputs_chunks = [doc.page_content for doc in corpus.values()]
    rag = RAG.make(
                rag_config=config,
                prompt_config=optimizer.prompt_config,
                inputs_chunks=inputs_chunks,
            )
    questions = [query["question"] for query in queries]
    outputs = [rag.get_output(q) for q in tqdm(questions)]
    generate_submission(
        outputs = outputs,
        queries = queries,
        corpus= corpus,
        model_name = model_name,
        dir_path = submission_dir,
        task = task,
        run = run,
    )
def submissions(
        query_path: Path,
        path_filtered_corpus: Path,
        path_augmented: Path,
        model_name: str,
        experiment_name: str,
        submission_dir: Path,
        task: int = 1,
        ):
    seed = load_seeds(num_seeds=1)[0]
    submission_dir.mkdir(exist_ok=True)
    queries = load_queries(query_path)
    corpus = get_corpus(path_filtered_corpus)

    optimizer = get_optimizer(path_augmented, seed, model_name, experiment_name)
    best_trials = optimizer.get_n_best_trials(number=5)
    for idx, trial in enumerate(best_trials):
        run_submission(
            trial,
            optimizer,
            queries,
            corpus,
            model_name,
            submission_dir,
            task,
            idx+1,
        )


if __name__ == "__main__":
    path_data = Path(PATH_DEFAULT_DATA_DIR)
    path_train = path_data / "eval_llm_2026_with_pages"
    submissions(
        query_path= path_data/"test_evallm_2026_raw"/"20260430_task1_test_query.json",
        path_filtered_corpus = path_train/"chunks",
        path_augmented = path_train/"augmented",
        model_name = "qwen3.5:2b",
        experiment_name= "optim",
        submission_dir= path_data /"submissions",
        task=1,
    )
