import json
from collections.abc import Iterable
from pathlib import Path

from langchain_core.documents import Document


def save_docs_to_jsonl(array:Iterable[Document], file_path:Path)->None:
    """Save a list of documents into a single jsonl (one per lines)."""
    with file_path.open("w") as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.model_dump_json() + "\n")

def load_docs_from_jsonl(file_path: Path)->list[Document]:
    docs = []
    with file_path.open("r") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            docs.append(obj)
    return docs
