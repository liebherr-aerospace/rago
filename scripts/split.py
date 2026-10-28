from pathlib import Path

from sklearn.model_selection import train_test_split

from rago.utils import PATH_DEFAULT_DATA_DIR
from scripts.seeds import load_seeds
from scripts.utils import load_docs_from_jsonl, save_docs_to_jsonl

path_data = Path(PATH_DEFAULT_DATA_DIR)
path_source = path_data/"eval_llm_2026_with_pages/processed_eval_llm_with_page/corpus.jsonl"
path_dataset = path_data/ "eval_llm_2026_with_pages/documents"
path_dataset.mkdir(exist_ok=True)

docs = load_docs_from_jsonl(path_source)

train_docs, test_docs = train_test_split(docs, test_size=0.2, random_state = load_seeds(num_seeds=1)[0])

save_docs_to_jsonl(train_docs, path_dataset/"train.jsonl")
save_docs_to_jsonl(test_docs, path_dataset/"test.jsonl")


