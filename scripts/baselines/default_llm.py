from pathlib import Path
from typing import cast

from rago.dataset import RAGDataset
from rago.utils import PATH_DEFAULT_DATA_DIR, PATH_PROJECT
from scripts.eval import eval_model_seeds

from .utils import get_judge, get_llm

if __name__ == "__main__":

    path_data = Path(PATH_DEFAULT_DATA_DIR)
    path_augmented = path_data/"eval_llm_2026_with_pages/augmented"
    path_default_llm = Path(PATH_PROJECT, "experiments", "baselines", "default_qwen3.5_2b")
    dataset = cast("dict[str, RAGDataset]", RAGDataset.load_dataset(path_augmented))["test"]
    num_samples = 600
    eval_model_seeds(
        get_llm,
        get_judge,
        dataset,
        num_samples,
        save_dir = path_default_llm,
        num_seeds = 5)
