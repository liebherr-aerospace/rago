from pathlib import Path

import optuna

from rago.utils import PATH_DEFAULT_DATA_DIR
from scripts.seeds import load_seeds

if __name__ == "__main__":
    data_path = Path(PATH_DEFAULT_DATA_DIR)
    seed = load_seeds()
    sampler = optuna.samplers.RandomSampler(seed = seed)
