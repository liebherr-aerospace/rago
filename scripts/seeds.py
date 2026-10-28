from pathlib import Path
from typing import Optional

import numpy as np

from rago.utils import PATH_DEFAULT_DATA_DIR


def generate_seeds(
    num_seeds: int = int(1e3),
    path: Optional[str | Path] = None,
    max_seed_val: int = 2**32 - 1,
    seed: Optional[int] = None,
    overwrite_if_exist: bool = False,
) -> None:
    path = Path(PATH_DEFAULT_DATA_DIR)/ "seeds.csv" if path is None else Path(path)
    if path.is_file() and not overwrite_if_exist:
        raise FileExistsError(path)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(max_seed_val, size=(num_seeds,))
    np.savetxt(path, seeds, delimiter=", ", fmt="% s")


def load_seeds(
        path: Optional[str| Path] = None,
        num_seeds: Optional[int] = None,
        generation_seed: Optional[int] = None,
        ) -> list[int]:
    path = Path(PATH_DEFAULT_DATA_DIR)/ "seeds.csv" if path is None else Path(path)
    if not path.exists():
        generate_seeds(path = path, seed = generation_seed)
    seeds = np.loadtxt(path, delimiter=",")
    return seeds.astype(int).tolist()[:num_seeds]
