"""Evaluate model and save a data."""
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, TypedDict, overload

from pydantic import TypeAdapter
from tqdm import tqdm

from rago.data_objects import Metric, ModelOutput
from rago.dataset import RAGDataset
from rago.eval.base import BaseEvaluator
from rago.model.configs.tunable_model_config import TunableModelConfig
from rago.model.wrapper import TunableModel
from scripts.seeds import load_seeds


class DatasetEval[ModelOutputType: ModelOutput](TypedDict):
    """Evaluation data of a model."""

    config: dict[str, Any]
    model_outputs: list[ModelOutputType]
    evals: list[dict[str, Metric]]

def get_seeds(
        num_seeds: int | None,
        seeds: list[int] | None,
        seed_path: str | Path) -> list[int]:
    """Get the seeds."""
    match num_seeds, seeds:
        case int(), None:
            return load_seeds(path = seed_path, num_seeds=num_seeds)
        case None, list():
            return seeds
        case _, _:
            msg = "Incompatible values for num_seeds and seeds."
            raise TypeError(msg)
@overload
def eval_model_seeds[ModelOutputType: ModelOutput, TunableModelConfigType: TunableModelConfig](
        make_model_from_seed: Callable[[int], TunableModel[ModelOutputType, TunableModelConfigType]],
        make_judge:Callable[[int], BaseEvaluator[ModelOutputType]],
        dataset: RAGDataset,
        num_samples: int,
        save_dir: str | Path,
        num_seeds: int = 5,
        seeds: None = None,
        seed_path: Optional[str | Path] = None,
        disable_progress_bar: bool = False,  # noqa: FBT001, FBT002
        ) -> None:...
@overload
def eval_model_seeds[ModelOutputType: ModelOutput, TunableModelConfigType: TunableModelConfig](
        make_model_from_seed: Callable[[int], TunableModel[ModelOutputType, TunableModelConfigType]],
        make_judge:Callable[[int], BaseEvaluator[ModelOutputType]],
        dataset: RAGDataset,
        num_samples: int,
        save_dir: str | Path,
        num_seeds: None,
        seeds: list[int],
        seed_path: Optional[str | Path] = None,
        disable_progress_bar: bool = False,  # noqa: FBT001, FBT002
        ) -> None:...
@overload
def eval_model_seeds[ModelOutputType: ModelOutput, TunableModelConfigType: TunableModelConfig](
        make_model_from_seed: Callable[[int], TunableModel[ModelOutputType, TunableModelConfigType]],
        make_judge:Callable[[int], BaseEvaluator[ModelOutputType]],
        dataset: RAGDataset,
        num_samples: int,
        save_dir: str | Path,
        num_seeds: int | None = 5,
        seeds: list[int] | None = None,
        seed_path: Optional[str | Path] = None,
        disable_progress_bar: bool = False,  # noqa: FBT001, FBT002
        ) -> None:...
def eval_model_seeds[ModelOutputType: ModelOutput, TunableModelConfigType: TunableModelConfig](
        make_model_from_seed: Callable[[int], TunableModel[ModelOutputType, TunableModelConfigType]],
        make_judge:Callable[[int], BaseEvaluator[ModelOutputType]],
        dataset: RAGDataset,
        num_samples: int,
        save_dir: str | Path,
        num_seeds: int | None = 5,
        seeds: list[int] | None = None,
        seed_path: Optional[str | Path] = None,
        disable_progress_bar: bool = False,  # noqa: FBT001, FBT002
        ) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    seeds = get_seeds(num_seeds, seeds, seed_path)
    for s in seeds:
        ds_subsample: RAGDataset = dataset.sample(num_samples, seed = s)
        model = make_model_from_seed(s)
        judge = make_judge(s)
        eval_model(model, judge, ds_subsample, save_dir/f"{s}.json", disable_progress_bar)


def eval_model[ModelOutputType: ModelOutput, TunableModelConfigType: TunableModelConfig](
    model: TunableModel[ModelOutputType, TunableModelConfigType],
    judge:BaseEvaluator[ModelOutputType],
    dataset: RAGDataset,
    save_path:  str | Path,
    disable_progress_bar: bool = False,# noqa: FBT001, FBT002
)-> None:
    """Evaluate a model on dataset."""
    dataset_eval: list[Metric] = []
    model_outputs: list[ModelOutputType] = []
    for sample in tqdm(dataset.samples, disable=disable_progress_bar):
        model_output = model.get_output(sample.query)
        sample_eval = judge.evaluate(model_output, sample)
        dataset_eval.append(sample_eval)
        model_outputs.append(model_output)
    with save_path.open("w", encoding="utf-8") as f:
        data = DatasetEval[ModelOutputType](
            config=model.config.convert_to_dict(),
            evals= dataset_eval,
            model_outputs=model_outputs,
            )
        json_str= TypeAdapter(DatasetEval[ModelOutputType]).dump_json(data, ensure_ascii=False, indent=2).decode()
        f.write(json_str)

