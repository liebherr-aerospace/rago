"""Test the Optuna Experiments Repository."""

from pathlib import Path
from unittest.mock import patch

import pytest

from rago.optimization.repository.optuna_experiments_repository import OptunaExperimentRepository


@pytest.fixture
def experiment001_repository() -> OptunaExperimentRepository:
    """Define the Base class for the experiment named 001."""
    path_experiments = "test_experiments"
    experiment_name = "test_experiment_0001"
    experiment_repository = OptunaExperimentRepository(path_experiments, experiment_name)
    return experiment_repository


def test_first_experiment_directory_creation(experiment001_repository: OptunaExperimentRepository) -> None:
    """Test if experiment directories are well defined at the first time."""
    assert experiment001_repository.path_experiments == "test_experiments"
    assert experiment001_repository.experiment_name == "test_experiment_0001"
    assert experiment001_repository.path_experiment == Path("test_experiments/test_experiment_0001")


def test_add_new_experiment_directory() -> None:
    """Test add a new experiment directory."""
    path_experiments = "test_experiments"
    experiment_name = "test_experiment_0002"
    experiment_repository = OptunaExperimentRepository(path_experiments, experiment_name)
    assert Path(path_experiments).exists()
    assert experiment_repository.path_experiment == Path("test_experiments/test_experiment_0002")


def test_get_storage_name(experiment001_repository: OptunaExperimentRepository) -> None:
    """Test if we get the right name of the DB for storage."""
    assert experiment001_repository.get_storage_name() == "sqlite:///test_experiments/test_experiment_0001/study.db"


def test_load_results_database_not_found(experiment001_repository: OptunaExperimentRepository) -> None:
    """Test load results when result database found in the directory."""
    with (
        patch("pathlib.Path.exists", return_value=False),
        pytest.raises(RuntimeError, match="Error loading the study: no result database found\\."),
    ):
        experiment001_repository.load_results()

