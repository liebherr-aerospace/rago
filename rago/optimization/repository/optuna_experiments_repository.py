"""Manage the Optimization experiments from Optuna.

The saving process is already managed by optuna during the optimization process.
"""

from pathlib import Path

import optuna

from rago.data_objects.rag_config import RAGConfig, RAGParam


class OptunaExperimentRepository:
    """Class to manage saving and loading Optuna optimization experiments using SQLite.

    :param path_experiments: The path to the directory for the experiments.
    :type path_experiments: str
    :param experiment_name: The name of the experiment directory.
    :type experiment_name: str
    """

    path_experiments: str
    experiment_name: str

    def __init__(self, path_experiments: str = "experiments", experiment_name: str = "experiment_001") -> None:
        """Initialize the Optuna Result Repository.

        :param path_experiments: The path to the experiment directories.
        :type path_experiments: str
        :param experiment_name: The name of the current experiment.
        :type experiment_name: str
        """
        self.path_experiments = path_experiments
        self.experiment_name = experiment_name
        self.path_experiment = Path(path_experiments) / experiment_name
        self.path_experiment.mkdir(parents=True, exist_ok=True)
        self.db_file_path = self.path_experiment / "study.db"

    def get_storage_name(self) -> str:
        """Generate the storage name for the SQLite database.

        :return: The storage name for the SQLite database.
        :rtype: str
        """
        return f"sqlite:///{self.db_file_path}"

    def load_results(self) -> optuna.study.Study:
        """Load the results of an Optuna study from a SQLite database.

        :return: The loaded study object.
        :rtype: optuna.study.Study
        :raise RuntimeError if no result databse found in the directory.
        """
        storage_name = self.get_storage_name()
        if Path(self.db_file_path).exists():
            study = optuna.load_study(study_name=self.experiment_name, storage=storage_name)
            return study
        error_msg = "Error loading the study: no result database found."
        raise RuntimeError(error_msg)

    def get_n_best_trials(self, number: int) -> list[optuna.trial.FrozenTrial]:
        """Get the top N best trials from an Optuna study.

        :param number: The number of top trials to retrieve.
        :type number: int
        :return: A list of the top N best trials.
        :rtype: list[optuna.trial.FrozenTrial]
        """
        study = self.load_results()
        completed_trials = [trial for trial in study.trials if trial.value is not None]
        sorted_trials = sorted(
            completed_trials,
            key=lambda trial: (float(trial.value) if trial.value is not None else float("-inf")),
            reverse=True,
        )
        return sorted_trials[:number]

    def convert_trial_to_rag_config(self, trial: optuna.trial.FrozenTrial) -> RAGConfig:
        """Convert an Optuna trial to a RAGConfig object.

        :param trial: The Optuna trial to convert.
        :type trial: optuna.trial.FrozenTrial
        :return: The corresponding RAGConfig object.
        :rtype: RAGConfig
        """
        params = {RAGParam(name, value) for name, value in trial.params.items()}
        return RAGConfig(params=params)
