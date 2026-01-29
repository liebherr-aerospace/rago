"""Defines an optim experiment class to run experiments."""

import argparse
from typing import cast

import urllib3
from optuna.samplers import TPESampler

from rago.dataset import RAGDataset
from rago.eval import BertScore, SimilarityScore, SimpleLLMEvaluator
from rago.experiment.base import Experiment
from rago.optimization.manager import OptimParams, SimpleDirectOptunaManager
from rago.optimization.search_space.llm_config_space import OllamaLLMConfigSpace
from rago.optimization.search_space.param_space import CategoricalParamSpace
from rago.optimization.search_space.rag_config_space import RAGConfigSpace
from rago.optimization.search_space.reader_config_space import LangchainReaderConfigSpace
from rago.optimization.search_space.retriever_config_space import RetrieverConfigSpace

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class OptimExperiment(Experiment):
    """Optim Experiment class to carry out optim experiments."""

    def run(self, experiment_name: str = "crag_experiment") -> None:
        """Run an optim experiment."""
        params = OptimParams(
            experiment_name=experiment_name,
            n_startup_trials=50,
            n_iter=1000,
        )
        sampler = TPESampler(n_startup_trials=params.n_startup_trials)
        crag_dataset = cast("RAGDataset", RAGDataset.load_dataset("crag"))
        self.datasets_dict = crag_dataset.split_dataset([0.1, 0.9], split_names=["train", "test"], seed=0)
        self.evaluator = BertScore()
        self.test_evaluators = [self.evaluator, SimilarityScore(), SimpleLLMEvaluator.make()]
        config_space = RAGConfigSpace(
            retriever_space=RetrieverConfigSpace(
                retriever_type_name=CategoricalParamSpace(choices=["VectorIndexRetriever", "BM25Retriever"]),
            ),
            reader_space=LangchainReaderConfigSpace(
                OllamaLLMConfigSpace(
                    model_name=CategoricalParamSpace(
                        choices=["ministral-3:8b", "qwen3:8b", "gemma3:12b", "gpt-oss:20b"],
                    ),
                ),
            ),
        )
        self.optimizer = SimpleDirectOptunaManager(
            params=params,
            datasets=self.datasets_dict,
            optim_evaluator=self.evaluator,
            optim_metric_name="bert_score_f1",
            test_evaluators=self.test_evaluators,
            config_space=config_space,
            sampler=sampler,
        )
        self.optimizer.run_experiment()


def main() -> None:
    """Run experiment with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run RAG optimization experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="crag_experiment",
        help="Name of the experiment (creates experiments/<name>/ directory)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1000,
        help="Total number of optimization trials",
    )
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=50,
        help="Number of random startup trials for TPE sampler",
    )

    args = parser.parse_args()

    experiment = OptimExperiment()
    experiment.run(experiment_name=args.experiment_name)


if __name__ == "__main__":
    main()
