#!/usr/bin/env python
"""Defines an optim experiment class to run experiments."""

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import argparse
import json
from pathlib import Path
from typing import cast

from optuna.samplers import TPESampler

from rago.dataset import QADatasetLoader, RAGDataset
from rago.eval import BertScore
from rago.model.wrapper.rag.base import RAG
from rago.optimization.manager import OptimParams, SimpleDirectOptunaManager
from rago.optimization.search_space.llm_config_space import OllamaLLMConfigSpace
from rago.optimization.search_space.param_space import CategoricalParamSpace
from rago.optimization.search_space.rag_config_space import RAGConfigSpace
from rago.optimization.search_space.reader_config_space import LangchainReaderConfigSpace
from rago.optimization.search_space.retriever_config_space import RetrieverConfigSpace

from rago.experiment.base import Experiment


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
        self.datasets_dict = cast(
            RAGDataset,
            QADatasetLoader.load_dataset(RAGDataset, "crag"),
        ).split_dataset([0.1, 0.9], split_names=["train", "valid"], seed=0)
        self.evaluator = BertScore()
        config_space = RAGConfigSpace(
            retriever_space=RetrieverConfigSpace(
                retriever_type_name=CategoricalParamSpace(choices=["VectorIndexRetriever", "BM25Retriever"]),
            ),
            reader_space=LangchainReaderConfigSpace(
                OllamaLLMConfigSpace(model_name=CategoricalParamSpace(choices=["ministral-3:8b", "qwen3:8b", "gemma3:12b","gpt-oss:20b"])),
            ),
        )
        self.optimizer = SimpleDirectOptunaManager.from_dataset(
            params=params,
            dataset=self.datasets_dict["train"],
            evaluator=self.evaluator,
            metric_name="bert_score_f1",
            config_space=config_space,
            sampler=sampler,
        )
        study = self.optimizer.optimize()
        best_trial = study.best_trial
        best_rag = self.optimizer.sample_rag(best_trial)
        best_rag_eval_score = self.evaluate(best_rag)
        with Path(f"experiments/{experiment_name}/best_score.json").open("w") as f:
            json.dump({"eval_score": best_rag_eval_score}, f)

    def evaluate(self, rag: RAG) -> float:
        """Evaluate a RAG on the validation dataset.

        :param rag: The RAG model to evaluate.
        :type rag: RAG
        :return: The evaluation score
        :rtype: float
        """
        score = 0.0
        for n, test_sample in enumerate(self.datasets_dict["valid"].samples):
            score_eval = self.optimizer.eval(test_sample, rag)
            score = self.evaluator.update_avg_score(score, score_eval, n)
        return score


def main():
    """Main entry point with argument parsing."""
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
    
    print(f"🚀 Starting experiment: {args.experiment_name}")
    experiment = OptimExperiment()
    experiment.run(experiment_name=args.experiment_name)
    print(f"✅ Experiment '{args.experiment_name}' completed!")


if __name__ == "__main__":
    main()
