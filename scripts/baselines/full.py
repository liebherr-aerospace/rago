"""Run multiple experiments sequentially."""
from scripts.baselines.optim import OPTIM_CONFIFGS, run_optim
from scripts.baselines.random_rags import run_random_rags
from scripts.seeds import load_seeds

if __name__ == "__main__":
    seeds = load_seeds(num_seeds=10)[7:]
    run_random_rags(num_seeds = None, seeds = seeds)
    for conf in OPTIM_CONFIFGS:
        run_optim(conf)
