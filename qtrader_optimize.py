import argparse
import json
import numpy as np
import optuna
from qtrader_trainer import trainer_run
from datetime import datetime


def objective(trial):
    def _report_evals(value, step):
        trial.report(value, step)

        if trial.should_prune():
            raise optuna.TrialPruned()

    name = f"{trial.study.study_name}-Tr{str(trial.number).zfill(4)}"

    stats = trainer_run(
        name=name,
        iters=500,
        params={
            # ===== STATIC =====
            "invest_pct": 0.05,
            "expl_decay": 0.98,
            "n_steps_warmup": 1024 * 10,
            "n_step_update": 2,
            "model_n_layers": 2,
            "model_fl_size": 128,
            "model_shape": "cone",
            "exp_memory_size": 365 * 10_000,
            "exp_mini_batch_size": 32,
            "exp_w_inc": 5e-5,
            # ===== TUNED =====
            "expl_min": trial.suggest_float("expl_min", 0.01, 0.15, step=0.01),
            "n_steps_checkpoint": trial.suggest_int("n_steps_checkpoint", 500, 5_000, step=250),
            "exp_weighting": trial.suggest_float("exp_weighting", 0.3, 0.7, step=0.05),
            "exp_alpha": trial.suggest_float("exp_alpha", 0.3, 0.8, step=0.05),
            "model_lr": trial.suggest_categorical("model_lr", [1e-5, 5e-5, 1e-4]),
            "rl_gamma": trial.suggest_categorical("rl_gamma", [0.9, 0.95, 0.99]),
        },
        n_test=10,
        prune=_report_evals,  # {"n_tests": 7, "n_trades": 30},
    )

    # Fix #5 & #6: guard against missing or malformed stats before key access
    try:
        total_perf = stats["totalPerformance"]
        port_stats = total_perf["portfolioStatistics"]
        pnl = float(port_stats["endEquity"]) - float(port_stats["startEquity"])
        sh_ratio = float(port_stats["sharpeRatio"])
    except (KeyError, TypeError, ValueError) as e:
        raise optuna.TrialPruned(f"Stats malformed or missing expected keys: {e}")

    trial.set_user_attr("pnl", pnl)
    trial.set_user_attr("stats", json.dumps(total_perf))
    return sh_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Study name", type=str, default="study-test")
    parser.add_argument(
        "--num-trials", help="Number of trials to run", type=int, default=1
    )
    parser.add_argument("--path", help="Study path", type=str, default=None)

    args = parser.parse_args()
    print(args)
    study_name = args.name
    num_trials = args.num_trials
    path = args.path

    pruner = optuna.pruners.PercentilePruner(
        percentile=33.0,         # prune bottom 1/3 of trials
        n_startup_trials=5,      # let first 5 trials run fully (need baseline data)
        n_warmup_steps=50,       # don't prune any trial before step 50
        n_min_trials=3,          # need ≥3 completed trials at a step to compare
    )
    study = optuna.load_study(study_name=study_name, storage=path, pruner=pruner)
    study.optimize(lambda t: objective(t), n_trials=num_trials, gc_after_trial=True)
