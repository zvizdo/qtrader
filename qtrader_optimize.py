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
            "invest_pct": (0.05, 0.5),
            "eval_invest_pct": 0.25,
            "expl_decay": 0.9925,
            "n_steps_warmup": 10_000,
            "n_step_update": 16, # 96 is once per day
            "model_n_layers": 1,
            "model_fl_size":  64, # 128,
            "model_shape": "flat",
            "exp_memory_size": 50_000, # 365 * 96 * 7 * 2, # 365 * 10_000,
            "exp_mini_batch_size": 256, 
            "exp_w_inc": 1e-5,
            "action_cooldown_bars": 8,
            # ===== TUNED =====
            "expl_min": trial.suggest_float("expl_min", 0.01, 0.10, step=0.01),
            "n_steps_checkpoint": trial.suggest_int("n_steps_checkpoint", 500, 2_000, step=250),
            "exp_alpha": trial.suggest_float("exp_alpha", 0.4, 0.8, step=0.1),
            "hold_cost_scale": trial.suggest_categorical("hold_cost_scale", [0.03, 0.05, 0.10]),
            "exit_bonus_scale": trial.suggest_categorical("exit_bonus_scale", [30.0, 50.0, 80.0]),
            # ===== LOCKED =====
            "exp_weighting": 0.4,
            "model_lr": 1e-5,
            "rl_gamma": 0.97,
            "target_tau": 0.005,
        },
        n_test=10,
        prune=_report_evals,  # {"n_tests": 7, "n_trades": 30},
    )

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
        n_warmup_steps=200,       # don't prune any trial before step 200
        n_min_trials=3,          # need ≥3 completed trials at a step to compare
    )
    study = optuna.load_study(study_name=study_name, storage=path, pruner=pruner)
    study.optimize(lambda t: objective(t), n_trials=num_trials, gc_after_trial=True)
