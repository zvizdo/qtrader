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
            "invest_pct": 0.05,
            "invest_max": 0.25,  # how much to invest in an order
            "expl_min": trial.suggest_float("expl_min", 0, 0.25, step=0.01),
            "expl_decay": trial.suggest_categorical(
                "expl_decay", [round(0.9 + 0.005 * k, 3) for k in range(20)]
            ),
            "n_steps_warmup": 1024 * 10,
            "n_step_update": 2,
            "n_steps_target_update": trial.suggest_int(
                "n_steps_target_update", 50, 5_000, step=50
            ),
            "model_n_layers": 4,  # trial.suggest_categorical("model_n_layers", [2, 3, 4]),
            "model_fl_size": 128,  # trial.suggest_categorical( "model_fl_size", [16, 32, 64, 128, 256]),
            "model_shape": "flat",
            "exp_memory_size": 365 * 100_000,
            "exp_mini_batch_size": 32,  # trial.suggest_categorical("exp_mini_batch_size", [16, 32, 64, 128, 256]),
            "exp_weighting": trial.suggest_float("exp_weighting", 0, 1, step=0.025),
            "exp_w_inc": trial.suggest_categorical(
                "exp_w_inc",
                [1e-4, 7.5e-5, 5e-5, 2.5e-5, 1e-5, 7.5e-6, 5e-6, 2.5e-6, 1e-6],
            ),
            "exp_alpha": trial.suggest_float("exp_alpha", 0.0, 1, step=0.025),
            "model_lr": trial.suggest_categorical(
                "model_lr",
                [1e-4, 7.5e-5, 5e-5, 2.5e-5, 1e-5, 7.5e-6, 5e-6, 2.5e-6, 1e-6],
            ),
            "model_act_func": "relu",
            "model_l2_reg": 0,
            "rl_gamma": trial.suggest_categorical(
                "rl_gamma", [0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99]
            ),
            "rl_reward_type": "position-relative-action-log",  # trial.suggest_categorical("rl_reward_type", ["portfolio-change-log", "position-relative-action-log"],),
            # "rl_nudge_reward_pct": trial.suggest_float(
            #     "rl_nudge_reward_pct", 0, 0.020, step=0.001
            # ),
        },
        n_test=10,
        prune=_report_evals,  # {"n_tests": 7, "n_trades": 30},
    )

    pnl = lambda s: float(
        s["totalPerformance"]["portfolioStatistics"]["endEquity"]
    ) - float(s["totalPerformance"]["portfolioStatistics"]["startEquity"])
    sh_ratio = lambda s: float(
        s["totalPerformance"]["portfolioStatistics"]["sharpeRatio"]
    )

    trial.set_user_attr("pnl", pnl(stats))
    trial.set_user_attr("stats", json.dumps(stats["totalPerformance"]))
    return sh_ratio(stats)


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

    study = optuna.load_study(study_name=study_name, storage=path)
    study.optimize(lambda t: objective(t), n_trials=num_trials, gc_after_trial=True)
