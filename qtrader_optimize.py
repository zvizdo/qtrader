import argparse
import json
import optuna
from qtrader_trainer import trainer_run

TRADE_FLOOR = 30
TRADE_PENALTY = 0.1

# Recency weights for intermediate eval averaging (most recent first)
_EVAL_WEIGHTS = [0.45, 0.25, 0.15, 0.1, 0.05]


def _penalized_sharpe(sharpe, num_trades):
    """Penalized Sharpe: pure Sharpe above TRADE_FLOOR, linear penalty below."""
    penalty = max(0, TRADE_FLOOR - num_trades) * TRADE_PENALTY
    return sharpe - penalty


def _extract_eval_score(stats):
    """Extract penalized Sharpe from a single LEAN eval stats dict."""
    port = stats.get("totalPerformance", {}).get("portfolioStatistics", {})
    trade_st = stats.get("totalPerformance", {}).get("tradeStatistics", {})
    sh = float(port.get("sharpeRatio", 0))
    nt = int(trade_st.get("totalNumberOfTrades", 0))
    return _penalized_sharpe(sh, nt)


def objective(trial, iters=150):
    def _report_evals(eval_stats, step):
        recent = eval_stats[-5:]
        scores = [_extract_eval_score(s) for s in recent]

        w = _EVAL_WEIGHTS[:len(scores)]
        w_sum = sum(w)
        # scores are oldest-first, weights are most-recent-first
        weighted = sum(s * wi / w_sum for s, wi in zip(reversed(scores), w))

        trial.report(weighted, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    name = f"{trial.study.study_name}-Tr{str(trial.number).zfill(4)}"

    stats = trainer_run(
        name=name,
        iters=iters,
        params={
            # ================================================================
            # LOCKED — Model architecture
            # ================================================================
            "model_n_layers": 1,
            "model_fl_size": 64,
            "model_shape": "flat",

            # ================================================================
            # LOCKED — Learning rate & optimizer
            # ================================================================
            "model_lr": 1e-5,

            # ================================================================
            # LOCKED — Discount factor
            # ================================================================
            # gamma=0.986 → effective horizon = 1/(1-γ) ≈ 71 bars ≈ 3 days.
            # Matches target trade duration (1-5 days). Exit bonuses at day 3
            # are discounted by γ^72 ≈ 0.36 — still a strong signal.
            "rl_gamma": 0.986,

            # ================================================================
            # LOCKED — Target network
            # ================================================================
            "target_tau": 0.005,

            # ================================================================
            # LOCKED — Update frequency & warmup
            # ================================================================
            "n_step_update": 8,
            "n_steps_warmup": 5_000,

            # ================================================================
            # LOCKED — Exploration schedule
            # ================================================================
            "expl_decay": 0.9925,
            "expl_min": 0.03,
            "n_steps_checkpoint": 250,

            # ================================================================
            # LOCKED — Experience replay
            # ================================================================
            "exp_memory_size": 50_000,
            "exp_mini_batch_size": 256,
            "exp_alpha": 0.6,
            "exp_weighting": 0.4,
            "exp_w_inc": 1e-5,

            # ================================================================
            # LOCKED — Position sizing & cooldown
            # ================================================================
            "invest_pct": 0.25,
            "eval_invest_pct": 0.25,
            "action_cooldown_bars": 2,

            # ================================================================
            # TUNED — Reward shaping (all in R_bar units, 0 = disabled)
            # ================================================================
            # hold_cost_scale: penalty per excess-day^1.5 past 72h, in R_bar.
            #   0    = no hold pressure (agent can hold indefinitely)
            #   0.05 = gentle (5d excess → 0.5σ/bar)
            #   0.1  = moderate (≈ current calibration)
            #   0.2  = firm (5d excess → 2σ/bar)
            #   0.5  = aggressive (5d excess → 5σ/bar)
            "hold_cost_scale": trial.suggest_categorical(
                "hold_cost_scale", [0, 0.05, 0.1, 0.2, 0.5]
            ),
            # exit_bonus_scale: peak tanh-compressed exit bonus in R_bar.
            #   0 = no exit shaping (pure market return only)
            #   2 = mild (peak ≈ 1.18)
            #   5 = moderate (peak ≈ 2.95, current equivalent)
            #   8 = strong (peak ≈ 4.72)
            "exit_bonus_scale": trial.suggest_categorical(
                "exit_bonus_scale", [0, 2, 5, 8]
            ),
            # exit_loss_ratio: loss penalty as fraction of profit bonus.
            #   0.5 = strong asymmetry (encourage risk-taking)
            #   0.7 = moderate asymmetry (current default)
            #   1.0 = symmetric (equal penalty for losses and reward for profits)
            "exit_loss_ratio": trial.suggest_categorical(
                "exit_loss_ratio", [0.5, 0.7, 1.0]
            ),
            # duration_bonus_scale: one-time exit bonus for holding 1-5d (peak 3d).
            #   0   = disabled
            #   0.5 = mild (peak ≈ 0.30)
            #   1.0 = moderate (peak = 1σ)
            #   2.0 = strong (peak ≈ 1.18)
            "duration_bonus_scale": trial.suggest_categorical(
                "duration_bonus_scale", [0, 0.5, 1.0, 2.0]
            ),
        },
        n_test=10,
        prune=_report_evals,
    )

    try:
        total_perf = stats["totalPerformance"]
        port_stats = total_perf["portfolioStatistics"]
        trade_stats = total_perf.get("tradeStatistics", {})
        pnl = float(port_stats["endEquity"]) - float(port_stats["startEquity"])
        sh_ratio = float(port_stats["sharpeRatio"])
        num_trades = int(trade_stats.get("totalNumberOfTrades", 0))
    except (KeyError, TypeError, ValueError) as e:
        raise optuna.TrialPruned(f"Stats malformed or missing expected keys: {e}")

    score = _penalized_sharpe(sh_ratio, num_trades)

    trial.set_user_attr("pnl", pnl)
    trial.set_user_attr("sharpe", sh_ratio)
    trial.set_user_attr("num_trades", num_trades)
    trial.set_user_attr("stats", json.dumps(total_perf))
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Study name", type=str, default="study-test")
    parser.add_argument(
        "--num-trials", help="Number of trials to run", type=int, default=1
    )
    parser.add_argument("--path", help="Study path", type=str, default=None)
    parser.add_argument("--iters", help="Training iterations per trial", type=int, default=150)

    args = parser.parse_args()
    print(args)
    study_name = args.name
    num_trials = args.num_trials
    path = args.path
    iters = args.iters

    pruner = optuna.pruners.PercentilePruner(
        percentile=33.0,         # prune bottom 1/3 of trials
        n_startup_trials=5,      # let first 5 trials run fully (need baseline data)
        n_warmup_steps=int(iters * 0.20),  # 20% of training budget
        n_min_trials=3,          # need ≥3 completed trials at a step to compare
    )
    study = optuna.load_study(study_name=study_name, storage=path, pruner=pruner)
    study.optimize(lambda t: objective(t, iters=iters), n_trials=num_trials, gc_after_trial=True)
