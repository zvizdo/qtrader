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
            "model_n_layers": 2,
            "model_fl_size": 256,
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
            "n_step_update": 64, # trial.suggest_categorical("n_step_update", [32, 48, 64]),
            "n_steps_warmup": 5_000,

            # ================================================================
            # TUNED — Exploration schedule
            # ================================================================
            # agent_class: control arm vs Boltzmann. Routes in main.py._create_agent.
            "agent_class": trial.suggest_categorical(
                "agent_class", ["boltzmann", "epsilon_greedy"]
            ),
            "expl_max": 3.0, # for BoltzmannDQTPAgent, 1.0 for DQTPAgent
            # expl_min acts as τ floor for Boltzmann; ε floor for ε-greedy.
            # Previous sweep used 0.1 → full determinism by mid-training.
            # Raised floors prevent exploration collapse.
            "expl_min": trial.suggest_categorical("expl_min", [0.2, 0.3, 0.5]),
            "expl_decay": trial.suggest_categorical("expl_decay", [0.995, 0.998]),
            "n_steps_checkpoint": 800,
            # Boltzmann uniform-mix floor: guarantees permanent exploration
            # regardless of τ decay. 0 = disabled, 0.05 ≈ 5% uniform mass.
            "boltz_uniform_floor": trial.suggest_categorical(
                "boltz_uniform_floor", [0.02, 0.05, 0.1]
            ),

            # ================================================================
            # TUNED — Experience replay
            # ================================================================
            # Buffer capacity rounds up to nearest power of 2 internally.
            # 1 iteration = 3,360 bars (140 days)
            # 16,384  ≈  5 iterations (highly reactive, discards old regimes quickly)
            # 32,768  ≈ 10 iterations (short memory, smooth locally)
            # 65,536  ≈ 20 iterations (balanced, good for small capacity networks)
            # 131,072 ≈ 39 iterations (long memory, highest before off-policy staleness harms learning)
            "exp_memory_size": 65536,
            "exp_mini_batch_size": 256,
            "exp_alpha": 0.6,
            "exp_weighting": 0.4,
            "exp_w_inc": 1e-5,

            # ================================================================
            # LOCKED — Position sizing & cooldown
            # ================================================================
            "invest_pct": 0.25,
            "eval_invest_pct": 0.25,
            # Locked to 2: Tr0030-0038 (cd=0) definitively overtraded and
            # diverged; cd=2 was stable across all prior trials.
            "action_cooldown_bars": 2,

            # ================================================================
            # LOCKED — Reward shaping (pinned from 39-trial reward-calib sweep)
            # ================================================================
            # hold_cost_scale: ≥0.02 strictly worse across all prior trials
            # (Tr0012,0014,0016 death-spiraled faster). 0.005 is the safe default.
            "hold_cost_scale": 0.005,
            # exit_bonus_scale: locked at 1.0 (1 R_bar peak) — after lowering
            # _TRADE_PNL_REF to 0.01 this restores tanh resolution in the
            # 0.5-3% pnl range where most 1H BTC trades live.
            "exit_bonus_scale": 1.0,
            # exit_loss_ratio: mildly asymmetric (loss penalty = 70% of profit bonus).
            "exit_loss_ratio": 0.7,
            # duration_bonus_scale: showed no measurable effect in any prior trial.
            "duration_bonus_scale": 0.0,

            # ================================================================
            # TUNED — Anti-flat lever (only reward knob still swept)
            # ================================================================
            # opp_cost_scale: per-bar flat penalty = scale × max(0, market_return).
            # Previous sweep capped at 0.5, which was insufficient to counter
            # the death-spiral asymmetry. Extended upper range to 2.0.
            #   0.25 = prior sweep upper mid (baseline)
            #   0.5  = prior sweep max
            #   1.0  = E[Q(FLAT)] ≈ -0.035/bar, meaningful anti-flat pressure
            #   2.0  = E[Q(FLAT)] ≈ -0.07/bar, strong anti-flat
            "opp_cost_scale": trial.suggest_categorical(
                "opp_cost_scale", [0.25, 0.5, 1.0, 2.0]
            ),
        },
        n_test=10,
        prune=_report_evals,
    )

    try:
        total_perf = stats["totalPerformance"]
        aggregated = stats.get("aggregated")
        if not aggregated:
            raise optuna.TrialPruned(
                "Stats missing 'aggregated' key — trainer must populate "
                "randomized multi-window final eval metrics."
            )
        sh_ratio = float(aggregated["sharpe_final_mean"])
        num_trades = int(aggregated["num_trades_final_mean"])
        pnl = float(aggregated["profit_final_mean"])
    except (KeyError, TypeError, ValueError) as e:
        raise optuna.TrialPruned(f"Stats malformed or missing expected keys: {e}")

    score = _penalized_sharpe(sh_ratio, num_trades)

    trial.set_user_attr("pnl", pnl)
    trial.set_user_attr("sharpe", sh_ratio)
    trial.set_user_attr("num_trades", num_trades)
    trial.set_user_attr("sharpe_median", float(aggregated.get("sharpe_final_median", sh_ratio)))
    trial.set_user_attr("sharpe_p25", float(aggregated.get("sharpe_final_p25", sh_ratio)))
    trial.set_user_attr("sharpe_p75", float(aggregated.get("sharpe_final_p75", sh_ratio)))
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
        n_startup_trials=15,      # let first 15 trials run fully (need baseline data)
        n_warmup_steps=int(iters * 0.50),  # 50% of training budget
        n_min_trials=10,          # need ≥10 completed trials at a step to compare
    )
    study = optuna.load_study(study_name=study_name, storage=path, pruner=pruner)
    study.optimize(lambda t: objective(t, iters=iters), n_trials=num_trials, gc_after_trial=True)
