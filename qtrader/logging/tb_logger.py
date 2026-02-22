"""
Dedicated TensorBoard logging module for DQN training.

Encapsulates all TensorBoard logic so the training loop and agent code
stay clean. Instantiate one ``TrainingLogger`` per training run and call
``log_train_step`` / ``log_eval_step`` after each backtest completes.
"""

from __future__ import annotations

import json
import tensorflow as tf
from typing import Any, Dict, Optional


def _safe_float(d: dict, *keys, default: float = 0.0) -> float:
    """Walk *keys* into nested dict *d*, returning *default* on any miss."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


class TrainingLogger:
    """Encapsulates TensorBoard scalar logging for DQN training & eval."""

    def __init__(self, log_dir: str) -> None:
        self._writer = tf.summary.create_file_writer(logdir=log_dir)

    # ------------------------------------------------------------------
    # TRAIN metrics (logged every iteration after a TRAIN backtest)
    # ------------------------------------------------------------------
    def log_train_step(self, step: int, stats: dict) -> None:
        """Log metrics from a completed TRAIN backtest.

        Expected custom statistics (set via ``set_summary_statistic`` in
        ``main.py``):
            - mean_td_error
            - mean_portfolio_change   (renamed from mean_reward)
            - mean_shaped_reward      (actual DQN training reward)
            - exploration_rate
            - n_updates
            - replay_buffer_size
            - mean_loss
            - mean_q_value
        """
        s = stats.get("statistics", {})
        perf = stats.get("totalPerformance", {})

        with self._writer.as_default(step=step):
            # -- Core DQN convergence signals --
            if "mean_td_error" in s:
                tf.summary.scalar("train/mean_td_error", float(s["mean_td_error"]))
            if "mean_loss" in s:
                tf.summary.scalar("train/mean_loss", float(s["mean_loss"]))
            if "mean_q_value" in s:
                tf.summary.scalar("train/mean_q_value", float(s["mean_q_value"]))

            # -- Reward signals --
            if "mean_shaped_reward" in s:
                tf.summary.scalar(
                    "train/mean_shaped_reward", float(s["mean_shaped_reward"])
                )
            if "mean_portfolio_change" in s:
                tf.summary.scalar(
                    "train/mean_portfolio_change", float(s["mean_portfolio_change"])
                )

            # -- Agent state --
            if "exploration_rate" in s:
                tf.summary.scalar(
                    "train/exploration_rate", float(s["exploration_rate"])
                )
            if "n_updates" in s:
                tf.summary.scalar("train/n_updates", float(s["n_updates"]))
            if "replay_buffer_size" in s:
                tf.summary.scalar(
                    "train/replay_buffer_size", float(s["replay_buffer_size"])
                )

            # -- Training PnL --
            pnl = (
                _safe_float(perf, "portfolioStatistics", "endEquity")
                - _safe_float(perf, "portfolioStatistics", "startEquity")
            )
            tf.summary.scalar("train/profit", pnl)

        self._writer.flush()

    # ------------------------------------------------------------------
    # EVAL metrics (logged every n_test iterations)
    # ------------------------------------------------------------------
    def log_eval_step(self, step: int, stats: dict) -> None:
        """Log metrics from a completed EVAL backtest."""
        perf = stats.get("totalPerformance", {})

        pnl = (
            _safe_float(perf, "portfolioStatistics", "endEquity")
            - _safe_float(perf, "portfolioStatistics", "startEquity")
        )

        with self._writer.as_default(step=step):
            tf.summary.scalar("eval/profit", pnl)
            tf.summary.scalar(
                "eval/sharpe_ratio",
                _safe_float(perf, "portfolioStatistics", "sharpeRatio"),
            )
            tf.summary.scalar(
                "eval/num_trades",
                _safe_float(perf, "tradeStatistics", "totalNumberOfTrades"),
            )
            tf.summary.scalar(
                "eval/drawdown",
                _safe_float(perf, "portfolioStatistics", "drawdown"),
            )
            tf.summary.scalar(
                "eval/win_rate",
                _safe_float(perf, "tradeStatistics", "winRate"),
            )

        self._writer.flush()

    # ------------------------------------------------------------------
    # Hyperparameters (logged once at the start of a run)
    # ------------------------------------------------------------------
    def log_hyperparams(self, params: dict) -> None:
        """Write hyperparameters as a TensorBoard text summary."""
        md_lines = ["| Param | Value |", "|-------|-------|"]
        for k, v in sorted(params.items()):
            md_lines.append(f"| {k} | {v} |")
        md_text = "\n".join(md_lines)

        with self._writer.as_default(step=0):
            tf.summary.text("hyperparams", md_text)
        self._writer.flush()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Flush remaining data and close the writer."""
        self._writer.flush()
        self._writer.close()
