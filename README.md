# QTrader — Deep Reinforcement Learning Trading Bot

A deep Q-learning trading agent for Bitcoin (BTCUSD) built on top of QuantConnect LEAN. The agent learns to trade autonomously using prioritized experience replay, multi-timeframe technical indicators, and a configurable reward function — all orchestrated through an Optuna-driven hyperparameter optimization pipeline.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Quickstart](#quickstart)
  - [1. Generate the Golden Cache](#1-generate-the-golden-cache-warmup)
  - [2. Train the Agent](#2-train-the-agent)
  - [3. Hyperparameter Optimization](#3-hyperparameter-optimization)
  - [4. Evaluate a Trained Model](#4-evaluate-a-trained-model)
- [How It Works](#how-it-works)
  - [Run Modes](#run-modes)
  - [RL Pipeline](#rl-pipeline)
  - [Agent (DQTP)](#agent-dqtp)
  - [Feature Engineering](#feature-engineering)
  - [Reward Function](#reward-function)
  - [Caching & Persistence](#caching--persistence)
- [LEAN Integration](#lean-integration)
- [Monitoring](#monitoring)
- [Backtest Analysis](#backtest-analysis)
- [Configuration Reference](#configuration-reference)

---

## Architecture Overview

At the top level, an **Optuna study** searches over hyperparameters. Each trial spins up a **trainer** that runs hundreds of LEAN backtests — every backtest is a full episode where the agent observes 1-hour bars, decides to go long or stay flat, and learns from the outcome via experience replay. Between episodes, the trainer samples random date windows, decays exploration, and periodically evaluates on a held-out period.

Inside each backtest the heavy lifting happens in four stages that fire on every consolidated bar: **state providers** (Bridge Bands, MACD, Trend Maturity, OHLCV, volume, position & account features) produce a 72-dim observation → the **DQTP agent** picks an action via ε-greedy Double DQN → the **feedback module** computes a shaped reward from log-returns, holding cost, and trade-completion bonus → the **learning step** samples a prioritized mini-batch and updates the online network. A pre-computed **golden cache** (diskcache.Index) ensures indicator values are calculated only once across all runs.

```
┌──────────────────────────────────────────────┐
│  Optimization Layer                          │
│  qtrader_optimize_runner.py  (parallel)      │
│      └── qtrader_optimize.py (Optuna HPO)    │
│              └── qtrader_trainer.py           │
│                      └── N episodes (LEAN)   │
└───────────────┬──────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────┐
│  QuantConnect LEAN Backtest Engine           │
│  main.py :: QTraderAlgorithm                 │
│                                              │
│  ┌────────────┐   ┌───────────────────────┐  │
│  │ Market Env │──▶│  RL Pipeline          │  │
│  │ (1H OHLCV) │   │  State → Act →        │  │
│  └────────────┘   │  Feedback → Learn     │  │
│                   └──────────┬────────────┘  │
│                              │               │
│          ┌───────────────────┼────────┐      │
│          │                   │        │      │
│  ┌───────▼───────┐  ┌───────▼─────┐  │      │
│  │ State         │  │ DQTP Agent  │  │      │
│  │ Providers     │  │ Double DQN  │  │      │
│  │ (BB, MACD,    │  │ + Prioritized│ │      │
│  │  OHLCV, Pos)  │  │   Replay    │  │      │
│  └───────┬───────┘  └─────────────┘  │      │
│          │                           │      │
│  ┌───────▼───────────────────────────▼──┐   │
│  │ diskcache.Index  (Golden Cache)      │   │
│  │ + LEAN Object Store                  │   │
│  └──────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

---

## Project Structure

```
qtrader/
├── main.py                     # LEAN algorithm entry point
├── qtrader_trainer.py          # Training pipeline (single run, N iterations)
├── qtrader_optimize.py         # Optuna study manager (single trial per process)
├── qtrader_optimize_runner.py  # Spawns parallel qtrader_optimize.py processes
├── qtrader_warmup.py           # Golden cache generator (parallel)
├── bt_analysis.py              # Backtest results → LLM-friendly Markdown
├── config.json                 # LEAN configuration
├── requirements.txt            # Python dependencies
├── AGENTS.md                   # Detailed agent architecture reference
├── TRIALS_LOG.md               # Experiment history and trial logs
├── notes.md                    # Research notes
│
├── qtrader/                    # Core package
│   ├── agents/
│   │   ├── base.py             # BaseAgent, RandomAgent
│   │   ├── dqtp.py             # DQTPAgent — Double DQN + prioritized replay
│   │   └── expreplay/
│   │       └── buffer.py       # Segment-tree prioritized replay buffer
│   │
│   ├── environments/
│   │   ├── base.py             # BaseMarketEnv ABC + get_order_pnl() (FIFO PnL)
│   │   ├── lean.py             # QuantConnect LEAN market environment
│   │   ├── backtesting.py      # Legacy standalone backtesting env (unused)
│   │   └── bitstamp.py         # Legacy Bitstamp env (unused)
│   │
│   ├── stateproviders/
│   │   ├── __init__.py         # BaseStateProvider, BaseSymbolStateProvider
│   │   ├── basic.py            # Account, Position, Trade, OHLCV providers
│   │   ├── indicators.py       # Bridge Bands, MACD, Trend Maturity, Trendlines
│   │   └── model.py            # (exists but empty/unused)
│   │
│   ├── rlflow/
│   │   ├── __init__.py         # BaseTask
│   │   ├── state.py            # State provider tasks & aggregation (w/ cache)
│   │   ├── action.py           # Action execution
│   │   ├── feedback.py         # Reward computation & agent feedback
│   │   └── persistence.py      # diskcache.Index persistence + caching layer
│   │
│   ├── logging/
│   │   ├── __init__.py         # exports TrainingLogger
│   │   └── tb_logger.py        # TensorBoard logger
│   │
│   └── dashboard/              # Streamlit monitoring dashboard
│       ├── main.py
│       └── utils.py
│
├── playground/                 # Research notebooks & scripts
│   ├── nb_bridgebands.ipynb
│   ├── nb_macd.ipynb
│   ├── nb_trendlines.ipynb
│   ├── nb_study_explore.ipynb
│   ├── nb_rf.ipynb
│   └── eda_features.py
│
├── research.ipynb              # Main research notebook
├── nb_qtrader_eval.ipynb       # Backtest evaluation
└── nb_study_monitor.ipynb      # Optuna study visualization
```

---

## Setup

### Prerequisites

- Python 3.10+
- [LEAN CLI](https://www.lean.io/docs/lean-cli/getting-started/lean-cli) installed and configured
- Historical BTCUSD minute-resolution data available through LEAN

### Install dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**

| Package      | Version  | Purpose                          |
|-------------|----------|----------------------------------|
| `lean`      | 1.0.*    | QuantConnect LEAN CLI & SDK      |
| `diskcache` | 5.6.3    | Persistent key-value cache       |

Additional dependencies (installed within the LEAN Docker container): `tensorflow`, `numpy`, `pandas`, `msgpack`.

---

## Quickstart

The typical workflow has three stages: **warm up the cache**, **train**, and **evaluate**. Each stage runs through LEAN backtests under the hood.

### 1. Generate the Golden Cache (Warmup)

Before training, pre-compute all indicator values across the full historical date range. This is a one-time step that avoids redundant computation during training iterations.

```bash
python qtrader_warmup.py \
    --date-start 2016-01-01 \
    --date-end 2026-02-01 \
    --golden-cache-dir ../storage/cache
```

This runs `main.py` in `WARMUP` mode — it walks through every 1-hour bar, computes all state providers (Bridge Bands, MACD, Trend Maturity, OHLCV features), and writes the results to a `diskcache.Index` directory. The date range is split into chunks and processed in parallel via separate Docker containers, then merged into a single index.

**What gets cached:**

- Bridge Bands (micro / daily / weekly scales)
- MACD (micro / daily / weekly scales)
- Trend Maturity (swing-structure features)
- OHLCV features

**Cache key format:** `Flow-State-{symbol}-{YYYYMMDDHHMM}-{ProviderClass}-{ParamsHash}`

Once generated, this directory is reused as a read-through cache by every training and evaluation iteration.

### 2. Train the Agent

Run a training session with a fixed set of hyperparameters:

```bash
python qtrader_trainer.py \
    --name "experiment-01" \
    --iters 25 \
    -p expl_min=0.01 expl_decay=0.98 model_lr=1e-4
```

**What happens per iteration:**

1. A random 140-day date window is sampled from the training period (2016–2023)
2. The golden cache directory is mounted as a read-through cache provider
3. LEAN runs a backtest (`main.py` in `TRAIN` mode) — the agent observes 1-hour bars, takes actions, and learns via experience replay
4. Every `n_test` iterations (default 10), an evaluation run executes on a held-out period with frozen weights

**Passing hyperparameters** — use `-p key=value` pairs:

```bash
python qtrader_trainer.py \
    --name "high-gamma" \
    --iters 50 \
    -p rl_gamma=0.99 model_fl_size=256 model_n_layers=3
```

### 3. Hyperparameter Optimization

Use Optuna to search over the hyperparameter space:

```bash
# Single-process mode:
python qtrader_optimize.py \
    --name "hpo-study" \
    --num-trials 50 \
    --path "sqlite:///trials/hpo-study.db" \
    --iters 150

# Multi-process mode (spawns parallel trial workers):
python qtrader_optimize_runner.py \
    --name "hpo-study" \
    --num-trials 50 \
    --path "sqlite:///trials/hpo-study.db" \
    --iters 150 \
    --n-jobs 2
```

**Tuned parameters** (defined in `qtrader_optimize.py`):

| Parameter              | Type        | Values                              |
|------------------------|-------------|-------------------------------------|
| `n_step_update`        | categorical | 32, 48, 64                          |
| `hold_cost_scale`      | categorical | 0, 0.005, 0.01, 0.02, 0.05         |
| `exit_bonus_scale`     | categorical | 0, 0.1, 0.25, 0.5, 1.0, 2.0       |
| `exit_loss_ratio`      | categorical | 0.5, 0.7, 1.0                      |
| `duration_bonus_scale` | categorical | 0, 0.05, 0.1, 0.25, 0.5, 1.0      |

**Static parameters** (fixed across all trials):

| Parameter              | Value       | Notes                                  |
|------------------------|-------------|----------------------------------------|
| `model_n_layers`       | 2           | Hidden layers                          |
| `model_fl_size`        | 256         | Hidden layer width                     |
| `model_shape`          | `"flat"`    | All layers same size                   |
| `model_lr`             | 1e-5        | Adam learning rate                     |
| `rl_gamma`             | 0.986       | Discount factor (~71 bar horizon)      |
| `target_tau`           | 0.005       | Target network soft update rate        |
| `n_steps_warmup`       | 5,000       | Steps before learning starts           |
| `expl_decay`           | 0.995       | Per-checkpoint decay                   |
| `expl_min`             | 0.03        | Minimum exploration rate               |
| `n_steps_checkpoint`   | 500         | Steps between checkpoints              |
| `exp_memory_size`      | 65,536      | Replay buffer capacity                 |
| `exp_mini_batch_size`  | 256         | Learning batch size                    |
| `exp_alpha`            | 0.6         | PER priority exponent                  |
| `exp_weighting`        | 0.4         | IS beta starting value                 |
| `exp_w_inc`            | 1e-5        | IS beta annealing increment            |
| `invest_pct`           | 0.25        | Fixed (not randomized)                 |
| `eval_invest_pct`      | 0.25        | Fixed during evaluation runs           |
| `action_cooldown_bars` | 2           | Bars before exit allowed               |
| `iters`                | 150         | Episodes per trial (CLI configurable)  |
| `n_test`               | 10          | Evaluate every 10 iterations           |

**Objective:** maximize penalized Sharpe ratio on the final full-period evaluation. Sharpe is penalized linearly when total trades < 30.

**Pruning:** `PercentilePruner(percentile=33, n_startup_trials=15, n_warmup_steps=50% of iters, n_min_trials=10)`. Intermediate eval scores are weighted by recency: last 5 evals with weights `[0.45, 0.25, 0.15, 0.1, 0.05]` (most recent first).

### 4. Evaluate a Trained Model

Evaluation happens automatically during training (every `n_test` iterations). Intermediate evals run on a fixed period (Jan 15, 2025 – Jan 14, 2026); the final iteration evaluates on the full out-of-sample period (Feb 1, 2023 – Jan 31, 2026).

Results are logged to TensorBoard and saved as LEAN backtest results in `backtests/`.

---

## How It Works

### Run Modes

The algorithm (`main.py`) supports four modes, controlled by the `run_type` parameter:

| Mode     | Learning | Exploration     | Agent created       | Purpose                                     |
|----------|----------|-----------------|---------------------|---------------------------------------------|
| `WARMUP` | Off      | N/A             | No (None)           | Pre-compute and cache all indicator values   |
| `TRAIN`  | On       | ε-greedy        | Yes                 | Train the agent via experience replay        |
| `EVAL`   | Off      | Off (greedy)    | Yes (no_learn=True) | Evaluate with frozen weights (greedy policy) |
| `LIVE`   | Off      | Off             | Yes (no_learn=True) | Live trading with full state persistence     |

### RL Pipeline

On every 1-hour consolidated bar, the pipeline executes:

```
State Providers    →    Aggregation    →    Agent Act    →    Feedback & Learn
(indicators,            (72-dim feature     (ε-greedy        (reward calc,
 position info,          vector)             action           experience replay,
 account info)                               selection)       weight update)
```

**Step by step:**

1. **State computation** — each state provider (Bridge Bands, MACD, Trend Maturity, OHLCV, position, account) runs and its output is cached
2. **Aggregation** — all provider outputs are combined into a flat 72-dimensional feature vector
3. **Action** — the agent picks `FLAT` (exit/stay flat) or `LONG` (enter/hold) via ε-greedy policy
4. **Feedback** — reward is computed from market return, trade cost, holding cost, and trade-completion bonus
5. **Learn** — every `n_step_update` steps, the agent samples a mini-batch from the replay buffer and updates the online network

### Agent (DQTP)

The agent is a **Double DQN** with **Prioritized Experience Replay**:

- **Online network** — predicts Q-values, updated via gradient descent
- **Target network** — provides stable TD targets, updated via soft copy: `θ_target ← (1-τ)·θ_target + τ·θ_online`
- **Replay buffer** — segment-tree based prioritized sampling (higher TD-error = higher sampling probability). New experiences are inserted at `max_priority`, which soft-decays (×0.999) toward the actual observed maximum to prevent stale priority inflation

**Network architecture:**

```
Input (72 features)
  → LayerNormalization
  → Dense(N, activation='elu', GlorotUniform(seed=42))  # repeated per layer
  → Dense(2, activation='linear')   # Q(FLAT), Q(LONG)

Loss:      Huber(delta=3.0)
Optimizer: Adam(lr=model_lr, clipnorm=1.0)
```

`model_shape="cone"`: each layer halves in size. `model_shape="flat"`: all layers same size.

**Constructor defaults** (fallbacks — actual training uses values from `params.json`):

| Parameter             | Default  | Description                                      |
|-----------------------|----------|--------------------------------------------------|
| `expl_max`            | 1.0      | Initial exploration rate                          |
| `expl_min`            | 0.01     | Minimum exploration rate                          |
| `expl_decay`          | 0.9      | Exploration decay per checkpoint                  |
| `invest_pct`          | 0.02     | Fraction of account value per trade               |
| `exp_memory_size`     | 365      | Replay buffer capacity (raw value)                |
| `exp_mini_batch_size` | 128      | Mini-batch size for learning                      |
| `exp_alpha`           | 0.8      | Priority exponent (α in PER)                     |
| `exp_weighting`       | 0.4      | Importance-sampling β (annealed toward 1.0)      |
| `exp_w_inc`           | 0.0005   | IS β increment per learning step                 |
| `n_steps_warmup`      | 1,000    | Steps before first weight update                  |
| `n_step_update`       | 10       | Steps between weight updates                      |
| `n_steps_checkpoint`  | 1,000    | Steps between exploration decay & model save      |
| `target_tau`          | 0.001    | Soft update rate for target network               |
| `rl_gamma`            | 0.9      | Discount factor                                   |
| `model_lr`            | 1e-4     | Adam learning rate                                |
| `model_l2_reg`        | 0.0      | L2 regularization (commented out in code)         |
| `model_layers`        | [32]     | Hidden layer sizes                                |
| `model_act_func`      | `"elu"`  | Hidden layer activation function                  |
| `hold_cost_scale`     | 0.085    | Hold cost scale in R_bar units                    |
| `exit_bonus_scale`    | 5.0      | Exit bonus scale in R_bar units                   |
| `exit_loss_ratio`     | 1.0      | Loss penalty as fraction of profit bonus          |
| `duration_bonus_scale`| 0.0      | Duration bonus (disabled by default)              |
| `action_cooldown_bars`| 0        | Bars before exit allowed (0 = disabled)           |

Note: the `main.py` `_create_agent()` method applies its own defaults when params are missing, which differ from `dqtp.py` constructor defaults (e.g. `expl_decay=0.9995`, `n_steps_warmup=1024`, `n_step_update=4`, `n_steps_checkpoint=5000`, `exp_memory_size=365*100000`, `exp_mini_batch_size=32`, `hold_cost_scale=0.085`, `exit_bonus_scale=5.0`, `exit_loss_ratio=0.7`, `model_shape="flat"`, `model_n_layers=2`, `model_fl_size=128`).

The target network is updated on **every learning step** (not on a separate schedule), making `target_tau` the primary knob for controlling target lag. Q-value targets are clipped to `±1.5/(1-γ)` and TD errors are clipped to [0, 5] to prevent runaway bootstrapping.

### Feature Engineering

The agent observes a **72-dimensional** feature vector built from multi-timeframe indicators:

| Group                    | Features | Description                                              |
|--------------------------|----------|----------------------------------------------------------|
| Time encoding            | 2        | sin/cos of weekday+hour (π/3 period)                    |
| Position sign            | 1        | 0 = flat, 1 = long                                      |
| Return on position       | 1        | `sign(rop) × log1p(100 × |rop|)` (0 when flat)         |
| Hold duration            | 1        | `min(hold_hours, 360) / 72` (0 when flat)               |
| PnL trajectory           | 3        | `log(close[-i] / entry) × 100` at [1, 4, 12] bars back |
| OHLC (vol-normalized)    | 3        | open/high/low relative to close, scaled by BB micro width |
| Candle body ratio        | 1        | `|close-open| / (high-low)` — conviction vs indecision  |
| Log-returns              | 3        | raw log-return at 1, 24, 72 bars back (×100)            |
| Relative volume          | 3        | `log1p(volume / mean_72)` at 1, 24, 72 bars back        |
| Bridge Bands (×3 scales) | 27       | band width, band position, Hurst exponent at 3 lookbacks |
| MACD (×3 scales)         | 18       | MACD value and histogram at 3 lookbacks                 |
| Trend Maturity           | 9        | direction, exhaustion, wave counts, efficiency, etc.     |
| **Total**                | **72**   |                                                          |

**Multi-timeframe scales:**

| Scale  | Bridge Bands Length | MACD (short/long/signal) | Captures            |
|--------|---------------------|--------------------------|---------------------|
| Micro  | 12 bars (12h)       | 6 / 13 / 4              | Intraday volatility  |
| Daily  | 96 bars (4 days)    | 24 / 72 / 24            | Swing momentum       |
| Weekly | 336 bars (14 days)  | 112 / 240 / 48          | Macro trend          |

**Trend Maturity** uses `swing_order=7` (7h each side) with a 120-bar (5-day) lookback. It produces 9 features per bar (current bar only, no lookbacks): direction, exhaustion, bull/bear wave counts, extending flag, retracement depth, impulse ratio, directional bias, and efficiency ratio.

Bridge Bands and MACD features are observed at **3 lookback windows** (1, 24, 72 bars back), giving the agent a sense of feature trajectory.

### Reward Function

All shaping components are anchored to `R_BAR = sigma_1h × 75 ≈ 0.59` (the per-bar noise floor). Each component disables when its scale = 0.

```
R(t) = P(t) × [clip(log_return × 75, -1, 1) - hold_cost]
     - comm_frac × 75
     + exit_bonus
     + duration_bonus
```

Five components:

1. **Market return** — clipped log-return scaled by 75, only received while LONG
2. **Holding cost** — zero for the first 72h, then ramps as `hold_cost_scale × R_BAR × ((hours-72)/24)^1.5`
3. **Trade cost** — commission as a fraction of position notional, scaled by 75
4. **Exit bonus** (on trade exit only) — `exit_bonus_scale × R_BAR × tanh(trade_pnl_pct / 0.03)` for profits; losses scaled by `exit_loss_ratio`
5. **Duration bonus** (on trade exit only) — reverse-U peaking at 3-day hold: `duration_bonus_scale × R_BAR × (1 - ((days-3)/2)²)` for holds in [1d, 5d]

Where:

- **P(t)** = 1 if in a LONG position, 0 if FLAT. **R_flat = 0** (no flat penalty — eliminates death spiral)
- **log_return** = `ln(price_future / price_current)`
- **comm_frac** = commission / position notional
- **Scale** = 75.0 (targets ~93% of rewards in [-1, 1] for BTC 1H bars)

The holding cost creates time pressure to exit stale positions while the exit bonus rewards profitable trade completion, encouraging the agent to hold through short-term noise when the trade thesis remains valid.

### Caching & Persistence

The persistence layer uses **diskcache.Index** (a SQLite-backed key-value store) instead of raw SQLite tables:

```
LeanDiskIndexPersistenceProvider        ← used in main.py
  ├── Optional read-through cache_provider (golden cache)
  ├── diskcache.Index (SQLite + filesystem)
  ├── msgpack serialization for dicts
  └── LEAN Object Store integration
```

**Golden cache workflow:**

1. `qtrader_warmup.py` runs once over the full date range, populating a `diskcache.Index` directory (the golden cache). Warmup runs in parallel — the date range is chunked and each chunk runs as a separate LEAN backtest in Docker, then results are merged into a single index.
2. At the start of each training/eval backtest, the golden cache directory is passed as a read-through `cache_provider` to `LeanDiskIndexPersistenceProvider`
3. During the backtest, key lookups check the golden cache first — misses fall through to the trial's own diskcache.Index

**Serialization:** [msgpack](https://msgpack.org/) (binary, fast) for dict payloads; native pickle for arbitrary objects (models, replay buffers).

---

## LEAN Integration

QTrader runs as a standard QuantConnect LEAN algorithm. The entry point is `main.py`, which subclasses `QCAlgorithm`.

### How LEAN is used

| LEAN Feature          | Usage in QTrader                                       |
|----------------------|--------------------------------------------------------|
| `add_crypto()`       | Subscribes to BTCUSD at minute resolution              |
| `TradeBarConsolidator` | Consolidates 1-minute bars into 1-hour bars          |
| `history()`          | Retrieves 60 days of historical bars for warmup        |
| `market_order()`     | Executes BUY orders                                    |
| `liquidate()`        | Closes open positions                                  |
| `object_store`       | Persists models, replay buffer, and cache between runs |
| `transactions`       | Tracks commissions for reward computation              |
| Brokerage model      | Coinbase (cash account) with realistic fees            |

### Running a LEAN backtest directly

```bash
lean backtest qtrader
```

With the VS Code debugger:

```bash
lean backtest qtrader --debug debugpy
```

The algorithm reads its parameters from a `params.json` file placed in the LEAN object store. The trainer scripts (`qtrader_trainer.py`, `qtrader_optimize.py`) handle writing this file automatically.

### Data requirements

LEAN must have access to BTCUSD minute-resolution data from 2016 onwards. Refer to the [LEAN data documentation](https://www.lean.io/docs/lean-cli/datasets/quantconnect) to download the required dataset.

---

## Monitoring

### TensorBoard

Training and evaluation metrics are logged via the built-in `TrainingLogger`:

```bash
tensorboard --logdir ./trials/tfboard
```

Logged metrics include Sharpe ratio, drawdown, number of trades, portfolio value, and per-step rewards.

### Optuna Dashboard

Inspect hyperparameter studies:

```bash
optuna-dashboard sqlite:///trials/hpo-study.db
```

### Notebooks

| Notebook                    | Purpose                                    |
|----------------------------|--------------------------------------------|
| `nb_qtrader_eval.ipynb`   | Analyze backtest results and trade history  |
| `nb_study_monitor.ipynb`  | Visualize Optuna study progress             |
| `research.ipynb`          | General research and experimentation        |

---

## Backtest Analysis

`bt_analysis.py` is a CLI tool that extracts, calculates, and formats backtest results from a LEAN output directory into an LLM-optimized Markdown summary.

```bash
python bt_analysis.py --dir <path_to_lean_output_folder>
python bt_analysis.py --dir <path_to_lean_output_folder> --show-trades
```

Instead of feeding raw, nested LEAN JSON files into an LLM — which wastes tokens and can cause hallucinations — this script produces a clean, highly compressed summary with tight key-value pairs, tables, and bullet points.

---

## Configuration Reference

### `qtrader_trainer.py` CLI

```
python qtrader_trainer.py --name NAME --iters N [-p key=value ...]
```

| Flag       | Description                          | Default |
|-----------|--------------------------------------|---------|
| `--name`  | Experiment / trial name              | —       |
| `--iters` | Number of training iterations        | 25      |
| `-p`      | Hyperparameter overrides (key=value) | —       |

### `qtrader_warmup.py` CLI

```
python qtrader_warmup.py --date-start YYYY-MM-DD --date-end YYYY-MM-DD --golden-cache-dir PATH
```

| Flag                 | Description                                     | Default        |
|---------------------|-------------------------------------------------|----------------|
| `--date-start`      | Start date for cache computation                | 2016-01-01     |
| `--date-end`        | End date for cache computation                  | 2026-02-01     |
| `--golden-cache-dir`| Output directory for the golden diskcache.Index | ../storage/cache |
| `--chunk-months`    | Months per parallel chunk                       | 4              |
| `--workers`         | Max parallel Docker workers                     | cpu_count - 2  |

### `qtrader_optimize.py` CLI

```
python qtrader_optimize.py --name NAME --num-trials N --path PATH --iters N
```

| Flag           | Description                             | Default     |
|---------------|-----------------------------------------|-------------|
| `--name`      | Optuna study name                        | study-test  |
| `--num-trials`| Number of optimization trials            | 1           |
| `--path`      | Path to Optuna study SQLite database     | None        |
| `--iters`     | Training iterations per trial            | 150         |

### `qtrader_optimize_runner.py` CLI

```
python qtrader_optimize_runner.py --name NAME --num-trials N --path PATH --iters N --n-jobs N
```

| Flag           | Description                             | Default     |
|---------------|-----------------------------------------|-------------|
| `--name`      | Optuna study name                        | study-test  |
| `--num-trials`| Total number of trials                   | 10          |
| `--path`      | Path to Optuna study SQLite database     | trial.db    |
| `--iters`     | Training iterations per trial            | 150         |
| `--n-jobs`    | Trials to run in parallel                | 2           |

### Agent hyperparameters (passed via `-p`)

| Key                     | Type    | Constructor Default | Description                               |
|-------------------------|---------|---------------------|-------------------------------------------|
| `expl_min`              | float   | 0.01                | Minimum exploration rate                  |
| `expl_decay`            | float   | 0.9                 | Exploration rate decay per checkpoint     |
| `exp_memory_size`       | int     | 365                 | Replay buffer capacity (raw value)        |
| `exp_mini_batch_size`   | int     | 128                 | Learning batch size                       |
| `exp_weighting`         | float   | 0.4                 | PER importance-sampling β (annealed)     |
| `exp_w_inc`             | float   | 0.0005              | IS β increment per learning step         |
| `exp_alpha`             | float   | 0.8                 | PER priority exponent                     |
| `n_steps_warmup`        | int     | 1,000               | Steps before learning starts              |
| `n_step_update`         | int     | 10                  | Steps between weight updates              |
| `n_steps_checkpoint`    | int     | 1,000               | Steps between checkpoints                 |
| `target_tau`            | float   | 0.001               | Target network soft update rate           |
| `model_n_layers`        | int     | 1                   | Number of hidden layers (default: [32])   |
| `model_fl_size`         | int     | 32                  | Hidden layer width                        |
| `model_shape`           | str     | `"cone"`            | `"cone"` (shrinking) or `"flat"`          |
| `model_act_func`        | str     | `"elu"`             | Hidden layer activation function          |
| `model_lr`              | float   | 1e-4                | Adam learning rate (clipnorm=1.0)         |
| `model_l2_reg`          | float   | 0.0                 | L2 regularization (commented out)         |
| `rl_gamma`              | float   | 0.9                 | Discount factor                           |
| `invest_pct`            | float   | 0.02                | Fraction of account per trade             |
| `hold_cost_scale`       | float   | 0.085               | Hold cost in R_bar units (0 = disabled)   |
| `exit_bonus_scale`      | float   | 5.0                 | Exit bonus in R_bar units (0 = disabled)  |
| `exit_loss_ratio`       | float   | 1.0                 | Loss as fraction of profit bonus          |
| `duration_bonus_scale`  | float   | 0.0                 | Duration bonus (0 = disabled)             |
| `action_cooldown_bars`  | int     | 0                   | Min bars before exit allowed              |
