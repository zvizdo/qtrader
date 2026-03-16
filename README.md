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
- [Research Notebooks](#research-notebooks)
- [Configuration Reference](#configuration-reference)

---

## Architecture Overview

At the top level, an **Optuna study** searches over hyperparameters. Each trial spins up a **trainer** that runs hundreds of LEAN backtests — every backtest is a full episode where the agent observes 15-minute bars, decides to go long or stay flat, and learns from the outcome via experience replay. Between episodes, the trainer samples random date windows, decays exploration, and periodically evaluates on a held-out period.

Inside each backtest the heavy lifting happens in four stages that fire on every consolidated bar: **state providers** (Bridge Bands, MACD, OHLCV, volume, position & account features) produce a 70-dim observation → the **DQTP agent** picks an action via ε-greedy Double DQN → the **feedback module** computes a shaped reward from log-returns and a holding penalty → the **learning step** samples a prioritized mini-batch and updates the online network. A pre-computed **golden cache** (diskcache.Index) ensures indicator values are calculated only once across all runs.

```
┌──────────────────────────────────────────────┐
│  Optimization Layer                          │
│  qtrader_optimize.py  (Optuna HPO)           │
│      └── qtrader_trainer.py                  │
│              └── N episodes (LEAN backtests)  │
└───────────────┬──────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────┐
│  QuantConnect LEAN Backtest Engine           │
│  main.py :: QTraderAlgorithm                 │
│                                              │
│  ┌────────────┐   ┌───────────────────────┐  │
│  │ Market Env │──▶│  RL Pipeline          │  │
│  │ (15m OHLCV)│   │  State → Act →        │  │
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
├── qtrader_optimize.py         # Optuna study manager
├── qtrader_warmup.py           # Golden cache generator
├── config.json                 # LEAN configuration
├── requirements.txt            # Python dependencies
│
├── qtrader/                    # Core package
│   ├── agents/
│   │   ├── base.py             # BaseAgent interface
│   │   ├── dqtp.py             # DQTPAgent — Double DQN + prioritized replay
│   │   └── expreplay/
│   │       └── buffer.py       # Segment-tree prioritized replay buffer
│   │
│   ├── environments/
│   │   ├── base.py             # BaseMarketEnv interface
│   │   ├── lean.py             # QuantConnect LEAN market environment
│   │   ├── backtesting.py      # Standalone backtesting environment
│   │   └── bitstamp.py         # Live trading environment (Bitstamp)
│   │
│   ├── stateproviders/
│   │   ├── __init__.py         # Base classes
│   │   ├── basic.py            # Account, Position, Trade, OHLCV providers
│   │   ├── indicators.py       # Bridge Bands, MACD, Trendlines
│   │   └── model.py            # ML-based providers
│   │
│   ├── rlflow/
│   │   ├── state.py            # State provider tasks & aggregation
│   │   ├── action.py           # Action execution
│   │   ├── feedback.py         # Reward calculation
│   │   ├── learn.py            # Learning coordination
│   │   └── persistence.py      # diskcache.Index persistence + caching layer
│   │
│   ├── logging/
│   │   └── tb_logger.py        # TensorBoard logger
│   │
│   └── dashboard/              # Streamlit monitoring dashboard
│       └── main.py
│
├── playground/                 # Research notebooks
│   ├── nb_bridgebands.ipynb
│   ├── nb_macd.ipynb
│   └── nb_trendlines.ipynb
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
| `tensorflow`| 2.20.*   | Neural network (DQN model)       |
| `optuna`    | 4.*      | Hyperparameter optimization      |
| `msgpack`   | 1.1.2    | Fast binary serialization        |

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

This runs `main.py` in `WARMUP` mode — it walks through every 15-minute bar, computes all state providers (Bridge Bands, MACD, OHLCV features), and writes the results to a `diskcache.Index` directory. The date range is split into chunks and processed in parallel via separate Docker containers, then merged into a single index.

**What gets cached:**

- Bridge Bands (micro / daily / weekly scales)
- MACD (micro / daily / weekly scales)
- OHLCV features
- Account and position state snapshots

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

1. A random 3-year date window is sampled from the historical data
2. The golden cache directory is mounted as a read-through cache provider
3. LEAN runs a backtest (`main.py` in `TRAIN` mode) — the agent observes 15-minute bars, takes actions, and learns via experience replay
4. Every `n_test` iterations (default 5), an evaluation run executes on a held-out period with frozen weights

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
python qtrader_optimize.py \
    --name "hpo-study" \
    --num-trials 50 \
    --path "../trials/hpo-study.db"
```

**Search space** (defined in `qtrader_optimize.py`):

| Parameter            | Type        | Range / Values              |
|---------------------|-------------|-----------------------------|
| `expl_min`          | float       | 0.01 – 0.10 (step 0.01)    |
| `n_steps_checkpoint`| int         | 500 – 5,000 (step 250)     |
| `alpha_dd`          | float       | 2.0 – 10.0 (step 1.0)      |
| `exp_alpha`         | float       | 0.4 – 0.8 (step 0.1)       |

**Static parameters** (fixed across all trials):

| Parameter            | Value       | Notes                                  |
|---------------------|-------------|----------------------------------------|
| `invest_pct`        | (0.05, 0.5) | Uniformly randomized per iteration    |
| `eval_invest_pct`   | 0.25        | Fixed during evaluation runs           |
| `expl_decay`        | 0.9925      | Per-checkpoint decay                   |
| `n_step_update`     | 8           | Weight update every 8 steps            |
| `exp_memory_size`   | ~4.9M       | 7 years × 96 bars/day × 2             |
| `exp_mini_batch_size`| 128        |                                        |
| `exp_weighting`     | 0.4         | IS beta starting value                 |
| `exp_w_inc`         | 5e-5        | IS beta annealing increment            |
| `model_lr`          | 1e-4        | Adam learning rate                     |
| `rl_gamma`          | 0.9         | Discount factor                        |
| `target_tau`        | 0.001       | Target network soft update rate        |
| `model_n_layers`    | 1           | Single hidden layer                    |
| `model_fl_size`     | 256         | Hidden layer width                     |
| `iters`             | 500         | Episodes per trial                     |
| `n_test`            | 10          | Evaluate every 10 iterations           |

**Objective:** maximize Sharpe ratio on the evaluation period.

**Pruning:** Optuna's `PercentilePruner` drops the bottom third of trials after 200 steps, with 5 startup trials and a minimum of 3 completed trials before pruning kicks in.

### 4. Evaluate a Trained Model

Evaluation happens automatically during training (every `n_test` iterations). You can also run a standalone evaluation by setting the `run_type` to `EVAL` in the params passed to `main.py`.

Results are logged to TensorBoard and saved as LEAN backtest results in `backtests/`.

---

## How It Works

### Run Modes

The algorithm (`main.py`) supports four modes, controlled by the `run_type` parameter:

| Mode     | Learning | Exploration | Purpose                                     |
|----------|----------|-------------|---------------------------------------------|
| `WARMUP` | Off      | N/A         | Pre-compute and cache all indicator values   |
| `TRAIN`  | On       | Decaying    | Train the agent via experience replay        |
| `EVAL`   | Off      | Off         | Evaluate with frozen weights (greedy policy) |
| `LIVE`   | Off      | Off         | Live trading with full state persistence     |

### RL Pipeline

On every 15-minute consolidated bar, the pipeline executes:

```
State Providers    →    Aggregation    →    Agent Act    →    Feedback & Learn
(indicators,            (70-dim feature     (ε-greedy        (reward calc,
 position info,          vector)             action           experience replay,
 account info)                               selection)       weight update)
```

**Step by step:**

1. **State computation** — each state provider (Bridge Bands, MACD, OHLCV, position, account) runs and its output is cached
2. **Aggregation** — all provider outputs are combined into a flat 70-dimensional feature vector
3. **Action** — the agent picks `FLAT` (exit/hold) or `LONG` (enter/hold) via ε-greedy policy
4. **Feedback** — reward is computed from market return, trade cost, and drawdown penalty
5. **Learn** — every `n_step_update` steps, the agent samples a mini-batch from the replay buffer and updates the online network

### Agent (DQTP)

The agent is a **Double DQN** with **Prioritized Experience Replay**:

- **Online network** — predicts Q-values, updated via gradient descent
- **Target network** — provides stable TD targets, updated via soft copy: `θ_target ← (1-τ)·θ_target + τ·θ_online`
- **Replay buffer** — segment-tree based prioritized sampling (higher TD-error = higher sampling probability). New experiences are inserted at `max_priority`, which soft-decays (×0.999) toward the actual observed maximum to prevent stale priority inflation

**Key hyperparameters:**

| Parameter             | Default  | Description                                      |
|-----------------------|----------|--------------------------------------------------|
| `expl_max`            | 1.0      | Initial exploration rate                          |
| `expl_min`            | 0.01     | Minimum exploration rate                          |
| `expl_decay`          | 0.9      | Exploration decay per checkpoint                  |
| `exp_memory_size`     | 365      | Replay buffer capacity (passed as raw value)      |
| `exp_mini_batch_size` | 128      | Mini-batch size for learning                      |
| `exp_alpha`           | 0.8      | Priority exponent (α in PER)                     |
| `exp_weighting`       | 0.4      | Importance-sampling β (annealed toward 1.0)      |
| `exp_w_inc`           | 0.0005   | IS β increment per learning step                 |
| `n_steps_warmup`      | 1,000    | Steps before first weight update                  |
| `n_step_update`       | 10       | Steps between weight updates                      |
| `n_steps_checkpoint`  | 1,000    | Steps between exploration decay & model save      |
| `target_tau`          | 0.001    | Soft update rate for target network               |
| `rl_gamma`            | 0.9      | Discount factor                                   |
| `model_lr`            | 1e-4     | Adam learning rate (with gradient clipnorm=1.0)   |
| `model_l2_reg`        | 0.0      | L2 regularization strength                        |
| `invest_pct`          | 0.02     | Fraction of account value per trade               |

Note: the target network is updated on **every learning step** (not on a separate schedule), making `target_tau` the primary knob for controlling target lag. Q-value targets are clipped to [-15, 15] and TD errors are clipped to [0, 5] to prevent runaway bootstrapping.

### Feature Engineering

The agent observes a **70-dimensional** feature vector built from multi-timeframe indicators:

| Group                    | Features | Description                                              |
|-------------------------|----------|----------------------------------------------------------|
| Time encoding            | 6        | sin/cos of weekday, hour, minute                         |
| Position info            | 5        | sign, ROE (position-relative), trade count, days since/in trade |
| OHLC (normalized)        | 3        | open/high/low relative to close, scaled by BB width      |
| Candle body ratio        | 1        | abs(close-open) / (high-low) — conviction vs indecision  |
| Log-returns              | 3        | raw log-return at 1, 4, 16 bars back (×100)              |
| Relative volume          | 3        | log1p(volume / mean_16) at 1, 4, 16 bars back            |
| Bridge Bands (×3 scales) | 27       | band width, band position, Hurst exponent at 3 lookbacks |
| BB position velocity     | 3        | 4-bar change in band position per scale                  |
| MACD (×3 scales)         | 18       | MACD value and histogram at 3 lookbacks                  |
| **Total**                | **70**   |                                                          |

**Multi-timeframe scales:**

| Scale  | Bridge Bands Lookback | MACD (short/long/signal) | Captures            |
|--------|----------------------|--------------------------|---------------------|
| Micro  | 14 bars (3.5h)       | 12 / 26 / 9             | Intraday volatility  |
| Daily  | 96 bars (1 day)      | 96 / 288 / 96           | Swing momentum       |
| Weekly | 480 bars (5 days)    | 480 / 960 / 192         | Macro trend          |

Each indicator is also observed at **3 lookback windows** (1, 4, 16 bars back), giving the agent a sense of feature trajectory.

### Reward Function

```
R(t) = P(t) × log_return × scale
     - comm_frac × scale
     - P(t) × α_dd × max(-ROE, 0)
```

Three independent terms:

1. **Market return** — only received while LONG, gated by P(t)
2. **Trade cost** — commission penalty applied on every position change (entry *and* exit), independent of P(t)
3. **Drawdown penalty** — proportional to how underwater the current position is; winning positions incur zero penalty regardless of hold duration

Where:

- **P(t)** = 1 if in a LONG position, 0 if FLAT
- **log_return** = log(price_t / price_{t-1})
- **comm_frac** = commission / position notional (position-relative, not account-relative)
- **ROE** = unrealized profit / cost basis of the current position
- **scale** = 150 (targets ~90% of raw rewards in [-1, 1] for BTC 15-minute σ ≈ 0.004)
- **α_dd** = 5.0 (drawdown penalty coefficient)

Drawdown penalty at typical ROE levels (vs a ±0.6 typical bar reward):

| Position ROE | Penalty | Relative to bar |
|---|---|---|
| -1% | 0.05 | ~8% — gentle nudge |
| -5% | 0.25 | ~40% — real pressure |
| -10% | 0.50 | ~80% — strong exit signal |
| ≥ 0% | 0.00 | no penalty |

The commission is measured relative to the position notional (`commission / (account_value × invest_pct)`), making it directly comparable to the market log-return regardless of position sizing.

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
| `TradeBarConsolidator` | Consolidates 1-minute bars into 15-minute bars       |
| `history()`          | Retrieves historical OHLCV for indicator warmup        |
| `market_order()`     | Executes BUY/SELL orders                               |
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

## Configuration Reference

### `qtrader_trainer.py` CLI

```
python qtrader_trainer.py --name NAME --iters N [-p key=value ...]
```

| Flag       | Description                          | Default |
|-----------|--------------------------------------|---------|
| `--name`  | Experiment / trial name              | —       |
| `--iters` | Number of training iterations        | —       |
| `-p`      | Hyperparameter overrides (key=value) | —       |

### `qtrader_warmup.py` CLI

```
python qtrader_warmup.py --date-start YYYY-MM-DD --date-end YYYY-MM-DD --golden-cache-dir PATH
```

| Flag                 | Description                                     |
|---------------------|-------------------------------------------------|
| `--date-start`      | Start date for cache computation                |
| `--date-end`        | End date for cache computation                  |
| `--golden-cache-dir`| Output directory for the golden diskcache.Index |
| `--chunk-months`    | Months per parallel chunk (default: 4)          |
| `--workers`         | Max parallel Docker workers                     |

### `qtrader_optimize.py` CLI

```
python qtrader_optimize.py --name NAME --num-trials N --path PATH
```

| Flag           | Description                             |
|---------------|------------------------------------------|
| `--name`      | Optuna study name                        |
| `--num-trials`| Number of optimization trials            |
| `--path`      | Path to Optuna study SQLite database     |

### Agent hyperparameters (passed via `-p`)

| Key                   | Type    | Default       | Description                               |
|-----------------------|---------|---------------|-------------------------------------------|
| `expl_min`            | float   | 0.01          | Minimum exploration rate                  |
| `expl_decay`          | float   | 0.9           | Exploration rate decay per checkpoint     |
| `exp_memory_size`     | int     | 365           | Replay buffer capacity (raw value)        |
| `exp_mini_batch_size` | int     | 128           | Learning batch size                       |
| `exp_weighting`       | float   | 0.4           | PER importance-sampling β (annealed)     |
| `exp_w_inc`           | float   | 0.0005        | IS β increment per learning step         |
| `exp_alpha`           | float   | 0.8           | PER priority exponent                     |
| `n_steps_warmup`      | int     | 1,000         | Steps before learning starts              |
| `n_step_update`       | int     | 10            | Steps between weight updates              |
| `n_steps_checkpoint`  | int     | 1,000         | Steps between checkpoints                 |
| `target_tau`          | float   | 0.001         | Target network soft update rate           |
| `model_n_layers`      | int     | 2             | Number of hidden layers                   |
| `model_fl_size`       | int     | 128           | First layer width                         |
| `model_shape`         | str     | `"cone"`      | `"cone"` (shrinking) or `"flat"`          |
| `model_lr`            | float   | 1e-4          | Adam learning rate (clipnorm=1.0)         |
| `model_l2_reg`        | float   | 0.0           | L2 regularization strength                |
| `rl_gamma`            | float   | 0.9           | Discount factor                           |
| `invest_pct`          | float   | 0.02          | Fraction of account per trade             |
