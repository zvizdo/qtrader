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

```
┌─────────────────────────────────────────┐
│   Optimization Layer                    │
│   qtrader_optimize.py  (Optuna)         │
│       └── qtrader_trainer.py            │
│               └── N training iterations │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│   QuantConnect LEAN Backtest Engine     │
│   main.py :: QTraderAlgorithm           │
│                                         │
│   ┌───────────┐    ┌─────────────────┐  │
│   │ Market Env│───▶│  RL Pipeline    │  │
│   │ (OHLCV)   │    │ State → Act →   │  │
│   └───────────┘    │ Feedback → Learn│  │
│                    └────────┬────────┘  │
│                             │           │
│                    ┌────────▼────────┐  │
│                    │ State Providers │  │
│                    │ (Indicators)    │  │
│                    └────────┬────────┘  │
│                             │           │
│                    ┌────────▼────────┐  │
│                    │ SQLite + Cache  │  │
│                    │ (Golden Cache)  │  │
│                    └────────────────┘  │
└─────────────────────────────────────────┘
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
│   │   └── persistence.py      # SQLite persistence + caching layer
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
    --golden-db ../storage/cache/golden_cache.db
```

This runs `main.py` in `WARMUP` mode — it walks through every 15-minute bar, computes all state providers (Bridge Bands, MACD, OHLCV features), and writes the results to a SQLite database (`golden_cache.db`).

**What gets cached:**

- Bridge Bands (micro / daily / weekly scales)
- MACD (micro / daily / weekly scales)
- OHLCV features
- Account and position state snapshots

**Cache key format:** `Flow-State-{symbol}-{YYYYMMDDHHMM}-{ProviderClass}-{ParamsHash}`

Once generated, this file is reused by every training iteration.

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
2. The golden cache is copied into the iteration's storage directory
3. The in-memory cache is warmed for the selected date range
4. LEAN runs a backtest (`main.py` in `TRAIN` mode) — the agent observes 15-minute bars, takes actions, and learns via experience replay
5. Every `n_test` iterations (default 5), an evaluation run executes on a held-out period (2023-02-01 to 2026-01-31) with frozen weights

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

| Parameter            | Range                         |
|---------------------|-------------------------------|
| `expl_min`          | 0.01 – 0.15 (step 0.01)      |
| `n_steps_checkpoint`| 500 – 5000 (step 250)        |
| `exp_weighting`     | 0.3 – 0.7 (step 0.05)        |
| `exp_alpha`         | 0.3 – 0.8 (step 0.05)        |
| `model_lr`          | {1e-5, 5e-5, 1e-4}           |
| `rl_gamma`          | {0.9, 0.95, 0.99}            |

**Pruning rules** — trials are killed early if:

- Fewer than 10 trades after step 200
- Max drawdown exceeds 50%
- Sharpe ratio drops below -1.0
- Performance is in the bottom 33rd percentile (after 5 startup trials)

**Objective:** maximize Sharpe ratio on the evaluation period.

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
(indicators,            (60-dim feature     (ε-greedy        (reward calc,
 position info,          vector)             action           experience replay,
 account info)                               selection)       weight update)
```

**Step by step:**

1. **State computation** — each state provider (Bridge Bands, MACD, OHLCV, position, account) runs and its output is cached
2. **Aggregation** — all provider outputs are combined into a flat 60-dimensional feature vector
3. **Action** — the agent picks `FLAT` (exit/hold) or `LONG` (enter/hold) via ε-greedy policy
4. **Feedback** — reward is computed from portfolio return and holding penalty
5. **Learn** — every `n_step_update` steps, the agent samples a mini-batch from the replay buffer and updates the online network

### Agent (DQTP)

The agent is a **Double DQN** with **Prioritized Experience Replay**:

- **Online network** — predicts Q-values, updated via gradient descent
- **Target network** — provides stable TD targets, updated via soft copy: `θ_target ← (1-τ)·θ_target + τ·θ_online`
- **Replay buffer** — segment-tree based prioritized sampling (higher TD-error = higher sampling probability)

**Network architecture:**

```
Input (60 features)
  → LayerNormalization
  → Dense(128, ELU) + optional L2 regularization
  → Dense(64, ELU)      ← "cone" shape (shrinking layers)
  → Dense(2, Linear)    ← Q-values for [FLAT, LONG]
```

**Key hyperparameters:**

| Parameter             | Default      | Description                               |
|-----------------------|-------------|-------------------------------------------|
| `expl_max`            | 1.0         | Initial exploration rate                   |
| `expl_min`            | 0.01        | Minimum exploration rate                   |
| `expl_decay`          | 0.9995      | Exploration decay per checkpoint           |
| `exp_memory_size`     | 36,500,000  | Replay buffer capacity                     |
| `exp_mini_batch_size` | 32          | Mini-batch size for learning               |
| `exp_alpha`           | 0.8         | Priority exponent (α in PER)              |
| `n_steps_warmup`      | 10,240      | Steps before first weight update           |
| `n_step_update`       | 96          | Update frequency (~1 day at 15m bars)     |
| `target_tau`          | 0.005       | Soft update rate for target network        |
| `rl_gamma`            | 0.9         | Discount factor                            |
| `model_lr`            | 1e-4        | Adam learning rate                         |
| `invest_pct`          | 0.05        | Fraction of account value per trade (5%)  |

### Feature Engineering

The agent observes a **60-dimensional** feature vector built from multi-timeframe indicators:

| Group                  | Features | Description                                              |
|-----------------------|----------|----------------------------------------------------------|
| Time encoding          | 6        | sin/cos of weekday, hour, minute                         |
| Position info          | 5        | sign, ROE, trade count, days since/in trade              |
| OHLC (normalized)      | 3        | open/high/low relative to close, scaled by BB width      |
| Bridge Bands (×3 scales) | 27     | band width, band position, Hurst exponent at 3 lookbacks |
| MACD (×3 scales)       | 18       | MACD value and histogram at 3 lookbacks                  |
| **Total**              | **60**   |                                                          |

**Multi-timeframe scales:**

| Scale  | Bridge Bands Lookback | MACD (short/long/signal) | Captures            |
|--------|----------------------|--------------------------|---------------------|
| Micro  | 14 bars (3.5h)       | 12 / 26 / 9             | Intraday volatility  |
| Daily  | 96 bars (1 day)      | 96 / 288 / 96           | Swing momentum       |
| Weekly | 480 bars (5 days)    | 480 / 960 / 192         | Macro trend          |

Each indicator is also observed at **3 lookback windows** (1, 4, 16 bars back), giving the agent a sense of feature trajectory.

### Reward Function

```
R(t) = P(t) × (log_return - commission_fraction) × scale  -  P(t) × holding_penalty
```

Where:

- **P(t)** = 1 if in a LONG position, 0 if FLAT
- **log_return** = log(price_t / price_{t-1})
- **commission_fraction** = commission paid / account value
- **scale** = 960 (normalizes daily return to ~1.0)
- **holding_penalty** = `α × (exp(days_in_trade / τ) - 1)` with α=0.002/96 per step and τ=14 days

The penalty discourages holding losing positions indefinitely while the log-return reward aligns the agent with profitable trading.

### Caching & Persistence

The persistence layer is a hierarchy of providers optimized for throughput:

```
LeanCachedSQLitePersistenceProvider     ← used in main.py
  ├── In-memory read cache (LRU, 1M entries)
  ├── Batched writes (flush every 256 ops)
  ├── SQLite with WAL mode
  └── LEAN Object Store integration
```

**Golden cache workflow:**

1. `qtrader_warmup.py` runs once over the full date range, populating `golden_cache.db`
2. At the start of each training iteration, `golden_cache.db` is copied to the trial directory
3. `warm_cache_for_range()` pre-loads the relevant date range into memory
4. During the backtest, cache hits are served from memory — cache misses fall through to SQLite

**Serialization:** [msgpack](https://msgpack.org/) (binary, fast) with gzip+JSON fallback for legacy data.

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
python qtrader_warmup.py --date-start YYYY-MM-DD --date-end YYYY-MM-DD --golden-db PATH
```

| Flag           | Description                              |
|---------------|------------------------------------------|
| `--date-start`| Start date for cache computation         |
| `--date-end`  | End date for cache computation           |
| `--golden-db` | Output path for the golden cache SQLite  |

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

| Key                   | Type    | Default       | Description                          |
|-----------------------|---------|---------------|--------------------------------------|
| `expl_min`            | float   | 0.01          | Minimum exploration rate             |
| `expl_decay`          | float   | 0.9995        | Exploration rate decay               |
| `exp_memory_size`     | int     | 36,500,000    | Replay buffer size                   |
| `exp_mini_batch_size` | int     | 32            | Learning batch size                  |
| `exp_weighting`       | float   | 0.4           | PER importance sampling beta         |
| `exp_alpha`           | float   | 0.8           | PER priority exponent                |
| `n_steps_warmup`      | int     | 10,240        | Steps before learning starts         |
| `n_step_update`       | int     | 96            | Steps between weight updates         |
| `n_steps_checkpoint`  | int     | 5,000         | Steps between checkpoints            |
| `target_tau`          | float   | 0.005         | Target network soft update rate      |
| `model_n_layers`      | int     | 2             | Number of hidden layers              |
| `model_fl_size`       | int     | 128           | First layer width                    |
| `model_shape`         | str     | `"cone"`      | `"cone"` (shrinking) or `"flat"`     |
| `model_lr`            | float   | 1e-4          | Adam learning rate                   |
| `model_l2_reg`        | float   | 0.0           | L2 regularization strength           |
| `rl_gamma`            | float   | 0.9           | Discount factor                      |
| `invest_pct`          | float   | 0.05          | Fraction of account per trade        |
