# QTrader Agent Architecture

Double DQN + Prioritized Experience Replay trading bot for **BTCUSD 1-hour bars** on QuantConnect LEAN.

## File Map

```
main.py                          # LEAN QCAlgorithm entry point — all run modes
qtrader_trainer.py               # Training loop: TRAIN iterations + periodic EVAL
qtrader_optimize.py              # Optuna HPO (single trial per process)
qtrader_optimize_runner.py       # Spawns parallel qtrader_optimize.py processes
qtrader_warmup.py                # Parallel pre-computation of indicator cache

qtrader/
  agents/
    base.py                      # BaseAgent, RandomAgent
    dqtp.py                      # DQTPAgent — core agent (act, feedback, learn, features, model)
    expreplay/buffer.py          # PrioritizedReplayBuffer (segment trees, O(log N) sampling)
  environments/
    base.py                      # BaseMarketEnv ABC + get_order_pnl() (FIFO PnL calc)
    lean.py                      # LeanMarketEnv — bridges LEAN API to BaseMarketEnv
    bitstamp.py                  # Legacy Bitstamp env (unused)
    backtesting.py               # Legacy backtesting env (unused)
  stateproviders/
    __init__.py                  # BaseStateProvider, BaseSymbolStateProvider
    basic.py                     # AccountInfo, Position, Trade, OHLCV providers
    indicators.py                # BridgeBands, MACD, TrendMaturity, Trendlines (trendlines unused)
    model.py                     # (exists but empty/unused)
  rlflow/
    __init__.py                  # BaseTask
    state.py                     # StateProviderTask (w/ cache), StateAggregatorTask
    action.py                    # ActTask — calls agent.act(), executes orders
    feedback.py                  # FeedbackTask — computes v_po_change, calls agent.feedback()
    persistence.py               # Persistence providers (diskcache.Index, SQLite, filesystem)
  logging/
    __init__.py                  # exports TrainingLogger
    tb_logger.py                 # TensorBoard logging for train/eval metrics
  dashboard/                     # Streamlit diagnostic dashboard (separate tool)
```

## Data Flow Per Bar

```
1-hour consolidated bar fires _on_consolidated_bar() [main.py:322]
  |
  +-> State providers compute state [main.py:335-390]
  |     AccountInfo, Position, Trade, OHLCV, BB(micro/daily/weekly), MACD(micro/daily/weekly), TrendMaturity
  |     -> StateAggregatorTask merges into single state dict [main.py:376-390]
  |
  +-> ActTask [main.py:400]
  |     agent.act(state) -> epsilon-greedy or model prediction
  |     _shape_action() maps to BUY/CLOSE_POSITION/DO_NOTHING
  |     Orders executed via LeanMarketEnv
  |
  +-> FeedbackTask [main.py:413]
  |     Computes v_po_change (portfolio % change)
  |     Calls agent.feedback(state_prev, action, reward, state_current)
  |       -> enriches reward with market_log_return, comm_frac, hold_hours, trade_pnl_pct
  |       -> generates feature vectors, adds experience to replay buffer
  |
  +-> ready_to_learn() check [main.py:421]
        If n_steps > warmup AND buffer >= batch_size AND n_steps % n_step_update == 0:
          -> soft-update target network (Polyak averaging)
          -> agent.learn() (Double DQN update)
        If n_steps % n_steps_checkpoint == 0:
          -> decay exploration rate, save model checkpoints
```

## Run Modes

| Mode | Learning | Exploration | Agent created | Purpose |
|------|----------|-------------|---------------|---------|
| WARMUP | N/A | N/A | No (None) | Pre-compute & cache indicator state |
| TRAIN | On | epsilon-greedy | Yes | Learn from random 140-day windows |
| EVAL | Off | Off (greedy) | Yes (no_learn=True) | Evaluate on 280-day held-out windows |
| LIVE | Off | Off | Yes (no_learn=True) | Live trading |

## Training Loop (`qtrader_trainer.py`)

1. `np.random.seed(42 + step)` per iteration (deterministic windows across trials)
2. Sample random 140-day window from 2016-01-01 to 2023-01-01 [L139-141]
3. Run LEAN backtest in TRAIN mode
4. Every `n_test` iterations (default 10 in optimize, configurable): run EVAL on random 280-day window from 2023-02-01 to 2026-01-31 [L143-145]
5. Final iteration: EVAL spans the full eval period
6. `invest_pct` can be randomized per train iteration via tuple range, e.g. `(0.05, 0.5)` [L191-193]

## DQTPAgent (`agents/dqtp.py`)

### Constructor Defaults vs Optimize Defaults

| Param | dqtp.py default | qtrader_optimize.py value |
|-------|----------------|--------------------------|
| `model_layers` | `[32]` | `[64]` (1 layer, 64 units) |
| `exp_memory_size` | `365` | `50,000` |
| `exp_mini_batch_size` | `128` | `256` |
| `n_step_update` | `10` | `4` |
| `n_steps_warmup` | `1000` | `2,500` |
| `n_steps_checkpoint` | `1000` | tuned 125-500 |
| `rl_gamma` | `0.9` | `0.986` |
| `action_cooldown_bars` | `0` | `2` |
| `expl_decay` | `0.9` | `0.9925` |
| `target_tau` | `0.001` | `0.005` |
| `model_lr` | `0.0001` | `1e-5` |
| `bar_period_seconds` | `3600` | (passed from main.py) |

**Important**: the constructor defaults are fallbacks. Actual training always uses values from `params.json` passed through `main.py:_create_agent()` [L71-108].

### Neural Network (`_create_model`, L633-665)

```
Input (71 features)
  -> LayerNormalization
  -> Dense(N, activation=model_act_func, GlorotUniform(seed=42))  # repeated per layer
  -> Dense(2, activation='linear')   # Q(FLAT), Q(LONG)

Loss:      Huber(delta=3.0)
Optimizer: Adam(lr=model_lr)
```

`model_shape="cone"`: each layer halves in size. `model_shape="flat"`: all layers same size.

### Feature Vector (72 dimensions, `_generate_example` L362-461)

| Idx | Group | Count | Computation |
|-----|-------|-------|-------------|
| 0-1 | Time | 2 | sin/cos of weekday(/3) |
| 2 | Position sign | 1 | 0=flat, 1=long |
| 3 | Return on position | 1 | `sign(rop) * log1p(100 * |rop|)` where `rop = profit / cost_basis` |
| 4 | Hold duration | 1 | `min(hold_hours, 360) / 72`, 0 when flat (1.0 = hold cost threshold) |
| 5-7 | PnL trajectory | 3 | `log(close[-i] / entry_price) * 100` for lookbacks [1, 4, 12], 0 when flat or hold_bars < lookback |
| 8-10 | Vol-norm OHLC | 3 | `(1 - open/close) / bb_micro_width` for open, high, low |
| 11 | Candle body ratio | 1 | `|close - open| / (high - low)` |
| 12-14 | Log-returns | 3 | `ln(close/close[-i]) * 100` for lookbacks [1, 24, 72] |
| 15-17 | Relative volume | 3 | `log1p(volume[-i] / mean(volume[-72:]))` for [1, 24, 72] |
| 18-44 | Bridge Bands | 27 | 3 scales x 3 lookbacks x 3 values (width, pos, hurst) |
| 45-62 | MACD | 18 | 3 scales x 3 lookbacks x 2 values (macd, hist) |
| 63-71 | Trend Maturity | 9 | Current bar only: direction, exhaustion, bull/bear waves, extending, retrace depth, impulse ratio, dir bias, efficiency |

**Bridge Bands scales** (from `main.py` state provider params):
- Micro: range/BB/hurst = 12 bars (12h), days_ago=4
- Daily: range/BB/hurst = 96 bars (4 days), days_ago=10
- Weekly: range/BB/hurst = 336 bars (14 days), days_ago=30

**MACD scales**:
- Micro: 6/13/4 (days_ago=4)
- Daily: 24/72/24 (days_ago=10)
- Weekly: 112/240/48 (days_ago=30)

**Trend Maturity** (swing-structure features):
- `swing_order=7` (7h each side), `lookback=120` (5 days), `days_ago=10`
- 9 features (current bar only, no lookbacks — TM is a windowed aggregate with high autocorrelation): direction, exhaustion, bull_waves, bear_waves, extending, retrace_depth, impulse_ratio, dir_bias, efficiency

State keys in `sym_state`: `bridge_bnds_micro`, `bridge_bnds_daily`, `bridge_bnds_weekly`, `macd_6_13_4`, `macd_24_72_24`, `macd_112_240_48`, `trend_maturity`

### Reward Function (`_reward_active`)

All shaping components are anchored to `_R_BAR = sigma_1h × REWARD_SCALE ≈ 0.59` (the per-bar noise floor). Each component disables when its scale=0.

```python
R = p_t * (clip(market_log_return * 75, -1, 1) - hold_cost) - trade_cost + exit_bonus + duration_bonus - opp_cost
```

- `p_t` = 1 if LONG, 0 if FLAT
- `opp_cost` (FLAT only): `opp_cost_scale * max(0, market_return)` — penalizes missed upside, zero when market falls
- `trade_cost = comm_frac * 75` (only on position changes)
- `hold_cost`: 0 if hold_hours ≤ 72h, else `hold_cost_scale * R_BAR * ((hours-72)/24)^1.5`
- `exit_bonus` (on trade exit only): `exit_bonus_scale * R_BAR * tanh(trade_pnl_pct / 0.03)` for profit; losses scaled by `exit_loss_ratio`
- `duration_bonus` (on trade exit only): reverse-U peaking at 3d hold, `duration_bonus_scale * R_BAR * (1 - ((days-3)/2)^2)` if hold in [1d, 5d]

**Locked constants** (class attrs):
- `_REWARD_SCALE = 75.0` — scales log returns, ~93% coverage in [-1, 1]
- `_R_BAR = 0.59` — sigma × REWARD_SCALE, the core unit for all shaping
- `_TRADE_PNL_REF = 0.03` — median |trade_pnl_pct|, normalizes tanh input
- `_HOLD_THRESHOLD_HOURS = 72.0`
- `_HOLD_COST_POWER = 1.5`
- `_DURATION_PEAK_DAYS = 3.0`, `_DURATION_HALF_WIDTH = 2.0`

**Tunable** (instance attrs, all in R_bar units): `hold_cost_scale`, `exit_bonus_scale`, `exit_loss_ratio`, `duration_bonus_scale`, `opp_cost_scale`

### Reward Enrichment (`feedback`, L246-323)

The `feedback()` method enriches reward dict with fields the reward function needs:
- `position_prev_indicator`: 1.0 if had position before this step
- `market_log_return`: `ln(price_future / price_current)` [L270]
- `comm_frac`: commission / position_notional (only on position changes) [L279-285]
- `hold_hours`: seconds since trade start / 3600 [L288-298]
- `trade_pnl_pct`: `profit / cost_basis` on exit bars only [L301-305]

### Learning (`learn`, L525-631)

1. Sample `exp_mini_batch_size` from PER buffer with IS weights
2. Batch predict: online model on current + future states [L557]
3. Double DQN target: online picks action, target evaluates [L583-592]
   - `q_clip = 1.5 / (1 - gamma)` (e.g. gamma=0.986 -> q_clip=107)
4. TD errors clipped to [0, 5] for priority updates [L594]
5. `model.fit()` with IS weights, 1 epoch [L607-616]
6. Target soft-update happens in `ready_to_learn()` [L339], before `learn()` is called

### Action Selection (`act`, L194-244)

- Exploration: `np.random.choice` with `exploration_bias = [0.5, 0.5]` masked by `_possible_actions`
- Exploitation: model predict, mask impossible actions with `-inf`, argmax
- `_possible_actions` [L159-171]: if `action_cooldown_bars > 0` and in position for fewer bars than cooldown, disable FLAT (idx 0). Uses `bar_period_seconds` for correct time-to-bars conversion.
- `_shape_action` [L173-192]: maps FLAT->CLOSE_POSITION or DO_NOTHING, LONG->BUY or DO_NOTHING

### Exploration Decay (`ready_to_learn`, L325-349)

- Every `n_steps_checkpoint` steps: `expl_rate *= expl_decay` (clamped to `expl_min`)
- IS beta: `exp_weighting += exp_w_inc` each learn step (clamped to 1.0)

### Persistence

- Config (expl_rate, n_steps, n_updates, n_checkpoints, exp_weighting): `QAgent-Params` dict
- Replay buffer: pickled to `QAgent-ReplayBuffer.pkl` (atomic write via tmp+replace)
- Models: `QAgent-Model-Online.keras`, `QAgent-Model-Target.keras`

## Replay Buffer (`expreplay/buffer.py`)

- Capacity rounded up to power of 2 internally, cycles at `max_capacity`
- Segment trees (sum + min) for O(log N) priority sampling
- Vectorized batch prefix-sum lookup [L116-128]
- `update_priorities`: clips to `MIN_PRIORITY = 1e-6`, soft-decays `max_priority *= 0.999`
- New samples added at `max_priority`

## LeanMarketEnv (`environments/lean.py`)

- `get_current_market_datetime()` [L24-38]: snaps to bar period boundaries using timedelta arithmetic
- `get_ohlcv()` [L76-81]: reads from `RollingWindow[TradeBar]` via `_fetch_ohlcv_cached` (LRU 128)
- `get_position()` [L101-111]: returns `{size, price_last, value, profit}` where value = price * qty, profit = unrealized
- `get_trades()` [L113-179]: groups filled orders into trades using `trade_builder.closed_trades`, caches order data in `_order_cache`
- `execute_buy_market()` [L83-84]: `qcl.market_order()`
- `execute_close_position()` [L98-99]: `qcl.liquidate()`

## Persistence Providers (`rlflow/persistence.py`)

| Provider | Backend | Used by |
|----------|---------|---------|
| `DiskIndexPersistenceProvider` | diskcache.Index + msgpack | trainer (host-side golden cache) |
| `LeanDiskIndexPersistenceProvider` | diskcache.Index via LEAN object_store | main.py (all modes) |
| `NoPersistenceProvider` | no-op | testing |
| `FileSystemPersistenceProvider` | gzip+JSON files | legacy |
| `SQLitePersistenceProvider` | SQLite + gzip | legacy |
| `CachedSQLitePersistenceProvider` | SQLite + msgpack + in-memory cache | legacy |

Active stack: `LeanDiskIndexPersistenceProvider` with optional `DiskIndexPersistenceProvider` as read-through cache (TRAIN/EVAL modes).

## State Providers

**`StateProviderTask`** (`rlflow/state.py` L12-73): wraps any state provider with optional diskcache caching. Cache key: `Flow-State-{symbol}-{YYYYMMDDHHMM}-{ClassName}-{params_md5}`.

| Provider | Returns key | Cached | days_ago | cache_truncate |
|----------|------------|--------|----------|----------------|
| `AccountInfoStateProvider` | `account` (global) | No | N/A | N/A |
| `PositionSymbolStateProvider` | `position` | No | N/A | N/A |
| `TradeSymbolStateProvider` | `trade`, `trades` | No | N/A | N/A |
| `OHLCVSymbolStateProvider` | `ohlcv` | Yes | 4 | 73 |
| `BridgeBandsSymbolStateProvider` | `bridge_bnds_{scale}` | Yes | 4/10/30 | 73 |
| `MACDSymbolStateProvider` | `macd_{s}_{l}_{sig}` | Yes | 4/10/30 | 73 |
| `TrendMaturitySymbolStateProvider` | `trend_maturity` | Yes | 14 | 73 |

All cached providers use `cache_truncate=73` to support `_LOOKBACKS[-1]=72`.

## main.py Key Details

- `BAR_PERIOD = timedelta(hours=1)` [L57]
- `history_window = RollingWindow[TradeBar](1400)` (~58 days) [L141]
- Warmup: fetches 60 days of minute bars, pushes through consolidator [L150-152]
- Cash: $1000, Coinbase brokerage model [L128-130]
- `LatestPriceFillModel` on exchange [L136]
- Last bar: liquidates all, returns (no agent action) [L326-331]
- `on_end_of_algorithm`: saves model+config (TRAIN only), logs summary stats [L430-497]

## Optuna HPO (`qtrader_optimize.py`)

- Objective: Sharpe ratio on final full-period eval
- Pruner: `PercentilePruner(percentile=33, n_startup_trials=5, n_warmup_steps=200, n_min_trials=3)`
- 500 iterations per trial, eval every 10 iterations
- Tuned params: `expl_min`, `n_steps_checkpoint`, `exp_alpha`, `hold_cost_scale`, `exit_bonus_scale`
- Static params include `action_cooldown_bars=2`, `model_fl_size=64`, 1 layer, `rl_gamma=0.986`
- Weighted Sharpe for pruning: last 5 evals with weights [0.45, 0.25, 0.15, 0.1, 0.05]

## Warmup (`qtrader_warmup.py`)

- Splits date range into N-month chunks (default 4 months)
- Runs each chunk as a LEAN backtest in WARMUP mode (parallel via `ProcessPoolExecutor`)
- Each chunk writes to isolated diskcache, then merged into golden cache sequentially
- Default range: 2016-01-01 to 2026-02-01
- **Note:** Warmup cache must be fully regenerated after any change to BB/MACD/TrendMaturity parameters or bar period, as cache keys include parameter hashes

## Backtest Analysis (`bt_analysis.py`)

A reusable CLI script to extract, calculate, and format backtest results from a QuantConnect/LEAN output directory into an LLM-optimized Markdown summary.

**Why use this tool?**
Instead of feeding raw, nested LEAN JSON files into an LLM for analysis—which wastes tokens and can cause hallucinations—this CLI parses the local LEAN data natively. It produces a clean, highly compressed summary with tight key-value pairs, tables, and bullet points so a downstream LLM can understand the backtest performance instantly.

**Usage:**
```bash
python bt_analysis.py --dir <path_to_lean_output_folder>
```

**Options:**
- `--show-trades`: Include a full trade-by-trade breakdown table (Trade #, Entry Date, Duration, PnL) at the end of the output.
