"""
Verify all statistical claims in 15M-1H-migration.md against raw data.
Uses 2016-01-01 to 2023-01-01 (training period) for consistency with the spec.
"""
import pandas as pd
import numpy as np
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "crypto" / "coinbase" / "minute" / "btcusd"

def load_day(date: datetime) -> pd.DataFrame:
    fname = date.strftime("%Y%m%d") + "_trade.zip"
    fpath = DATA_DIR / fname
    if not fpath.exists():
        return pd.DataFrame()
    with zipfile.ZipFile(fpath) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, header=None, names=["ms", "open", "high", "low", "close", "volume"])
    base = datetime(date.year, date.month, date.day)
    df["datetime"] = base + pd.to_timedelta(df["ms"], unit="ms")
    df = df.drop(columns=["ms"])
    return df

def load_range(start: datetime, end: datetime) -> pd.DataFrame:
    frames = []
    d = start
    while d <= end:
        df = load_day(d)
        if not df.empty:
            frames.append(df)
        d += timedelta(days=1)
    if not frames:
        print(f"No data found for range {start.date()} to {end.date()}")
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)

def resample_ohlcv(df_1m, freq):
    """Resample 1-minute data to given frequency."""
    df = df_1m.set_index("datetime")
    resampled = df.resample(freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    return resampled

print("=" * 70)
print("LOADING DATA: 2016-01-01 to 2023-01-01")
print("=" * 70)

df_1m = load_range(datetime(2016, 1, 1), datetime(2023, 1, 1))
print(f"Loaded {len(df_1m):,} 1-minute bars")
print(f"Date range: {df_1m['datetime'].min()} to {df_1m['datetime'].max()}")

# Resample to 15m, 1H, 4H, 1D
print("\nResampling...")
df_15m = resample_ohlcv(df_1m, "15min")
df_1h = resample_ohlcv(df_1m, "1h")
df_4h = resample_ohlcv(df_1m, "4h")
df_1d = resample_ohlcv(df_1m, "1D")

print(f"15m bars: {len(df_15m):,}")
print(f"1H bars:  {len(df_1h):,}")
print(f"4H bars:  {len(df_4h):,}")
print(f"1D bars:  {len(df_1d):,}")

# =============================================================================
# SECTION 1: Volatility Statistics
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1: VOLATILITY (log-return σ)")
print("=" * 70)

for label, df in [("15m", df_15m), ("1H", df_1h), ("4H", df_4h), ("1D", df_1d)]:
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    sigma = log_ret.std()
    mean_ret = log_ret.mean()
    inv_sigma = 1.0 / sigma
    print(f"  {label:>3s}: σ = {sigma:.6f}, mean = {mean_ret:.8f}, 1/σ = {inv_sigma:.1f}, N = {len(log_ret):,}")

log_ret_15m = np.log(df_15m["close"] / df_15m["close"].shift(1)).dropna()
log_ret_1h = np.log(df_1h["close"] / df_1h["close"].shift(1)).dropna()

sigma_15m = log_ret_15m.std()
sigma_1h = log_ret_1h.std()
ratio = sigma_15m / sigma_1h
sqrt4 = np.sqrt(4)
theoretical_ratio = 1.0 / sqrt4

print(f"\n  Spec claims: σ_15m = 0.004365, σ_1H = 0.008479")
print(f"  Actual:      σ_15m = {sigma_15m:.6f}, σ_1H = {sigma_1h:.6f}")
print(f"  Spec ratio (15m/1H): 0.5148")
print(f"  Actual ratio:        {ratio:.4f}")
print(f"  Theoretical (1/√4):  {theoretical_ratio:.4f}")
print(f"  Scaling factor (σ_1H / σ_15m): {sigma_1h/sigma_15m:.4f} (should be ~2.0 = √4)")

# =============================================================================
# SECTION 2: Reward Scale Clip Saturation
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: REWARD SCALE CLIP SATURATION")
print("=" * 70)

print("\n  Testing various scales for 15m bars:")
for scale in [100, 125, 150, 175, 200]:
    scaled = log_ret_15m * scale
    pct_in = (np.abs(scaled) <= 1.0).mean() * 100
    print(f"    scale={scale:>3d}: {pct_in:.1f}% in [-1, 1]")

print("\n  Testing various scales for 1H bars:")
for scale in [50, 60, 70, 75, 80, 90, 100, 118, 125, 150]:
    scaled = log_ret_1h * scale
    pct_in = (np.abs(scaled) <= 1.0).mean() * 100
    print(f"    scale={scale:>3d}: {pct_in:.1f}% in [-1, 1]")

# What scale gives exactly the same clip rate as 150 on 15m?
target_pct = (np.abs(log_ret_15m * 150) <= 1.0).mean()
print(f"\n  Target clip rate (150 on 15m): {target_pct*100:.2f}%")

# Binary search for matching scale on 1H
lo, hi = 1.0, 500.0
for _ in range(100):
    mid = (lo + hi) / 2
    pct = (np.abs(log_ret_1h * mid) <= 1.0).mean()
    if pct > target_pct:
        lo = mid
    else:
        hi = mid
optimal_scale_1h = (lo + hi) / 2
pct_at_optimal = (np.abs(log_ret_1h * optimal_scale_1h) <= 1.0).mean()
print(f"  Optimal 1H scale for same clip rate: {optimal_scale_1h:.1f} ({pct_at_optimal*100:.2f}%)")

# Also check: is simple σ ratio the right way to scale?
ratio_scale = 150.0 * (sigma_15m / sigma_1h)
pct_at_ratio = (np.abs(log_ret_1h * ratio_scale) <= 1.0).mean()
print(f"  σ-ratio based scale (150 × σ_15m/σ_1H): {ratio_scale:.1f} ({pct_at_ratio*100:.2f}%)")

# =============================================================================
# SECTION 3: Return Distribution Shape (Kurtosis, Skew)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: DISTRIBUTION SHAPE (fat tails matter for clip rate)")
print("=" * 70)

for label, lr in [("15m", log_ret_15m), ("1H", log_ret_1h)]:
    print(f"  {label}: skew={lr.skew():.3f}, excess_kurtosis={lr.kurtosis():.2f}")

# Check percentiles - are tails fatter at 1H?
print("\n  Percentile comparison (absolute log-returns):")
for p in [90, 95, 99, 99.5, 99.9]:
    v15 = np.percentile(np.abs(log_ret_15m), p)
    v1h = np.percentile(np.abs(log_ret_1h), p)
    print(f"    P{p:>5.1f}: 15m={v15:.6f}, 1H={v1h:.6f}, ratio={v1h/v15:.3f} (√4={sqrt4:.3f})")

# =============================================================================
# SECTION 4: Autocorrelation Analysis (Holding Period Justification)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: AUTOCORRELATION (holding period justification)")
print("=" * 70)

print("\n  1H return autocorrelation (lags 1-120 hours):")
acf_values = []
for lag in [1, 2, 3, 4, 6, 12, 24, 48, 72, 96, 120]:
    ac = log_ret_1h.autocorr(lag=lag)
    acf_values.append((lag, ac))
    hours_label = f"{lag}h"
    days_label = f"({lag/24:.1f}d)" if lag >= 24 else ""
    print(f"    lag={lag:>3d} {hours_label:>5s} {days_label:>6s}: autocorr = {ac:+.4f}")

# Absolute return autocorrelation (volatility clustering)
print("\n  1H |return| autocorrelation (volatility clustering):")
abs_ret_1h = np.abs(log_ret_1h)
for lag in [1, 2, 4, 12, 24, 48, 72, 120, 168, 336]:
    ac = abs_ret_1h.autocorr(lag=lag)
    hours_label = f"{lag}h"
    days_label = f"({lag/24:.1f}d)" if lag >= 24 else ""
    print(f"    lag={lag:>3d} {hours_label:>5s} {days_label:>6s}: |ret| autocorr = {ac:+.4f}")

# =============================================================================
# SECTION 5: Optimal Momentum Window
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: MOMENTUM / MEAN-REVERSION WINDOWS")
print("=" * 70)

print("\n  Signed return predictability (ret[t] × ret[t+horizon]):")
for horizon in [1, 2, 4, 6, 12, 24, 48, 72, 120, 168]:
    future_ret = log_ret_1h.shift(-horizon).dropna()
    current_ret = log_ret_1h.iloc[:len(future_ret)]
    corr = np.corrcoef(current_ret.values, future_ret.values)[0, 1]
    hours_label = f"{horizon}h"
    days_label = f"({horizon/24:.1f}d)" if horizon >= 24 else ""
    print(f"    horizon={horizon:>3d} {hours_label:>5s} {days_label:>6s}: corr = {corr:+.6f}")

# =============================================================================
# SECTION 6: Verify Training Horizon Step Count
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: TRAINING HORIZON STEP COUNT")
print("=" * 70)

bars_per_day_15m = 24 * 4  # 96
bars_per_day_1h = 24

steps_15m_35d = 35 * bars_per_day_15m
steps_1h_140d = 140 * bars_per_day_1h
steps_1h_35d = 35 * bars_per_day_1h

print(f"  15m × 35 days: {steps_15m_35d} steps")
print(f"  1H × 140 days: {steps_1h_140d} steps")
print(f"  1H × 35 days:  {steps_1h_35d} steps")
print(f"  Match? {steps_15m_35d == steps_1h_140d}")

# But does 140 days actually contain ~3360 trading bars? (crypto trades 24/7)
# Verify from data
sample_start = datetime(2020, 1, 1)
sample_end = datetime(2020, 5, 20)  # ~140 days
df_sample = df_1m[(df_1m["datetime"] >= sample_start) & (df_1m["datetime"] < sample_end)]
df_sample_1h = resample_ohlcv(df_sample, "1h")
print(f"\n  Actual 1H bars in 140-day sample (2020-01-01 to 2020-05-20): {len(df_sample_1h)}")
print(f"  Expected (140 × 24): {140 * 24}")

# Similarly check eval
eval_15m_70d = 70 * bars_per_day_15m
eval_1h_280d = 280 * bars_per_day_1h
print(f"\n  Eval: 15m × 70d = {eval_15m_70d}, 1H × 280d = {eval_1h_280d}")

# =============================================================================
# SECTION 7: Gamma / Effective Horizon
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: GAMMA / EFFECTIVE HORIZON")
print("=" * 70)

gamma_old = 0.97
H_old_steps = 1 / (1 - gamma_old)
H_old_hours = H_old_steps * 0.25  # 15m bars

gamma_new = 0.986
H_new_steps = 1 / (1 - gamma_new)
H_new_hours = H_new_steps * 1.0  # 1H bars

print(f"  Old: γ=0.97 → H={H_old_steps:.1f} steps × 0.25h = {H_old_hours:.1f} hours ({H_old_hours/24:.1f} days)")
print(f"  New: γ=0.986 → H={H_new_steps:.1f} steps × 1h = {H_new_hours:.1f} hours ({H_new_hours/24:.1f} days)")

# What gamma gives various horizons?
print("\n  Gamma → Horizon mapping (1H bars):")
for target_hours in [24, 48, 72, 96, 120, 168]:
    target_steps = target_hours  # 1H bars
    g = 1.0 - 1.0 / target_steps
    print(f"    {target_hours:>3d}h ({target_hours/24:.1f}d): γ = {g:.4f}")

# What was the OLD effective horizon in real hours, and should we match it?
print(f"\n  Old effective horizon in real time: {H_old_hours:.1f} hours = {H_old_hours/24:.1f} days")
# To match the same real-time horizon (8.3h) on 1H:
gamma_same_time = 1.0 - 1.0/H_old_hours
print(f"  γ to match same real-time horizon on 1H: {gamma_same_time:.4f} (H={H_old_hours:.1f} steps)")

# =============================================================================
# SECTION 8: Bridge Band Period Analysis
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8: BRIDGE BAND LOOKBACK ANALYSIS")
print("=" * 70)

# Check: what is the typical range/BB width at different lookbacks?
# And does the proposed scheme capture different regimes?
print("\n  Rolling std of close at various windows (1H bars):")
close_1h = df_1h["close"]
for window in [6, 12, 24, 48, 72, 96, 168, 336, 480]:
    rolling_std = close_1h.rolling(window).std().dropna()
    rel_std = (rolling_std / close_1h.iloc[window:]).dropna()
    hours_label = f"{window}h"
    days_label = f"({window/24:.1f}d)"
    print(f"    window={window:>3d} {hours_label:>5s} {days_label:>7s}: "
          f"median rel_std = {rel_std.median():.6f}, mean = {rel_std.mean():.6f}")

# Correlation between different BB windows (do they capture different info?)
print("\n  Correlation between rolling std windows (should be low for good feature diversity):")
windows_to_check = [12, 72, 336]
stds = {}
for w in windows_to_check:
    s = close_1h.rolling(w).std().dropna()
    # Align all to same index
    stds[w] = s

# Align
common_idx = stds[336].index
for w in windows_to_check:
    stds[w] = stds[w].reindex(common_idx).dropna()
common_idx = stds[336].dropna().index
for w in windows_to_check:
    stds[w] = stds[w].reindex(common_idx)

print(f"    corr(12h, 72h):  {stds[12].corr(stds[72]):.3f}")
print(f"    corr(12h, 336h): {stds[12].corr(stds[336]):.3f}")
print(f"    corr(72h, 336h): {stds[72].corr(stds[336]):.3f}")

# Compare with old 15m regime equivalents mapped to 1H
# Old: 14 bars (3.5h) → 3.5h on 1H ≈ 3-4 bars (too short for 1H)
# Old: 96 bars (24h/1d) → 24h on 1H = 24 bars
# Old: 480 bars (5d) → 5d on 1H = 120 bars
print("\n  Alternative: direct time-equivalent mapping from old 15m:")
print(f"    Old micro=14×15m=3.5h → 1H equivalent: 4 bars (too noisy)")
print(f"    Old daily=96×15m=24h  → 1H equivalent: 24 bars")
print(f"    Old weekly=480×15m=5d → 1H equivalent: 120 bars")

alt_windows = [4, 24, 120]
stds_alt = {}
for w in alt_windows:
    stds_alt[w] = close_1h.rolling(w).std().reindex(common_idx)
print(f"    corr(4h, 24h):   {stds_alt[4].corr(stds_alt[24]):.3f}")
print(f"    corr(4h, 120h):  {stds_alt[4].corr(stds_alt[120]):.3f}")
print(f"    corr(24h, 120h): {stds_alt[24].corr(stds_alt[120]):.3f}")

# =============================================================================
# SECTION 9: History Window Size
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 9: HISTORY WINDOW")
print("=" * 70)

max_lookback_proposed = 336  # 14 days for weekly BB
# Need some extra for warmup of the indicators
# BB uses range, Bollinger, and Hurst over the window
# days_ago=30 → 30 * 24 = 720 bars needed for the state provider
# Plus the BB window itself: 336

print(f"  Max BB window: 336 bars (14 days)")
print(f"  days_ago=30 → {30*24} bars needed in state provider")
print(f"  Proposed history_window=1000 → {1000/24:.1f} days")
print(f"  Is 1000 sufficient for 30 days + warmup? {1000 >= 30*24}")
print(f"  Safety margin: {1000 - 30*24} bars ({(1000 - 30*24)/24:.1f} days)")

# MACD largest: 240-bar slow line needs 240 bars warmup
# Combined with days_ago=30: need 720 + warmup
print(f"\n  MACD max slow period: 240 bars → needs ~240 bars warmup")
print(f"  Total needed: max(720+240, 720+336) = {max(720+240, 720+336)} bars")
print(f"  1000 bars sufficient? {1000 >= max(720+240, 720+336)}")

# =============================================================================
# SECTION 10: Cooldown and n_step_update
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 10: COOLDOWN AND N_STEP_UPDATE")
print("=" * 70)

print(f"  Old cooldown: 8 bars × 15m = {8*15/60:.0f} hours")
print(f"  Proposed: 2 bars × 1H = 2 hours (same real time)")
print(f"  Proposed: 4 bars × 1H = 4 hours (2× real time)")
print(f"")
print(f"  Old n_step_update: 16 bars × 15m = {16*15/60:.0f} hours between learns")
print(f"  Proposed: 4 bars × 1H = 4 hours (same real time)")
print(f"")
print(f"  But consider: with 4× fewer steps, buffer fills 4× slower.")
print(f"  Old: n_steps_warmup=10000 at 96 steps/day → {10000/96:.0f} days warmup")
print(f"  New: n_steps_warmup=10000 at 24 steps/day → {10000/24:.0f} days warmup")
print(f"  If reduce warmup proportionally: 2500 steps → {2500/24:.0f} days")
print(f"  With train_sample_duration_days=140, warmup 2500 steps uses {2500/24/140*100:.0f}% of each iteration")

# =============================================================================
# SECTION 11: Hold Cost Threshold
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 11: HOLD COST THRESHOLD")
print("=" * 70)

print(f"  Current _HOLD_THRESHOLD_HOURS = 72h (3 days)")
print(f"  Target hold: 1-5 days (spec says target ~3 days)")
print(f"  If agent targets 3-day holds, 72h threshold means cost kicks in exactly at target.")
print(f"  This might be too aggressive — agent gets penalized right at the target hold duration.")
print(f"")
# Check: what's the mean absolute 1H return at various horizons?
print("  Expected absolute return at various hold durations (1H data):")
for hours in [12, 24, 48, 72, 120, 168, 336]:
    future_rets = np.log(close_1h.shift(-hours) / close_1h).dropna()
    mean_abs = np.abs(future_rets).mean()
    median_abs = np.abs(future_rets).median()
    print(f"    {hours:>3d}h ({hours/24:.1f}d): mean |ret| = {mean_abs:.4f} ({mean_abs*100:.2f}%), "
          f"median = {median_abs:.4f} ({median_abs*100:.2f}%)")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
