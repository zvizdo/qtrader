"""
EDA: Evaluate DQTPAgent feature parameters for 1H BTCUSD, 1-5 day trade duration.

For each feature group, tests multiple parameter variants and measures predictive
power via Spearman correlation and directional accuracy against forward returns
at horizons: 24h (1d), 48h (2d), 72h (3d), 96h (4d), 120h (5d).
"""

import pandas as pd
import numpy as np
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def spearmanr(a, b):
    """Manual Spearman correlation using pandas rank (no scipy needed)."""
    a_rank = pd.Series(a).rank()
    b_rank = pd.Series(b).rank()
    corr = a_rank.corr(b_rank)
    n = len(a_rank.dropna())
    # Approximate p-value (t-test on correlation)
    if abs(corr) >= 1.0 or n < 3:
        return corr, 0.0
    t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
    # Very rough p-value approximation (good enough for ranking)
    pval = 2.0 * np.exp(-0.717 * abs(t_stat) - 0.416 * t_stat**2 / n)
    return corr, min(pval, 1.0)

# ── Data Loading ──

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
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def resample_1h(df_min: pd.DataFrame) -> pd.DataFrame:
    df = df_min.set_index("datetime").resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    df = df.reset_index()
    return df

# ── Forward Returns ──

HORIZONS = {"1d": 24, "2d": 48, "3d": 72, "4d": 96, "5d": 120}

def add_forward_returns(df):
    for name, bars in HORIZONS.items():
        df[f"fwd_{name}"] = np.log(df["close"].shift(-bars) / df["close"])
        df[f"fwd_{name}_dir"] = np.sign(df[f"fwd_{name}"])
    return df

# ── Metrics ──

def predictive_power(feature_series, df, label=""):
    """Compute Spearman corr and directional accuracy for each horizon."""
    results = {}
    valid = feature_series.notna() & np.isfinite(feature_series)
    for name, bars in HORIZONS.items():
        fwd = df[f"fwd_{name}"]
        mask = valid & fwd.notna()
        if mask.sum() < 100:
            continue
        corr, pval = spearmanr(feature_series[mask], fwd[mask])
        # Directional accuracy: does sign(feature) predict sign(fwd)?
        fwd_dir = df[f"fwd_{name}_dir"][mask]
        feat_dir = np.sign(feature_series[mask])
        dir_acc = (feat_dir == fwd_dir).mean()
        results[name] = {"corr": round(corr, 4), "pval": round(pval, 6),
                         "dir_acc": round(dir_acc, 4), "abs_corr": round(abs(corr), 4)}
    return results

def summarize_results(results_dict, label=""):
    """Print a summary table of results across parameter variants."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    rows = []
    for variant, horizons in results_dict.items():
        row = {"variant": variant}
        abs_corrs = []
        for h_name, metrics in horizons.items():
            row[f"{h_name}_corr"] = metrics["corr"]
            row[f"{h_name}_dir"] = metrics["dir_acc"]
            abs_corrs.append(metrics["abs_corr"])
        # Weight: 3d most important (center of 1-5d range), then 2d/4d, then 1d/5d
        weights = {"1d": 0.10, "2d": 0.20, "3d": 0.40, "4d": 0.20, "5d": 0.10}
        weighted_corr = sum(
            horizons.get(h, {}).get("abs_corr", 0) * w for h, w in weights.items()
        )
        row["weighted_abs_corr"] = round(weighted_corr, 4)
        row["mean_abs_corr"] = round(np.mean(abs_corrs), 4) if abs_corrs else 0
        rows.append(row)

    rdf = pd.DataFrame(rows).sort_values("weighted_abs_corr", ascending=False)
    print(rdf.to_string(index=False))
    return rdf

# ── Bridge Bands ──

def calculate_bridge_bands(df, bridge_range_length=14, bollinger_bands_length=14,
                           bollinger_bands_num_std=2, hurst_exp_length=14):
    """Compute Bridge Bands (from indicators.py)."""
    df_bb = df.copy()
    L = bridge_range_length
    close_vals = df_bb['close'].values

    if len(close_vals) < L + 1:
        for col in ["bridge_bands_pos", "bridge_bands_width", "hurst_exp"]:
            df_bb[col] = np.nan
        return df_bb

    windows = np.lib.stride_tricks.sliding_window_view(close_vals, window_shape=L + 1)
    slopes = (windows[:, -1] - windows[:, 0]) / L
    intercepts = windows[:, 0]
    trends = slopes[:, None] * np.arange(L + 1) + intercepts[:, None]
    diffs = windows - trends
    max_min_sum = np.abs(diffs.max(axis=1)) + np.abs(diffs.min(axis=1))

    br_upper = np.full(len(close_vals), np.nan)
    br_lower = np.full(len(close_vals), np.nan)
    br_upper[L:] = close_vals[L:] + max_min_sum
    br_lower[L:] = close_vals[L:] - max_min_sum

    df_bb["br_upper"] = br_upper
    df_bb["br_lower"] = br_lower

    tp = (df_bb["high"] + df_bb["low"] + df_bb["close"]) / 3
    df_bb["ema"] = tp.ewm(span=bollinger_bands_length, adjust=False).mean()
    df_bb["tp_std"] = tp.rolling(bollinger_bands_length).std()
    df_bb["bb_upper"] = df_bb["ema"] + bollinger_bands_num_std * df_bb["tp_std"]
    df_bb["bb_lower"] = df_bb["ema"] - bollinger_bands_num_std * df_bb["tp_std"]

    # Hurst exponent
    df_bb["hh"] = df_bb["high"].rolling(hurst_exp_length).max()
    df_bb["ll"] = df_bb["low"].rolling(hurst_exp_length).min()
    df_bb["close_prev"] = df_bb["close"].shift(1)
    df_bb["tr"] = np.vstack([
        abs(df_bb["high"] - df_bb["low"]).values,
        abs(df_bb["high"] - df_bb["close_prev"]).values,
        abs(df_bb["low"] - df_bb["close_prev"]).values,
    ]).max(axis=0)
    df_bb["atr"] = df_bb["tr"].rolling(hurst_exp_length).mean()
    df_bb["hurst_exp"] = (
        np.log(df_bb["hh"] - df_bb["ll"] + 1e-8) - np.log(df_bb["atr"] + 1e-8)
    ) / np.log(hurst_exp_length)

    df_bb["bridge_bands_lower"] = df_bb["bb_lower"] + (
        (df_bb["br_lower"] - df_bb["bb_lower"]) * abs(df_bb["hurst_exp"] * 2 - 1)
    )
    df_bb["bridge_bands_upper"] = df_bb["bb_upper"] - (
        (df_bb["bb_upper"] - df_bb["br_upper"]) * abs(df_bb["hurst_exp"] * 2 - 1)
    )
    band_width = df_bb["bridge_bands_upper"] - df_bb["bridge_bands_lower"]
    band_width_safe = np.where(band_width == 0, 1e-8, band_width)
    df_bb["bridge_bands_pos"] = 2 * ((df_bb["close"] - df_bb["bridge_bands_lower"]) / band_width_safe) - 1
    bw_std = band_width.std()
    df_bb["bridge_bands_width"] = (band_width - band_width.mean()) / (bw_std if bw_std > 1e-8 else 1e-8)

    return df_bb

# ── MACD ──

def calculate_macd(df, ema_short_length=12, ema_long_length=26, signal_length=9):
    df_m = df.copy()
    tp = (df_m["high"] + df_m["low"] + df_m["close"]) / 3
    df_m["ema_short"] = tp.ewm(span=ema_short_length, min_periods=ema_short_length).mean()
    df_m["ema_long"] = tp.ewm(span=ema_long_length, min_periods=ema_long_length).mean()
    df_m["macd"] = 100 * (1 - df_m["ema_long"] / df_m["ema_short"])
    df_m["macd"] = np.sign(df_m["macd"]) * np.log1p(np.abs(df_m["macd"]) + 1e-8)
    df_m["macd_signal"] = df_m["macd"].ewm(span=signal_length, adjust=False).mean()
    df_m["macd_hist"] = df_m["macd"] - df_m["macd_signal"]
    return df_m

# ── Trend Maturity ──

def detect_swing_points(highs, lows, order):
    n = len(highs)
    sh_idx, sh_val, sl_idx, sl_val = [], [], [], []
    for i in range(order, n - order):
        window_h = highs[i - order: i + order + 1]
        if highs[i] == window_h.max() and np.sum(window_h == highs[i]) == 1:
            sh_idx.append(i)
            sh_val.append(highs[i])
        window_l = lows[i - order: i + order + 1]
        if lows[i] == window_l.min() and np.sum(window_l == lows[i]) == 1:
            sl_idx.append(i)
            sl_val.append(lows[i])
    return (np.array(sh_idx, dtype=int), np.array(sh_val, dtype=np.float64),
            np.array(sl_idx, dtype=int), np.array(sl_val, dtype=np.float64))

def merge_swing_points(sh_idx, sh_val, sl_idx, sl_val):
    points = []
    for i, v in zip(sh_idx, sh_val):
        points.append((int(i), float(v), "H"))
    for i, v in zip(sl_idx, sl_val):
        points.append((int(i), float(v), "L"))
    points.sort(key=lambda x: x[0])
    if not points:
        return []
    merged = [points[0]]
    for pt in points[1:]:
        if pt[2] == merged[-1][2]:
            if pt[2] == "H" and pt[1] > merged[-1][1]:
                merged[-1] = pt
            elif pt[2] == "L" and pt[1] < merged[-1][1]:
                merged[-1] = pt
        else:
            merged.append(pt)
    return merged

def compute_tm_features(swings):
    zeros = np.zeros(9, dtype=np.float64)
    if len(swings) < 3:
        return zeros
    legs = []
    for i in range(len(swings) - 1):
        s, e = swings[i], swings[i + 1]
        direction = "bull" if e[1] > s[1] else "bear"
        legs.append({"start_price": s[1], "end_price": e[1],
                      "direction": direction, "magnitude": abs(e[1] - s[1])})
    if len(legs) < 2:
        return zeros

    bull_highs = [l["end_price"] for l in legs if l["direction"] == "bull"]
    bear_lows = [l["end_price"] for l in legs if l["direction"] == "bear"]

    def _count_consecutive(vals, cmp):
        count = 0
        if len(vals) >= 2:
            for i in range(len(vals) - 1, 0, -1):
                if cmp(vals[i], vals[i - 1]):
                    count += 1
                else:
                    break
        return count

    hh = _count_consecutive(bull_highs, lambda a, b: a > b)
    hl = _count_consecutive(bear_lows, lambda a, b: a > b)
    ll = _count_consecutive(bear_lows, lambda a, b: a < b)
    lh = _count_consecutive(bull_highs, lambda a, b: a < b)

    bull_wc = min(hh, hl) + 1 if (hh > 0 and hl > 0) else 0
    bear_wc = min(ll, lh) + 1 if (ll > 0 and lh > 0) else 0

    recent_n = min(len(legs), 10)
    recent = legs[-recent_n:]
    bull_mag = sum(l["magnitude"] for l in recent if l["direction"] == "bull")
    bear_mag = sum(l["magnitude"] for l in recent if l["direction"] == "bear")
    total_mag = bull_mag + bear_mag
    dir_bias = (bull_mag - bear_mag) / total_mag if total_mag > 0 else 0.0

    first_price = legs[-recent_n]["start_price"]
    last_price = legs[-1]["end_price"]
    net_disp = abs(last_price - first_price)
    total_travel = sum(l["magnitude"] for l in recent)
    eff_ratio = net_disp / total_travel if total_travel > 0 else 0.0

    if bull_wc > bear_wc:
        direction, wave_count = "bull", bull_wc
    elif bear_wc > bull_wc:
        direction, wave_count = "bear", bear_wc
    elif dir_bias > 0.2:
        direction, wave_count = "bull", max(hh, hl)
    elif dir_bias < -0.2:
        direction, wave_count = "bear", max(ll, lh)
    else:
        direction, wave_count = "neutral", 0

    exhaustion = np.clip((wave_count - 1) / 4.0, 0.0, 1.0) if wave_count > 0 else 0.0
    if abs(dir_bias) > 0.3 and eff_ratio < 0.3:
        exhaustion = max(exhaustion, 0.5)

    is_ext = 0.0
    if direction != "neutral":
        rel = [l for l in legs if l["direction"] == direction]
        if len(rel) >= 2:
            is_ext = 1.0 if rel[-1]["magnitude"] > rel[-2]["magnitude"] else 0.0

    retrace = 0.0
    if len(legs) >= 2 and legs[-2]["magnitude"] > 0:
        retrace = np.clip(legs[-1]["magnitude"] / legs[-2]["magnitude"], 0.0, 2.0)

    imp_ratio = 0.0
    if direction != "neutral":
        same_dir = [l["magnitude"] for l in legs if l["direction"] == direction]
        if len(same_dir) >= 2 and same_dir[-2] > 0:
            imp_ratio = np.clip(same_dir[-1] / same_dir[-2], 0.0, 4.0)

    dir_sign = {"bull": 1.0, "bear": -1.0, "neutral": 0.0}[direction]
    return np.array([dir_sign, exhaustion, np.log1p(bull_wc), np.log1p(bear_wc),
                     is_ext, retrace, imp_ratio, dir_bias, eff_ratio], dtype=np.float64)

def calculate_trend_maturity(df, swing_order=5, lookback=168):
    highs = df["high"].values
    lows = df["low"].values
    n = len(highs)
    feat_names = ["tm_direction", "tm_exhaustion", "tm_bull_waves", "tm_bear_waves",
                  "tm_extending", "tm_retrace_depth", "tm_impulse_ratio",
                  "tm_dir_bias", "tm_efficiency"]
    result = {name: np.full(n, np.nan) for name in feat_names}

    if n < 2 * swing_order + 1:
        for name in feat_names:
            df[name] = result[name]
        return df

    sh_idx, sh_val, sl_idx, sl_val = detect_swing_points(highs, lows, swing_order)
    all_swings = merge_swing_points(sh_idx, sh_val, sl_idx, sl_val)

    for t in range(lookback, n):
        window_start = max(0, t - lookback)
        cutoff = t - swing_order
        bar_swings = [s for s in all_swings if window_start <= s[0] <= cutoff]
        feats = compute_tm_features(bar_swings)
        for j, name in enumerate(feat_names):
            result[name][t] = feats[j]

    for name in feat_names:
        df[name] = result[name]
    return df


# ════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading BTCUSD data...")
    # Use 3 years of data for robust statistics: 2020-2023
    df_min = load_range(datetime(2020, 1, 1), datetime(2023, 1, 1))
    print(f"  Loaded {len(df_min):,} minute bars")

    df = resample_1h(df_min)
    print(f"  Resampled to {len(df):,} 1H bars")
    df = add_forward_returns(df)

    # ──────────────────────────────────────────────────────────────────
    # 1. LOG-RETURNS LOOKBACKS
    # ──────────────────────────────────────────────────────────────────
    print("\n\n" + "#"*80)
    print("# 1. LOG-RETURNS: Testing lookback periods")
    print("#"*80)

    lookback_sets = {
        "current [1,24,72]": [1, 24, 72],
        "[1,12,48]": [1, 12, 48],
        "[1,12,72]": [1, 12, 72],
        "[1,24,48]": [1, 24, 48],
        "[1,24,120]": [1, 24, 120],
        "[2,12,72]": [2, 12, 72],
        "[1,6,24,72]": [1, 6, 24, 72],
        "[1,12,24,72]": [1, 12, 24, 72],
        "[1,24,72,168]": [1, 24, 72, 168],
        "[4,24,72]": [4, 24, 72],
        "[6,24,72]": [6, 24, 72],
        "[1,48,120]": [1, 48, 120],
    }

    results_lr = {}
    for variant_name, lbs in lookback_sets.items():
        per_feature = {}
        for lb in lbs:
            if lb >= len(df):
                continue
            feat = np.log(df["close"] / df["close"].shift(lb)) * 100
            pp = predictive_power(feat, df)
            for h, m in pp.items():
                per_feature.setdefault(h, []).append(m["abs_corr"])

        # Average across all lookbacks in this set
        combined = {}
        for h in HORIZONS:
            corrs = per_feature.get(h, [])
            if corrs:
                combined[h] = {"corr": round(np.mean([predictive_power(
                    np.log(df["close"] / df["close"].shift(lb)) * 100, df
                ).get(h, {}).get("corr", 0) for lb in lbs]), 4),
                    "abs_corr": round(np.mean(corrs), 4),
                    "dir_acc": round(np.mean([predictive_power(
                        np.log(df["close"] / df["close"].shift(lb)) * 100, df
                    ).get(h, {}).get("dir_acc", 0.5) for lb in lbs]), 4),
                    "pval": 0}
        results_lr[variant_name] = combined

    summarize_results(results_lr, "LOG-RETURNS: Mean |Spearman corr| across lookbacks in set")

    # Also show individual lookback performance
    print("\n  Individual lookback correlations:")
    individual_lbs = [1, 2, 4, 6, 12, 24, 48, 72, 120, 168]
    ind_results = {}
    for lb in individual_lbs:
        feat = np.log(df["close"] / df["close"].shift(lb)) * 100
        ind_results[f"lb={lb}h"] = predictive_power(feat, df)
    summarize_results(ind_results, "LOG-RETURNS: Individual lookback performance")

    # ──────────────────────────────────────────────────────────────────
    # 2. RELATIVE VOLUME LOOKBACKS
    # ──────────────────────────────────────────────────────────────────
    print("\n\n" + "#"*80)
    print("# 2. RELATIVE VOLUME: Testing lookback periods")
    print("#"*80)

    vol_results = {}
    vol_mean_windows = [24, 48, 72, 120, 168]
    for mean_w in vol_mean_windows:
        vol_mean = df["volume"].rolling(mean_w).mean() + 1e-8
        for lb in [1, 6, 12, 24, 48, 72]:
            feat = np.log1p(df["volume"].shift(lb - 1) / vol_mean)
            key = f"mean={mean_w}h, lb={lb}h"
            vol_results[key] = predictive_power(feat, df)

    summarize_results(vol_results, "RELATIVE VOLUME: lookback x mean_window")

    # ──────────────────────────────────────────────────────────────────
    # 3. BRIDGE BANDS
    # ──────────────────────────────────────────────────────────────────
    print("\n\n" + "#"*80)
    print("# 3. BRIDGE BANDS: Testing scale parameters")
    print("#"*80)

    bb_param_sets = {
        # Micro candidates
        "micro_6":   {"bridge_range_length": 6,   "bollinger_bands_length": 6,   "hurst_exp_length": 6},
        "micro_8":   {"bridge_range_length": 8,   "bollinger_bands_length": 8,   "hurst_exp_length": 8},
        "micro_12 (current)": {"bridge_range_length": 12,  "bollinger_bands_length": 12,  "hurst_exp_length": 12},
        "micro_16":  {"bridge_range_length": 16,  "bollinger_bands_length": 16,  "hurst_exp_length": 16},
        "micro_24":  {"bridge_range_length": 24,  "bollinger_bands_length": 24,  "hurst_exp_length": 24},
        # Daily candidates
        "daily_36":  {"bridge_range_length": 36,  "bollinger_bands_length": 36,  "hurst_exp_length": 36},
        "daily_48":  {"bridge_range_length": 48,  "bollinger_bands_length": 48,  "hurst_exp_length": 48},
        "daily_72 (current)": {"bridge_range_length": 72,  "bollinger_bands_length": 72,  "hurst_exp_length": 72},
        "daily_96":  {"bridge_range_length": 96,  "bollinger_bands_length": 96,  "hurst_exp_length": 96},
        "daily_120": {"bridge_range_length": 120, "bollinger_bands_length": 120, "hurst_exp_length": 120},
        # Weekly candidates
        "weekly_168":  {"bridge_range_length": 168, "bollinger_bands_length": 168, "hurst_exp_length": 168},
        "weekly_240":  {"bridge_range_length": 240, "bollinger_bands_length": 240, "hurst_exp_length": 240},
        "weekly_336 (current)": {"bridge_range_length": 336, "bollinger_bands_length": 336, "hurst_exp_length": 336},
        "weekly_504":  {"bridge_range_length": 504, "bollinger_bands_length": 504, "hurst_exp_length": 504},
    }

    # Test each BB variant's pos, width, and hurst features
    for feat_name in ["bridge_bands_pos", "bridge_bands_width", "hurst_exp"]:
        bb_results = {}
        for variant, params in bb_param_sets.items():
            print(f"  Computing BB {variant} ...", end="\r")
            df_bb = calculate_bridge_bands(df.copy(), **params)
            bb_results[variant] = predictive_power(df_bb[feat_name], df)
        summarize_results(bb_results, f"BRIDGE BANDS: {feat_name}")

    # BB position velocity (pos[-1] - pos[-4])
    print("\n  BB Position Velocity analysis:")
    bb_vel_results = {}
    for variant, params in bb_param_sets.items():
        print(f"  Computing BB velocity {variant} ...", end="\r")
        df_bb = calculate_bridge_bands(df.copy(), **params)
        vel = df_bb["bridge_bands_pos"] - df_bb["bridge_bands_pos"].shift(3)
        bb_vel_results[variant] = predictive_power(vel, df)
    summarize_results(bb_vel_results, "BRIDGE BANDS: Position Velocity (pos[-1] - pos[-4])")

    # Test different velocity windows
    print("\n  BB Velocity Window analysis (using current micro_12):")
    df_bb_micro = calculate_bridge_bands(df.copy(), bridge_range_length=12,
                                          bollinger_bands_length=12, hurst_exp_length=12)
    vel_window_results = {}
    for window in [2, 3, 4, 6, 8, 12]:
        vel = df_bb_micro["bridge_bands_pos"] - df_bb_micro["bridge_bands_pos"].shift(window)
        vel_window_results[f"vel_window={window}"] = predictive_power(vel, df)
    summarize_results(vel_window_results, "BB VELOCITY: Window size comparison (micro_12)")

    # ──────────────────────────────────────────────────────────────────
    # 4. MACD
    # ──────────────────────────────────────────────────────────────────
    print("\n\n" + "#"*80)
    print("# 4. MACD: Testing parameter sets")
    print("#"*80)

    macd_param_sets = {
        # Micro
        "micro_4_9_3":    {"ema_short_length": 4,  "ema_long_length": 9,   "signal_length": 3},
        "micro_6_13_4 (current)": {"ema_short_length": 6,  "ema_long_length": 13,  "signal_length": 4},
        "micro_8_17_5":   {"ema_short_length": 8,  "ema_long_length": 17,  "signal_length": 5},
        "micro_12_26_9":  {"ema_short_length": 12, "ema_long_length": 26,  "signal_length": 9},
        # Daily
        "daily_12_36_12": {"ema_short_length": 12, "ema_long_length": 36,  "signal_length": 12},
        "daily_18_54_18": {"ema_short_length": 18, "ema_long_length": 54,  "signal_length": 18},
        "daily_24_72_24 (current)": {"ema_short_length": 24, "ema_long_length": 72,  "signal_length": 24},
        "daily_36_96_24": {"ema_short_length": 36, "ema_long_length": 96,  "signal_length": 24},
        # Weekly
        "weekly_72_168_36":  {"ema_short_length": 72,  "ema_long_length": 168, "signal_length": 36},
        "weekly_112_240_48 (current)": {"ema_short_length": 112, "ema_long_length": 240, "signal_length": 48},
        "weekly_168_336_72": {"ema_short_length": 168, "ema_long_length": 336, "signal_length": 72},
        "weekly_96_240_48":  {"ema_short_length": 96,  "ema_long_length": 240, "signal_length": 48},
    }

    for feat_name in ["macd", "macd_hist"]:
        macd_results = {}
        for variant, params in macd_param_sets.items():
            df_m = calculate_macd(df.copy(), **params)
            macd_results[variant] = predictive_power(df_m[feat_name], df)
        summarize_results(macd_results, f"MACD: {feat_name}")

    # ──────────────────────────────────────────────────────────────────
    # 5. TREND MATURITY
    # ──────────────────────────────────────────────────────────────────
    print("\n\n" + "#"*80)
    print("# 5. TREND MATURITY: Testing swing_order and lookback")
    print("#"*80)

    tm_param_sets = {
        "so=3_lb=72":   {"swing_order": 3, "lookback": 72},
        "so=3_lb=120":  {"swing_order": 3, "lookback": 120},
        "so=3_lb=168":  {"swing_order": 3, "lookback": 168},
        "so=5_lb=72":   {"swing_order": 5, "lookback": 72},
        "so=5_lb=120 (current)": {"swing_order": 5, "lookback": 120},
        "so=5_lb=168":  {"swing_order": 5, "lookback": 168},
        "so=5_lb=240":  {"swing_order": 5, "lookback": 240},
        "so=7_lb=120":  {"swing_order": 7, "lookback": 120},
        "so=7_lb=168":  {"swing_order": 7, "lookback": 168},
        "so=7_lb=240":  {"swing_order": 7, "lookback": 240},
        "so=10_lb=168": {"swing_order": 10, "lookback": 168},
        "so=10_lb=240": {"swing_order": 10, "lookback": 240},
        "so=10_lb=336": {"swing_order": 10, "lookback": 336},
    }

    tm_feat_names = ["tm_direction", "tm_exhaustion", "tm_bull_waves", "tm_bear_waves",
                     "tm_extending", "tm_retrace_depth", "tm_impulse_ratio",
                     "tm_dir_bias", "tm_efficiency"]

    # Per-feature analysis for each TM variant
    for feat_name in tm_feat_names:
        tm_results = {}
        for variant, params in tm_param_sets.items():
            print(f"  Computing TM {variant} for {feat_name}...", end="\r")
            df_tm = calculate_trend_maturity(df.copy(), **params)
            tm_results[variant] = predictive_power(df_tm[feat_name], df)
        summarize_results(tm_results, f"TREND MATURITY: {feat_name}")

    # ──────────────────────────────────────────────────────────────────
    # 6. CANDLE FEATURES (body ratio, vol-norm OHLC)
    # ──────────────────────────────────────────────────────────────────
    print("\n\n" + "#"*80)
    print("# 6. CANDLE FEATURES")
    print("#"*80)

    # Body ratio
    full_range = df["high"] - df["low"] + 1e-8
    body_ratio = abs(df["close"] - df["open"]) / full_range
    candle_results = {"body_ratio": predictive_power(body_ratio, df)}

    # Vol-norm OHLC using different BB scales for normalization
    for bb_scale, params in [("micro_12", {"bridge_range_length": 12, "bollinger_bands_length": 12, "hurst_exp_length": 12}),
                              ("daily_72", {"bridge_range_length": 72, "bollinger_bands_length": 72, "hurst_exp_length": 72})]:
        df_bb = calculate_bridge_bands(df.copy(), **params)
        vol_norm = abs(df_bb["bridge_bands_width"]) + 1e-8
        for sub_feat, col in [("norm_open", "open"), ("norm_high", "high"), ("norm_low", "low")]:
            feat = (1.0 - df[col] / df["close"]) / vol_norm
            candle_results[f"{sub_feat}_{bb_scale}"] = predictive_power(feat, df)

    summarize_results(candle_results, "CANDLE FEATURES")

    # ──────────────────────────────────────────────────────────────────
    # 7. TIME FEATURES
    # ──────────────────────────────────────────────────────────────────
    print("\n\n" + "#"*80)
    print("# 7. TIME FEATURES")
    print("#"*80)

    import math
    time_results = {}
    hours = df["datetime"].dt.hour
    weekdays = df["datetime"].dt.weekday

    time_results["cos_weekday_pi3"] = predictive_power(
        pd.Series(np.cos(np.pi * weekdays / 3), index=df.index), df)
    time_results["sin_weekday_pi3"] = predictive_power(
        pd.Series(np.sin(np.pi * weekdays / 3), index=df.index), df)
    time_results["cos_hour_2pi24"] = predictive_power(
        pd.Series(np.cos(2 * np.pi * hours / 24), index=df.index), df)
    time_results["sin_hour_2pi24"] = predictive_power(
        pd.Series(np.sin(2 * np.pi * hours / 24), index=df.index), df)

    # Alternative time encodings
    time_results["cos_weekday_2pi7"] = predictive_power(
        pd.Series(np.cos(2 * np.pi * weekdays / 7), index=df.index), df)
    time_results["sin_weekday_2pi7"] = predictive_power(
        pd.Series(np.sin(2 * np.pi * weekdays / 7), index=df.index), df)

    summarize_results(time_results, "TIME FEATURES")

    # ──────────────────────────────────────────────────────────────────
    # 8. BB LOOKBACK ANALYSIS
    # ──────────────────────────────────────────────────────────────────
    print("\n\n" + "#"*80)
    print("# 8. BB LOOKBACK INDEX: Testing _LOOKBACKS for BB features")
    print("#"*80)

    # For the current micro_12 BB, test which lookback indices matter
    df_bb_m = calculate_bridge_bands(df.copy(), bridge_range_length=12,
                                      bollinger_bands_length=12, hurst_exp_length=12)
    lb_bb_results = {}
    for lb in [1, 2, 4, 6, 12, 24, 48, 72]:
        for feat in ["bridge_bands_pos", "bridge_bands_width", "hurst_exp"]:
            shifted = df_bb_m[feat].shift(lb - 1)  # shift(0)=current, shift(1)=1-bar-ago
            key = f"{feat}_lb{lb}"
            lb_bb_results[key] = predictive_power(shifted, df)
    summarize_results(lb_bb_results, "BB LOOKBACK INDEX: Which lag of BB features is most predictive?")

    # Same for MACD
    df_macd_m = calculate_macd(df.copy(), ema_short_length=6, ema_long_length=13, signal_length=4)
    lb_macd_results = {}
    for lb in [1, 2, 4, 6, 12, 24, 48, 72]:
        for feat in ["macd", "macd_hist"]:
            shifted = df_macd_m[feat].shift(lb - 1)
            key = f"{feat}_lb{lb}"
            lb_macd_results[key] = predictive_power(shifted, df)
    summarize_results(lb_macd_results, "MACD LOOKBACK INDEX: Which lag of MACD features is most predictive?")

    # ──────────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────────
    print("\n\n" + "="*80)
    print("  ANALYSIS COMPLETE")
    print("="*80)
    print(f"  Data: BTCUSD 1H bars, 2020-01-01 to 2023-01-01 ({len(df):,} bars)")
    print(f"  Forward return horizons: {list(HORIZONS.keys())}")
    print(f"  Metric: Spearman rank correlation (robust to outliers)")
    print(f"  Weighted score: 10% 1d, 20% 2d, 40% 3d, 20% 4d, 10% 5d")
