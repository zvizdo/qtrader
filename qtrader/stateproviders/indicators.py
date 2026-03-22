from itertools import combinations

import numpy as np
import pandas as pd
import ta
import math
from datetime import timedelta, datetime
from qtrader.environments.base import BaseMarketEnv
from qtrader.stateproviders import BaseSymbolStateProvider
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from qtrader.rlflow.persistence import BasePersistenceProvider


class TrendlinesSymbolStateProvider(BaseSymbolStateProvider):

    def __init__(
        self, stock_env: BaseMarketEnv, symbol: str, days_ago: int = 365, **kwargs
    ):
        super(TrendlinesSymbolStateProvider, self).__init__(stock_env, symbol, **kwargs)
        self.days_ago = days_ago
        self.best_n = 4

        # self.skip_save = True
        # self.result = None

    @staticmethod
    def get_extrema(df, atr_factor=2, window=14):
        highs = df.high
        lows = df.low
        close = df.close

        min_max = np.zeros(close.size)
        atr = ta.volatility.AverageTrueRange(
            highs, lows, close, window=window, fillna=True
        )
        atr = atr.average_true_range() * atr_factor

        last_min_max_index = 0
        is_next_max = close[0] < close[1]

        for i in range(1, close.size):
            th = atr[i]

            if is_next_max:
                if close[last_min_max_index] < close[i]:
                    last_min_max_index = i

                elif (
                    close[last_min_max_index] - close[i] > th
                    or lows[i] - close[last_min_max_index] > th
                    or lows[i] - highs[last_min_max_index] > th
                ):
                    min_max[last_min_max_index] = 1
                    is_next_max = False
                    last_min_max_index = i

            else:
                if close[last_min_max_index] > close[i]:
                    last_min_max_index = i

                elif (
                    close[i] - close[last_min_max_index] > th
                    or highs[i] - close[last_min_max_index] > th
                    or highs[i] - lows[last_min_max_index] > th
                ):
                    min_max[last_min_max_index] = -1
                    is_next_max = True
                    last_min_max_index = i

        min_max = np.array(min_max)
        return [m for m in np.where(min_max == -1)[0].tolist() if m > window], [
            m for m in np.where(min_max == 1)[0].tolist() if m > window
        ]

    @staticmethod
    def get_lines(close, extrema, n_lines=5):
        lines = []
        extrema_mix = list(combinations(extrema, 3)) + list(combinations(extrema, 4))
        for x in extrema_mix:
            xs = np.array(x)
            ys = close[xs]
            r = np.polyfit(xs, ys, deg=1, full=True)
            residuals = r[1]
            if residuals.size == 0:
                continue
            r = {
                "x": xs,
                "y": ys.values,
                "m": float(r[0][0]),
                "b": float(r[0][1]),
                "err": float(residuals[0]),
                "pp": [int(p) for p in xs],
            }
            if (
                not math.isnan(r["m"])
                and not math.isnan(r["b"])
                and not math.isnan(r["err"])
            ):
                lines.append(r)

        lines = sorted(lines, key=lambda x: x["err"])
        if len(lines) == 0:
            return []
        if len(lines) <= n_lines:
            return lines

        lines = np.array(lines)
        X = pd.DataFrame([[l["m"], l["b"]] for i, l in enumerate(lines)])
        ss = StandardScaler()
        X = ss.fit_transform(X)

        clusters = AgglomerativeClustering(n_clusters=n_lines, linkage="single").fit(X)
        clbls = np.array(clusters.labels_)

        mlines = []
        for c in range(n_lines):
            ls = lines[clbls == c][:2]
            w = np.array([l["err"] for l in ls])
            w = w.sum() / (w.size * w)
            w /= w.sum()
            m = (np.array([l["m"] for l in ls]) * w).sum()
            b = (np.array([l["b"] for l in ls]) * w).sum()
            err = (np.array([l["err"] for l in ls]) * w).sum()
            pp = set()
            y = np.zeros(3)
            for l in ls:
                for p in l["x"]:
                    pp.add(int(p))

                for k in range(1, 4):
                    y[-k] += l["y"][-k]

            pp = sorted(list(pp))
            y /= len(ls)

            if (
                not math.isnan(float(m))
                and not math.isnan(float(b))
                and not math.isnan(float(err))
            ):
                mlines.append(
                    {
                        "m": float(m),
                        "b": float(b),
                        "err": float(err),
                        "pp": pp,
                        "y": y.tolist(),
                    }
                )

        return sorted(mlines, key=lambda x: x["err"])

    def provide(self):
        cdt = self.env.get_current_market_datetime()
        data = self.env.get_ohlcv(
            self.symbol, dt_from=cdt - timedelta(days=self.days_ago), dt_to=cdt
        )
        if len(data) < 5 * 4 * 6:
            return {"trendlines": None}

        df = data.copy().reset_index()

        # calculate
        mins, maxs = TrendlinesSymbolStateProvider.get_extrema(
            df, atr_factor=2, window=14
        )

        if len(mins) == 0 or len(maxs) == 0:
            return {"trendlines": None}

        # compile
        df_mi = data.iloc[mins].copy()
        df_mi.loc[:, "type"] = "MIN"
        df_mi.loc[:, "ind"] = df_mi["datetime"].apply(lambda dt: (cdt - dt).days)
        df_ma = data.iloc[maxs].copy()
        df_ma.loc[:, "type"] = "MAX"
        df_ma.loc[:, "ind"] = df_ma["datetime"].apply(lambda dt: (cdt - dt).days)

        df_pp = pd.concat([df_mi, df_ma], ignore_index=True, axis=0)
        df_pp = df_pp.sort_values("ind", ascending=True)
        df_pp.loc[:, "datetime"] = df_pp["datetime"].apply(lambda x: x.isoformat())

        sup = TrendlinesSymbolStateProvider.get_lines(df.close, mins, n_lines=4)
        res = TrendlinesSymbolStateProvider.get_lines(df.close, maxs, n_lines=4)
        lines = sorted(sup + res, key=lambda x: x["err"])[: self.best_n]
        lines = sorted(lines, key=lambda x: x["pp"][-1], reverse=True)

        for l in lines:
            l["pp"] = (
                data.iloc[l["pp"]]
                .datetime.apply(lambda dt: (cdt.date() - dt.date()).days)
                .tolist()
            )

        data.loc[:, "datetime"] = data["datetime"].apply(lambda x: x.isoformat())
        return {
            "trendlines": {
                "pivot_points": df_pp.to_dict(orient="list"),
                "lines": lines,
                "data": data.to_dict(orient="list"),
            }
        }


class BridgeBandsSymbolStateProvider(BaseSymbolStateProvider):

    def __init__(self, env: BaseMarketEnv, symbol: str, days_ago: int = 365,
                 bridge_range_length: int = 14, bollinger_bands_length: int = 14,
                 bollinger_bands_num_std: int = 2, hurst_exp_length: int = 14,
                 state_key: str = "bridge_bnds", cache_truncate: int = 24, **kwargs):
        super(BridgeBandsSymbolStateProvider, self).__init__(env, symbol, **kwargs)
        self.days_ago = days_ago
        self.bridge_range_length = bridge_range_length
        self.bollinger_bands_length = bollinger_bands_length
        self.bollinger_bands_num_std = bollinger_bands_num_std
        self.hurst_exp_length = hurst_exp_length
        self.state_key = state_key
        self.cache_truncate = cache_truncate

    @staticmethod
    def calculate_bridge_bands(
        df,
        bridge_range_length=14,
        bollinger_bands_length=14,
        bollinger_bands_num_std=2,
        hurst_exp_length=14,
    ):
        df_bbnd = df.copy()

        # detrend / bridge range
        L = bridge_range_length
        df_bbnd.loc[:, "br_close"] = df_bbnd["close"].shift(L)
        df_bbnd.loc[:, "br_slope"] = (df_bbnd["close"] - df_bbnd["br_close"]) / L
        df_bbnd.loc[:, "br_intercept"] = df_bbnd["br_close"]

        close_vals = df_bbnd['close'].values
        
        # Use sliding window for O(1) loop behavior
        windows = np.lib.stride_tricks.sliding_window_view(close_vals, window_shape=L + 1)
        
        # Calculate slopes and intercepts for each window
        # slope = (close[t] - close[t-L]) / L
        slopes = (windows[:, -1] - windows[:, 0]) / L
        intercepts = windows[:, 0]
        
        # trends: shape (N-L, L+1) 
        # slopes: shape (N-L,)
        trends = slopes[:, None] * np.arange(L + 1) + intercepts[:, None]
        diffs = windows - trends
        
        max_min_sum = np.abs(diffs.max(axis=1)) + np.abs(diffs.min(axis=1))
        
        br_upper = np.full(len(close_vals), np.nan)
        br_lower = np.full(len(close_vals), np.nan)
        
        br_upper[L:] = close_vals[L:] + max_min_sum
        br_lower[L:] = close_vals[L:] - max_min_sum

        df_bbnd.loc[:, "br_upper"] = br_upper
        df_bbnd.loc[:, "br_lower"] = br_lower

        # bollinger bands
        df_bbnd.loc[:, "ema"] = (
            ((df_bbnd["high"] + df_bbnd["low"] + df_bbnd["close"]) / 3)
            .ewm(span=bollinger_bands_length, adjust=False)
            .mean()
        )
        df_bbnd.loc[:, "tp_std"] = (
            ((df_bbnd["high"] + df_bbnd["low"] + df_bbnd["close"]) / 3)
            .rolling(bollinger_bands_length)
            .std()
        )
        df_bbnd.loc[:, "bb_upper"] = (
            df_bbnd["ema"] + bollinger_bands_num_std * df_bbnd["tp_std"]
        )
        df_bbnd.loc[:, "bb_lower"] = (
            df_bbnd["ema"] - bollinger_bands_num_std * df_bbnd["tp_std"]
        )

        # hurst exponent
        df_bbnd.loc[:, "hh"] = df_bbnd["high"].rolling(hurst_exp_length).max()
        df_bbnd.loc[:, "ll"] = df_bbnd["low"].rolling(hurst_exp_length).min()
        df_bbnd.loc[:, "close_prev"] = df_bbnd["close"].shift(1)
        df_bbnd.loc[:, "h-l"] = abs(df_bbnd["high"] - df_bbnd["low"])
        df_bbnd.loc[:, "h-cp"] = abs(df_bbnd["high"] - df_bbnd["close_prev"])
        df_bbnd.loc[:, "l-cp"] = abs(df_bbnd["low"] - df_bbnd["close_prev"])
        df_bbnd.loc[:, "tr"] = np.vstack(
            (df_bbnd["h-l"].values, df_bbnd["h-cp"].values, df_bbnd["l-cp"].values)
        ).max(axis=0)
        df_bbnd.loc[:, "atr"] = df_bbnd["tr"].rolling(hurst_exp_length).mean()
        df_bbnd.loc[:, "hurst_exp"] = (
            np.log(df_bbnd["hh"] - df_bbnd["ll"] + 1e-8) - np.log(df_bbnd["atr"] + 1e-8)
        ) / np.log(hurst_exp_length)

        # bridge bands — write results into the copy, not the original df
        df_bbnd.loc[:, "bridge_bands_lower"] = df_bbnd["bb_lower"] + (
            (df_bbnd["br_lower"] - df_bbnd["bb_lower"])
            * abs(df_bbnd["hurst_exp"] * 2 - 1)
        )
        df_bbnd.loc[:, "bridge_bands_upper"] = df_bbnd["bb_upper"] - (
            (df_bbnd["bb_upper"] - df_bbnd["br_upper"])
            * abs(df_bbnd["hurst_exp"] * 2 - 1)
        )

        band_width = df_bbnd["bridge_bands_upper"] - df_bbnd["bridge_bands_lower"]
        band_width_safe = np.where(band_width == 0, 1e-8, band_width)
        df_bbnd.loc[:, "bridge_bands_pos"] = (
            2 * ((df_bbnd["close"] - df_bbnd["bridge_bands_lower"]) / band_width_safe) - 1
        )
        band_width_std = band_width.std()
        df_bbnd.loc[:, "bridge_bands_width"] = (
            (band_width - band_width.mean()) / (band_width_std if band_width_std > 1e-8 else 1e-8)
        )

        # Copy only the result columns back to df
        for col in ["bridge_bands_lower", "bridge_bands_upper", "hurst_exp",
                    "bridge_bands_pos", "bridge_bands_width"]:
            df.loc[:, col] = df_bbnd[col]

        return df

    def provide(self):
        cdt = self.env.get_current_market_datetime()
        data = self.env.get_ohlcv(
            self.symbol, dt_from=cdt - timedelta(days=self.days_ago), dt_to=cdt
        )
        min_rows = max(self.bridge_range_length, self.bollinger_bands_length,
                       self.hurst_exp_length) * 2
        if len(data) < min_rows:
            return {self.state_key: None}

        data["datetime"] = data["datetime"].apply(lambda x: x.isoformat())
        data = BridgeBandsSymbolStateProvider.calculate_bridge_bands(
            data,
            bridge_range_length=self.bridge_range_length,
            bollinger_bands_length=self.bollinger_bands_length,
            bollinger_bands_num_std=self.bollinger_bands_num_std,
            hurst_exp_length=self.hurst_exp_length,
        )

        if self.cache_truncate > 0:
            data = data.tail(self.cache_truncate)

        return {self.state_key: data.to_dict(orient="list")}


class MACDSymbolStateProvider(BaseSymbolStateProvider):

    def __init__(
        self,
        env: BaseMarketEnv,
        symbol: str,
        days_ago: int = 365,
        ema_short_length: int = 12,
        ema_long_length: int = 26,
        signal_length: int = 9,
        cache_truncate: int = 24,
        **kwargs,
    ):
        super(MACDSymbolStateProvider, self).__init__(env, symbol, **kwargs)
        self.days_ago = days_ago
        self.ema_short_length = ema_short_length
        self.ema_long_length = ema_long_length
        self.signal_length = signal_length
        self.cache_truncate = cache_truncate

    @staticmethod
    def calculate_macd(df, ema_short_length=12, ema_long_length=26, signal_length=9):
        df_t = df.copy()
        df_t["ema_short"] = (
            ((df_t["high"] + df_t["low"] + df_t["close"]) / 3)
            .ewm(span=ema_short_length, min_periods=ema_short_length)
            .mean()
        )
        df_t["ema_long"] = (
            ((df_t["high"] + df_t["low"] + df_t["close"]) / 3)
            .ewm(span=ema_long_length, min_periods=ema_long_length)
            .mean()
        )

        df_t["macd"] = 100 * (1 - df_t["ema_long"] / df_t["ema_short"])
        df_t["macd"] = np.sign(df_t["macd"]) * np.log1p(np.abs(df_t["macd"]) + 1e-8)
        df_t["macd_signal"] = df_t["macd"].ewm(span=signal_length, adjust=False).mean()
        df_t["macd_hist"] = df_t["macd"] - df_t["macd_signal"]

        return df_t

    def provide(self):
        cdt = self.env.get_current_market_datetime()
        data = self.env.get_ohlcv(
            self.symbol, dt_from=cdt - timedelta(days=self.days_ago), dt_to=cdt
        )
        if len(data) == 0:
            return {"macd": {}}

        df_macd = MACDSymbolStateProvider.calculate_macd(
            data,
            ema_short_length=self.ema_short_length,
            ema_long_length=self.ema_long_length,
            signal_length=self.signal_length,
        )
        df_macd["datetime"] = df_macd["datetime"].apply(lambda x: x.isoformat())

        if self.cache_truncate > 0:
            df_macd = df_macd.tail(self.cache_truncate)

        return {
            f"macd_{self.ema_short_length}_{self.ema_long_length}_{self.signal_length}": df_macd[
                ["datetime", "macd_hist", "macd", "macd_signal"]
            ].to_dict(
                orient="list"
            )
        }


class TrendMaturitySymbolStateProvider(BaseSymbolStateProvider):
    """Trend maturity features based on Elliott Wave swing structure.

    Detects swing highs/lows, classifies directional legs, and computes
    9 trend maturity features per bar:
        trend_direction, trend_exhaustion, bull_wave_count, bear_wave_count,
        is_extending, retracement_depth, impulse_ratio, directional_bias,
        efficiency_ratio
    """

    def __init__(self, env: BaseMarketEnv, symbol: str, days_ago: int = 14,
                 swing_order: int = 5, lookback: int = 168,
                 state_key: str = "trend_maturity", cache_truncate: int = 73,
                 **kwargs):
        super(TrendMaturitySymbolStateProvider, self).__init__(env, symbol, **kwargs)
        self.days_ago = days_ago
        self.swing_order = swing_order
        self.lookback = lookback
        self.state_key = state_key
        self.cache_truncate = cache_truncate

    # -- Swing detection (inlined from fibonacci.py) --

    @staticmethod
    def _detect_swing_points(highs, lows, order):
        """Detect swing highs/lows using local extrema with `order` bars each side."""
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

    @staticmethod
    def _merge_swing_points(sh_idx, sh_val, sl_idx, sl_val):
        """Merge swing highs/lows into alternating sequence."""
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

    # -- Trend maturity computation --

    @staticmethod
    def _compute_features_for_bar(swings):
        """Compute 9 trend maturity features from a list of swing points.

        Returns array of 9 floats:
            [0] trend_direction: +1 bull, -1 bear, 0 neutral
            [1] trend_exhaustion: 0.0 (fresh) to 1.0 (exhausted)
            [2] bull_wave_count (log1p)
            [3] bear_wave_count (log1p)
            [4] is_extending: 1.0 if latest impulse > previous
            [5] retracement_depth: last correction / last impulse
            [6] impulse_ratio: last impulse / prev impulse (same dir)
            [7] directional_bias: -1 to +1
            [8] efficiency_ratio: 0 to 1
        """
        zeros = np.zeros(9, dtype=np.float64)
        if len(swings) < 3:
            return zeros

        # Classify legs
        legs = []
        for i in range(len(swings) - 1):
            s, e = swings[i], swings[i + 1]
            direction = "bull" if e[1] > s[1] else "bear"
            legs.append({
                "start_price": s[1], "end_price": e[1],
                "direction": direction,
                "magnitude": abs(e[1] - s[1]),
            })

        if len(legs) < 2:
            return zeros

        # -- Strict HH/HL and LH/LL counting --
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

        # -- Directional bias (last 10 legs) --
        recent_n = min(len(legs), 10)
        recent = legs[-recent_n:]
        bull_mag = sum(l["magnitude"] for l in recent if l["direction"] == "bull")
        bear_mag = sum(l["magnitude"] for l in recent if l["direction"] == "bear")
        total_mag = bull_mag + bear_mag
        dir_bias = (bull_mag - bear_mag) / total_mag if total_mag > 0 else 0.0

        # -- Efficiency ratio --
        first_price = legs[-recent_n]["start_price"]
        last_price = legs[-1]["end_price"]
        net_disp = abs(last_price - first_price)
        total_travel = sum(l["magnitude"] for l in recent)
        eff_ratio = net_disp / total_travel if total_travel > 0 else 0.0

        # -- Trend direction --
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

        # -- Exhaustion --
        exhaustion = np.clip((wave_count - 1) / 4.0, 0.0, 1.0) if wave_count > 0 else 0.0
        if abs(dir_bias) > 0.3 and eff_ratio < 0.3:
            exhaustion = max(exhaustion, 0.5)

        # -- Is extending --
        is_ext = 0.0
        if direction != "neutral":
            rel = [l for l in legs if l["direction"] == direction]
            if len(rel) >= 2:
                is_ext = 1.0 if rel[-1]["magnitude"] > rel[-2]["magnitude"] else 0.0

        # -- Retracement depth (last leg vs previous) --
        retrace = 0.0
        if len(legs) >= 2 and legs[-2]["magnitude"] > 0:
            retrace = np.clip(legs[-1]["magnitude"] / legs[-2]["magnitude"], 0.0, 2.0)

        # -- Impulse ratio (same-direction: last vs prev) --
        imp_ratio = 0.0
        if direction != "neutral":
            same_dir = [l["magnitude"] for l in legs if l["direction"] == direction]
            if len(same_dir) >= 2 and same_dir[-2] > 0:
                imp_ratio = np.clip(same_dir[-1] / same_dir[-2], 0.0, 4.0)

        dir_sign = {"bull": 1.0, "bear": -1.0, "neutral": 0.0}[direction]
        return np.array([
            dir_sign, exhaustion,
            np.log1p(bull_wc), np.log1p(bear_wc),
            is_ext, retrace, imp_ratio,
            dir_bias, eff_ratio,
        ], dtype=np.float64)

    @staticmethod
    def calculate_trend_maturity(df, swing_order=5, lookback=168):
        """Compute trend maturity features for each bar in df.

        Detects swings once over the full range, then for each output bar
        filters to swings within its lookback window and computes features.
        """
        highs = df["high"].values
        lows = df["low"].values
        n = len(highs)

        feat_names = [
            "tm_direction", "tm_exhaustion",
            "tm_bull_waves", "tm_bear_waves",
            "tm_extending", "tm_retrace_depth", "tm_impulse_ratio",
            "tm_dir_bias", "tm_efficiency",
        ]
        result = {name: np.full(n, np.nan) for name in feat_names}

        if n < 2 * swing_order + 1:
            for name in feat_names:
                df[name] = result[name]
            return df

        # Detect swings over the full range
        sh_idx, sh_val, sl_idx, sl_val = TrendMaturitySymbolStateProvider._detect_swing_points(
            highs, lows, swing_order
        )
        all_swings = TrendMaturitySymbolStateProvider._merge_swing_points(
            sh_idx, sh_val, sl_idx, sl_val
        )

        # For each bar, filter swings up to that bar and within lookback
        min_start = max(0, n - lookback)  # earliest bar we compute for
        for t in range(min_start, n):
            window_start = max(0, t - lookback)
            # Filter swings: index <= t (causal) and index >= window_start
            # Swing points need `swing_order` future bars to confirm, so
            # only swings with index <= t - swing_order are confirmed
            cutoff = t - swing_order
            bar_swings = [s for s in all_swings if window_start <= s[0] <= cutoff]

            feats = TrendMaturitySymbolStateProvider._compute_features_for_bar(bar_swings)
            for j, name in enumerate(feat_names):
                result[name][t] = feats[j]

        for name in feat_names:
            df[name] = result[name]
        return df

    def provide(self):
        cdt = self.env.get_current_market_datetime()
        data = self.env.get_ohlcv(
            self.symbol, dt_from=cdt - timedelta(days=self.days_ago), dt_to=cdt
        )
        min_rows = 2 * self.swing_order + 1
        if len(data) < min_rows:
            return {self.state_key: None}

        data["datetime"] = data["datetime"].apply(lambda x: x.isoformat())
        data = TrendMaturitySymbolStateProvider.calculate_trend_maturity(
            data, swing_order=self.swing_order, lookback=self.lookback,
        )

        feat_cols = ["datetime", "tm_direction", "tm_exhaustion",
                     "tm_bull_waves", "tm_bear_waves",
                     "tm_extending", "tm_retrace_depth", "tm_impulse_ratio",
                     "tm_dir_bias", "tm_efficiency"]

        if self.cache_truncate > 0:
            data = data.tail(self.cache_truncate)

        # Fill NaN with 0.0 for bars where not enough swing data exists
        data = data.fillna(0.0)

        return {self.state_key: data[feat_cols].to_dict(orient="list")}
