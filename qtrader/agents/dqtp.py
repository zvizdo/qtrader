import time as tm
import numpy as np
from datetime import datetime
import pickle
import math
from qtrader.agents.base import BaseAgent
from qtrader.rlflow.persistence import BasePersistenceProvider
from typing import Dict, Optional
import pandas as pd
import tensorflow as tf
from qtrader.agents.expreplay.buffer import PrioritizedReplayBuffer

to_dt = datetime.fromisoformat


class DQTPAgent(BaseAgent):
    ACTION_FLAT = "FLAT"
    ACTION_LONG = "LONG"

    ACTIONS = [
        ACTION_FLAT,
        ACTION_LONG,
    ]

    def __init__(
        self,
        name: str,
        pprovider: BasePersistenceProvider,
        expl_max=1,
        expl_min=0.01,
        expl_decay=0.9,  # exploration
        invest_pct=0.02,
        n_steps_warmup=1000,
        n_step_update=10,
        n_steps_checkpoint=1000,
        target_tau=0.001,
        exp_memory_size=365,
        exp_mini_batch_size=128,
        exp_weighting=0.4,
        exp_w_inc=0.0005,
        exp_alpha=0.8,  # retraining and exp replay
        model_lr=0.0001,
        model_l2_reg=0.0,
        model_layers=[32],
        model_act_func="elu",  # model related
        rl_gamma=0.9,
        hold_cost_scale=0.085,
        exit_bonus_scale=5.0,
        exit_loss_ratio=1.0,
        action_cooldown_bars=0,
        duration_bonus_scale=0.0,
        opp_cost_scale=0.0,
        bar_period_seconds=3600,
        no_learn=False,
        no_full_state=True,
    ):
        super(DQTPAgent, self).__init__()

        self.name = name

        self.pprovider = pprovider
        assert isinstance(self.pprovider, BasePersistenceProvider)

        self.expl_min = expl_min
        self.expl_decay = expl_decay
        self.expl_rate = expl_max  # needs to be loaded and persisted

        self.invest_pct = invest_pct

        self.n_step_update = n_step_update
        self.n_steps_checkpoint = n_steps_checkpoint
        self.target_tau = target_tau
        self.n_steps_warmup = n_steps_warmup

        self.exp_memory_size = exp_memory_size
        self.exp_mini_batch_size = exp_mini_batch_size
        self.exp_weighting = exp_weighting  # needs to be loaded and persisted
        self.exp_w_inc = exp_w_inc
        self.exp_alpha = exp_alpha

        self.model_lr = model_lr
        self.model_l2_reg = model_l2_reg
        self.model_layers = model_layers
        self.model_act_func = model_act_func

        self.rl_gamma = rl_gamma
        self.hold_cost_scale = hold_cost_scale
        self.exit_bonus_scale = exit_bonus_scale
        self.exit_loss_ratio = exit_loss_ratio
        self.action_cooldown_bars = action_cooldown_bars
        self.duration_bonus_scale = duration_bonus_scale
        self.opp_cost_scale = opp_cost_scale
        self.bar_period_seconds = bar_period_seconds

        self.model_name = "QAgent-Model-{}.keras"
        self.model_online = None
        self.model_target = None

        self.load_model(src="online", dst="online")
        self.load_model(src="target", dst="target")

        self.n_steps = 0  # needs to be loaded and persisted
        self.n_updates = 0  # needs to be loaded and persisted
        self.n_checkpoints = 0  # needs to be loaded and persisted

        self.state_list = "QAgent-State"
        self.state_prefix = f"{self.state_list}"

        # replay buffer
        self.rb_prefix = "QAgent-ReplayBuffer.pkl"
        self.rb = PrioritizedReplayBuffer(
            capacity=self.exp_memory_size, alpha=self.exp_alpha, max_priority=1.0
        )

        self.no_learn = no_learn
        self.no_full_state = no_full_state

        self.learn_timer = dict()
        self.td_tracker, self.td_rb_max_priority, self.td_tracker_n = 0, 0, 0
        self.loss_tracker, self.loss_tracker_n = 0.0, 0
        self.q_value_tracker, self.q_value_diff_tracker, self.q_value_tracker_n = 0.0, 0, 0
        self.reward_tracker, self.reward_tracker_n = 0.0, 0

        # -- Action distribution (per env-step, accumulated in act()) --
        self.action_flat_count = 0
        self.action_long_count = 0
        self.action_total = 0

        # -- Per-action Q-values (per learn() batch) --
        self.q_flat_tracker = 0.0
        self.q_long_tracker = 0.0
        self.q_action_tracker_n = 0

        # -- Reward components (per replay-sample via _reward_active) --
        self.comm_tracker = 0.0
        self.hold_cost_tracker = 0.0
        self.exit_bonus_tracker = 0.0
        self.duration_bonus_tracker = 0.0
        self.opp_cost_tracker = 0.0
        self.reward_component_n = 0

        # -- Boltzmann exploration diagnostics (only populated by bdqtp) --
        self.boltz_argmax_count = 0
        self.boltz_sample_count = 0

    def load_config(self):
        try:
            c = self.pprovider.load_dict("QAgent-Params")
            self.expl_rate = c.get("expl_rate", self.expl_rate)
            self.n_steps = c.get("n_steps", 0)
            self.n_updates = c.get("n_updates", 0)
            self.n_checkpoints = c.get("n_checkpoints", 0)
            self.exp_weighting = c.get("exp_weighting", self.exp_weighting)

            path = self.pprovider.root_join(self.rb_prefix)
            with open(path, "rb") as f:
                self.rb = pickle.load(f)

        except Exception as e:
            print(f"FAILED TO LOAD CONFIG OR REPLAY BUFFER: {e}")

    def save_config(self):
        self.pprovider.persist_dict(
            name="QAgent-Params",
            obj={
                "expl_rate": self.expl_rate,
                "n_steps": self.n_steps,
                "n_updates": self.n_updates,
                "n_checkpoints": self.n_checkpoints,
                "exp_weighting": self.exp_weighting,
            },
        )
        path = self.pprovider.root_join(self.rb_prefix)
        path_tmp = f"{path}.tmp"
        
        try:
            with open(path_tmp, "wb") as f:
                pickle.dump(self.rb, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                import os
                os.fsync(f.fileno())
                
            import os
            os.replace(path_tmp, path)
        except Exception as e:
            print(f"FAILED TO SAVE REPLAY BUFFER: {e}")

    def _possible_actions(self, symbol: str, state: dict):
        pa = np.ones(len(self.ACTIONS))
        if getattr(self, "action_cooldown_bars", 0) > 0:
            pos = state["state_symbol"][symbol]["position"]
            if pos:
                trade = state["state_symbol"][symbol]["trade"]
                if trade:
                    cdt = datetime.fromisoformat(state["state_global"]["account"]["current_datetime"])
                    entry_dt = datetime.fromisoformat(trade[0]["datetime"])
                    bars_held = (cdt - entry_dt).total_seconds() / self.bar_period_seconds
                    if bars_held < self.action_cooldown_bars:
                        pa[0] = 0.0 # Disable FLAT
        return pa

    def _shape_action(self, a: dict, symbol: str, state: dict):
        account_value = state["state_global"]["account"]["value"]
        pos = state["state_symbol"][symbol]["position"]
        price = state["state_symbol"][symbol]["ohlcv"]["close"][-1]

        if a["action_private"] == self.ACTION_FLAT:
            if pos:
                a["action"] = "CLOSE_POSITION"
            else:
                a["action"] = "DO_NOTHING"

        elif a["action_private"] == self.ACTION_LONG:
            if pos:
                a["action"] = "DO_NOTHING"
            else:
                a["action"] = "BUY"
                a["type"] = "market"
                a["size"] = (account_value * self.invest_pct) / price

        return a

    def act(self, state: dict) -> Dict:
        symbols = state["state_global"]["symbols"]

        actions = {}
        for sy in symbols:
            pa = self._possible_actions(sy, state)

            # exploration or model agent prediction
            if self.model_online is None or (
                not self.no_learn and np.random.rand() < self.expl_rate
            ):
                exploration_bias = np.array([0.5, 0.5]) * pa
                exploration_bias /= exploration_bias.sum()
                ai = np.random.choice(len(self.ACTIONS), p=exploration_bias)
                a = self.ACTIONS[ai]
                actions[sy] = {
                    "action_private": a,
                    "method": "random",
                    "actions_possible": pa,
                    "action_index": ai,
                }

            else:
                # use model
                ex = self._generate_example(sy, state)
                ex = np.array([ex], dtype=np.float64)
                p = self.model_online.predict(ex, verbose=0)[0]

                p[pa == 0] = -np.inf
                ai = np.argmax(p)
                a = self.ACTIONS[ai]
                actions[sy] = {
                    "action_private": a,
                    "method": "model",
                    "actions_possible": pa,
                    "predictions": p,
                    "action_index": ai,
                }

        # shape actions
        for sy in symbols:
            actions[sy] = self._shape_action(actions[sy], sy, state)

        # -- Action-fraction tracking (post-shaping, private action space) --
        for sy in symbols:
            ap = actions[sy].get("action_private")
            if ap == self.ACTION_FLAT:
                self.action_flat_count += 1
            elif ap == self.ACTION_LONG:
                self.action_long_count += 1
            self.action_total += 1

        return actions

    def feedback(self, state: dict, action: dict, reward: dict, state_future: dict) -> None:
        # find valid state for symbols
        symbols_valid = [
            sy
            for sy in state["state_global"]["symbols"]
            if sy in state_future["state_global"]["symbols"]
        ]

        if len(symbols_valid) == 0:
            return

        state["state_global"]["symbols"] = symbols_valid

        # enrich reward with fields needed by the reward formula
        for sy in state["state_global"]["symbols"]:
            reward[sy]["action_private"] = action[sy]["action_private"]

            # P(t-1): was there a position before this step?
            pos_prev = state["state_symbol"][sy]["position"]
            reward[sy]["position_prev_indicator"] = 1.0 if pos_prev else 0.0

            # Market log-return: ln(close_future / close_current)
            price_current = state["state_symbol"][sy]["ohlcv"]["close"][-1]
            price_future = state_future["state_symbol"][sy]["ohlcv"]["close"][-1]
            reward[sy]["market_log_return"] = np.log(price_future / price_current)

            # Commission as fraction of position notional
            p_t = 1.0 if action[sy]["action_private"] == self.ACTION_LONG else 0.0
            p_t_prev = reward[sy]["position_prev_indicator"]
            account_value = state["state_global"]["account"]["value"]
            position_notional = account_value * self.invest_pct
            comm_frac = 0.0
            if abs(p_t - p_t_prev) > 0.5:
                trade_future = state_future["state_symbol"][sy]["trade"]
                if trade_future:
                    last_order = trade_future[-1]
                    comm_frac = last_order.get("comm", 0.0) / (position_notional + 1e-8)
            reward[sy]["comm_frac"] = comm_frac

            # Hold duration in hours (for duration-based holding cost)
            cdt = datetime.fromisoformat(
                state["state_global"]["account"]["current_datetime"]
            )
            trade_history = state["state_symbol"][sy]["trade"]
            if pos_prev and trade_history:
                trade_start_dt = datetime.fromisoformat(trade_history[0]["datetime"])
                reward[sy]["hold_hours"] = (
                    abs((cdt - trade_start_dt).total_seconds()) / 3600.0
                )
            else:
                reward[sy]["hold_hours"] = 0.0

            # Trade PnL % on exit (for trade-completion bonus)
            if p_t_prev > 0.5 and p_t < 0.5 and pos_prev:
                cost_basis = pos_prev["value"] - pos_prev["profit"]
                reward[sy]["trade_pnl_pct"] = pos_prev["profit"] / (
                    cost_basis + 1e-8
                )

        # update state
        state["reward"] = reward
        state["state_future"] = state_future

        # persist updated state
        if not self.no_full_state:
            dt = to_dt(state["state_global"]["account"]["current_datetime"]).strftime(
                "%Y%m%d%H%M"
            )
            state_name = f"{self.state_prefix}-{self.name}-{dt}"
            self.pprovider.persist_dict(name=state_name, obj=state)

        # generate example and save state to replay buffer
        if not self.no_learn:
            state_computed = self._generate_examples_from_state(state)
            for s in state_computed:
                self.rb.add(s)

    def ready_to_learn(self, state: dict) -> bool:
        if self.no_learn:
            return False

        run_learn = False

        if (
            self.n_steps > self.n_steps_warmup
            and self.rb.size >= self.exp_mini_batch_size
            and self.n_steps % self.n_step_update == 0
        ):
            self.exp_weighting = min(1.0, self.exp_weighting + self.exp_w_inc)
            self.n_updates += 1
            run_learn = True
            self.copy_weights_to_target()

        if self.n_updates > 0 and self.n_steps % self.n_steps_checkpoint == 0:
            self.expl_rate = max(self.expl_min, self.expl_rate * self.expl_decay)
            self.n_checkpoints += 1
            self.save_model(online=True)
            self.save_model(online=False)

        self.n_steps += 1

        return run_learn

    # Lookback indices: 1h, 24h (1 day), 72h (3 days) on 1H bars
    _LOOKBACKS = [1, 24, 72]

    # PnL trajectory lookbacks (1h, 4h, 12h) — only active when hold_bars >= lookback
    _PNL_LOOKBACKS = [1, 4, 12]

    # Bridge Bands state keys (micro → daily → weekly)
    _BB_KEYS = ["bridge_bnds_micro", "bridge_bnds_daily", "bridge_bnds_weekly"]

    # MACD state keys (micro → daily → weekly)
    _MACD_KEYS = ["macd_6_13_4", "macd_24_72_24", "macd_112_240_48"]

    # Trend Maturity state key
    _TM_KEY = "trend_maturity"
    _TM_FEATURES = [
        "tm_direction", "tm_exhaustion", "tm_bull_waves", "tm_bear_waves",
        "tm_extending", "tm_retrace_depth", "tm_impulse_ratio",
        "tm_dir_bias", "tm_efficiency",
    ]

    def _generate_example(self, symbol: str, state: dict) -> list:
        ex = []

        cdt = to_dt(state["state_global"]["account"]["current_datetime"])
        account_value = state["state_global"]["account"]["value"]

        sym_state = state["state_symbol"][symbol]
        pos = sym_state["position"]
        trade = sym_state["trade"]
        ohlcv = sym_state["ohlcv"]
        price = ohlcv["close"][-1]

        closes = ohlcv["close"]
        cs_open = ohlcv["open"][-1]
        cs_high = ohlcv["high"][-1]
        cs_low = ohlcv["low"][-1]

        # ── TIME (2 features) ──
        week_pos = cdt.weekday() + cdt.hour / 24.0
        ex.append(math.cos(math.pi * week_pos / 3))
        ex.append(math.sin(math.pi * week_pos / 3))

        # ── POSITION (6 features) ──
        ex.append(0.0 if not pos else float(np.sign(pos["size"])))

        if not pos:
            ex.append(0.0)
        else:
            cost_basis = pos["value"] - pos["profit"]
            return_on_position = pos["profit"] / (cost_basis + 1e-8)
            ex.append(
                math.copysign(1.0, return_on_position)
                * math.log1p(100 * abs(return_on_position))
            )

        # Hold duration normalized (0=flat, 1.0=hold cost threshold, max 5.0)
        if pos and trade:
            trade_start_dt = to_dt(trade[0]["datetime"])
            hold_hours = abs((cdt - trade_start_dt).total_seconds()) / 3600.0
            ex.append(min(hold_hours, 360.0) / 72.0)
        else:
            hold_hours = 0.0
            ex.append(0.0)

        # PnL trajectory at [1, 4, 12] bar lookbacks (relative to entry price)
        if pos and trade:
            entry_price = trade[0]["price"]
            hold_bars = hold_hours  # 1H bars, so hold_hours == hold_bars
            for i in self._PNL_LOOKBACKS:
                if hold_bars >= i:
                    ratio = closes[-i] / (entry_price + 1e-8)
                    ex.append(math.log(max(ratio, 1e-8)) * 100)
                else:
                    ex.append(0.0)
        else:
            for _ in self._PNL_LOOKBACKS:
                ex.append(0.0)

        # ── VOL-NORMALIZED OHLC (3 features) ──
        bb_micro = sym_state["bridge_bnds_micro"]
        vol = abs(bb_micro["bridge_bands_width"][-1]) + 1e-8
        ex.append((1.0 - cs_open / price) / vol)
        ex.append((1.0 - cs_high / price) / vol)
        ex.append((1.0 - cs_low / price) / vol)

        # ── CANDLE BODY RATIO (1 feature) ──
        full_range = cs_high - cs_low + 1e-8
        ex.append(abs(price - cs_open) / full_range)

        # ── LOG-RETURNS (3 features) ──
        for i in self._LOOKBACKS:
            ex.append(math.log(closes[-1] / (closes[-1 - i] + 1e-8)) * 100)

        # ── RELATIVE VOLUME (3 features) ──
        volumes = ohlcv["volume"]
        vol_mean = np.mean(volumes[-self._LOOKBACKS[-1]:]) + 1e-8
        for i in self._LOOKBACKS:
            ex.append(math.log1p(volumes[-i] / vol_mean))

        # ── BRIDGE BANDS × 3 scales (9 features each = 27 total) ──
        for bb_key in self._BB_KEYS:
            bb = sym_state[bb_key]
            bb_w = bb["bridge_bands_width"]
            bb_pos = bb["bridge_bands_pos"]
            bb_hexp = bb["hurst_exp"]
            for i in self._LOOKBACKS:
                ex.append(bb_w[-i])
                ex.append(bb_pos[-i])
                ex.append(bb_hexp[-i])

        # ── MACD × 3 scales (6 features each = 18 total) ──
        for macd_key in self._MACD_KEYS:
            macd = sym_state[macd_key]
            m_macd = macd["macd"]
            m_hist = macd["macd_hist"]
            for i in self._LOOKBACKS:
                ex.append(m_macd[-i])
                ex.append(m_hist[-i])

        # ── TREND MATURITY (9 features, current bar only) ──
        tm = sym_state[self._TM_KEY]
        for feat in self._TM_FEATURES:
            ex.append(tm[feat][-1])

        return ex

    _REWARD_SCALE = 75.0  # targets ~93% of rewards in [-1, 1] for BTC 1H
    _R_BAR = 0.59  # sigma_1h × REWARD_SCALE — the per-bar noise floor
    _HOLD_THRESHOLD_HOURS = 72.0  # 3 days free hold, then cost ramps
    _HOLD_COST_POWER = 1.5  # power-law ramp after threshold
    _TRADE_PNL_REF = 0.01  # median |trade_pnl_pct| for tanh normalization (1H BTC scale)
    _DURATION_PEAK_DAYS = 3.0  # reverse-U peak for duration bonus
    _DURATION_HALF_WIDTH = 2.0  # half-width: bonus spans peak ± this (1d to 5d)

    def _generate_reward(self, rewards):
        """Active trader reward: clipped market return + hold duration cost + exit bonus."""
        return np.array([self._reward_active(re) for re in rewards])

    def _reward_active(self, re):
        """Sigma-anchored reward with tanh-compressed exit bonus.

        All shaping components (hold_cost, exit_bonus, duration_bonus) are in
        R_bar units (sigma × REWARD_SCALE ≈ 0.59) and disable when scale=0.

        LONG: clipped market return − hold cost (ramps after 72h)
        FLAT: 0 (no penalty)
        Exit bar: tanh-compressed PnL bonus + duration bonus (reverse-U, peak 3d)
        """
        p_t = 1.0 if re["action_private"] == self.ACTION_LONG else 0.0
        market_return = np.clip(
            re.get("market_log_return", 0.0) * self._REWARD_SCALE, -1.0, 1.0
        )
        trade_cost = re.get("comm_frac", 0.0) * self._REWARD_SCALE

        # Hold cost: free until threshold, then power-law ramp in R_bar units
        hold_hours = re.get("hold_hours", 0.0)
        hold_cost = 0.0
        if self.hold_cost_scale > 0 and hold_hours > self._HOLD_THRESHOLD_HOURS:
            excess_days = (hold_hours - self._HOLD_THRESHOLD_HOURS) / 24.0
            hold_cost = self.hold_cost_scale * self._R_BAR * excess_days ** self._HOLD_COST_POWER

        # Exit bonus + duration bonus: both fire once on trade exit
        exit_bonus = 0.0
        duration_bonus = 0.0
        trade_pnl_pct = re.get("trade_pnl_pct", None)
        if trade_pnl_pct is not None:
            # PnL-based exit bonus (tanh compression, peak = exit_bonus_scale × R_bar)
            if self.exit_bonus_scale > 0:
                x = trade_pnl_pct / self._TRADE_PNL_REF
                if trade_pnl_pct >= 0:
                    exit_bonus = self.exit_bonus_scale * self._R_BAR * math.tanh(x)
                else:
                    exit_bonus = self.exit_bonus_scale * self.exit_loss_ratio * self._R_BAR * math.tanh(x)

            # Duration bonus: one-time at exit, reverse-U peaking at 3d
            if self.duration_bonus_scale > 0:
                hold_days = hold_hours / 24.0
                w = self._DURATION_HALF_WIDTH
                peak = self._DURATION_PEAK_DAYS
                if abs(hold_days - peak) < w:
                    duration_bonus = self.duration_bonus_scale * self._R_BAR * (
                        1.0 - ((hold_days - peak) / w) ** 2
                    )

        # Opportunity cost: penalize flat by missed upside
        opp_cost = 0.0
        if self.opp_cost_scale > 0 and p_t < 0.5:
            opp_cost = self.opp_cost_scale * max(0.0, market_return)

        R_long = market_return - hold_cost

        # -- Reward component tracking (per replay-sample means via learn()) --
        self.comm_tracker += float(trade_cost)
        self.hold_cost_tracker += float(p_t * hold_cost)
        self.exit_bonus_tracker += float(exit_bonus)
        self.duration_bonus_tracker += float(duration_bonus)
        self.opp_cost_tracker += float(opp_cost)
        self.reward_component_n += 1

        return p_t * R_long - trade_cost + exit_bonus + duration_bonus - opp_cost

    def _generate_examples_from_state(self, sf) -> Optional[tuple]:
        state = self.pprovider.load_dict(name=sf) if not isinstance(sf, dict) else sf
        state_future = state["state_future"]

        state_computed = []
        for sy in state["state_global"]["symbols"]:
            ex = self._generate_example(sy, state)
            ex_future = self._generate_example(sy, state_future)
            reward = state["reward"][sy]
            action = state["action"][sy]
            action_future = state_future["action"][sy]

            state_computed.append((sy, ex, ex_future, reward, action, action_future))

        return state_computed

    def learn(self) -> None:
        st_total = tm.time()
        st = tm.time()

        states, weights, indexes = self.rb.sample(
            self.exp_mini_batch_size, beta=self.exp_weighting
        )

        columns = len(states[0][1])
        r = len(states)
        examples_all = np.zeros((r * 2, columns), dtype=np.float64)
        examples_all[:r] = [s[1] for s in states]
        examples_all[r:] = [s[2] for s in states]
        reward = self._generate_reward([s[3] for s in states])

        # Track mean shaped reward
        self.reward_tracker += reward.mean()
        self.reward_tracker_n += 1

        if self.model_online is None:
            self.model_online = self._create_model(
                input_size=columns,
                output_size=len(self.ACTIONS),
            )
            self.copy_weights_to_target()

        self.learn_timer["sampling_dataset_gen"] = round(tm.time() - st, 3)
        st = tm.time()

        model_online = self.model_online
        model_target = self.model_target

        q_values_all = model_online.predict(examples_all, verbose=0, batch_size=1024)
        q_values = q_values_all[:r]
        q_values_future = q_values_all[r:]

        self.learn_timer["model_prediction_dur"] = round(tm.time() - st, 3)
        st = tm.time()

        X = examples_all[:r]
        pa = np.array([s[5]["actions_possible"] for s in states], dtype=np.int64)
        a_index = 1 == np.array(
            [
                np.bincount([s[4]["action_index"]], minlength=len(self.ACTIONS))
                for s in states
            ],
            dtype=np.int64,
        )
        rws = np.arange(r)

        q = q_values
        q_a = q[a_index]

        # Track mean Q-value across the batch
        self.q_value_tracker += q_values.mean()
        self.q_value_diff_tracker += np.ptp(q_values, axis=1).mean()
        self.q_value_tracker_n += 1

        # Per-action Q split (FLAT=idx 0, LONG=idx 1).
        self.q_flat_tracker += float(q_values[:, 0].mean())
        self.q_long_tracker += float(q_values[:, 1].mean())
        self.q_action_tracker_n += 1
        q_values_future[pa == 0] = -np.inf
        q_f_action_index = np.argmax(q_values_future, axis=1)

        q_values_future_t = model_target.predict(
            examples_all[r:], verbose=0, batch_size=1024
        )
        q_t_t = q_values_future_t[rws, q_f_action_index]
        q_a_t = reward + self.rl_gamma * q_t_t
        # q_a_t = np.clip(q_a_t, -15.0, 15.0)
        q_clip = 1.5 / (1.0 - self.rl_gamma)
        q_a_t = np.clip(q_a_t, -q_clip, q_clip)

        tds = np.clip(np.abs(q_a_t - q_a), 0.0, 5.0) + 1e-6 # np.abs(q_a_t - q_a) + 1e-6
        q[a_index] = q_a_t
        y = q

        self.learn_timer["reward_calc"] = round(tm.time() - st, 3)
        st = tm.time()

        # update priorities
        self.rb.update_priorities(indexes, tds)

        self.learn_timer["update_sample_priorities"] = round(tm.time() - st, 3)
        st = tm.time()

        try:
            history = model_online.fit(
                x=X,
                y=y,
                batch_size=self.exp_mini_batch_size,
                epochs=1,
                sample_weight=weights,
                shuffle=False,
                verbose=0,
            )

            # Track Keras training loss
            if history and history.history.get("loss"):
                self.loss_tracker += history.history["loss"][-1]
                self.loss_tracker_n += 1

        except Exception as e:
            print(f"EXCEPTION model_online.fit: {e}")

        self.td_tracker += tds.mean()
        self.td_rb_max_priority += self.rb.max_priority
        self.td_tracker_n += 1

        self.learn_timer["model_fit"] = round(tm.time() - st, 3)
        self.learn_timer["total"] = round(tm.time() - st_total, 3)

    def _create_model(self, input_size: int, output_size: int):
        ki = tf.keras.initializers.GlorotUniform(seed=42)

        layers = [
            tf.keras.Input(shape=(input_size,)),
            tf.keras.layers.LayerNormalization()
        ]
        for n, l in enumerate(self.model_layers):
            layers.append(
                tf.keras.layers.Dense(
                    l,  # input_shape=(input_size,) if n == 0 else None,
                    activation=self.model_act_func,  # "sigmoid",  # "relu",
                    kernel_initializer=ki,
                    # kernel_regularizer=(
                    #     tf.keras.regularizers.l2(self.model_l2_reg)
                    #     if self.model_l2_reg > 0
                    #     else None
                    # ),
                )
            )

        layers.append(tf.keras.layers.Dense(output_size, activation="linear"))

        m = tf.keras.Sequential(layers)
        m.compile(
            loss=tf.keras.losses.Huber(delta=3.0),
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.model_lr,
                clipnorm=1.0,
            ),
        )

        return m

    def load_model(self, src, dst):
        path = self.pprovider.root_join(self.model_name.format(src.capitalize()))

        try:
            if dst == "online":
                self.model_online = tf.keras.models.load_model(path)

            if dst == "target":
                self.model_target = tf.keras.models.load_model(path)

        except Exception as e:
            if dst == "online":
                self.model_online = None

            if dst == "target":
                self.model_target = None

            print(f"NO MODEL FOUND TO LOAD: {e}")

    def save_model(self, prefix="", online=True):
        model = self.model_online if online else self.model_target
        if model is None:
            return

        tag = "Online" if online else "Target"
        tf.keras.models.save_model(
            model,
            self.pprovider.root_join(prefix + self.model_name.format(tag)),
        )

    def copy_weights_to_target(self):
        """Soft-update target model weights via Polyak averaging."""
        if self.model_online is None:
            return
        if self.model_target is None:
            self.model_target = self._create_model(
                input_size=self.model_online.input_shape[1],
                output_size=self.model_online.output_shape[1],
            )
            self.model_target.set_weights(self.model_online.get_weights())
            return
        tau = self.target_tau
        updated = [
            tau * w_online + (1 - tau) * w_target
            for w_online, w_target in zip(
                self.model_online.get_weights(),
                self.model_target.get_weights(),
            )
        ]
        self.model_target.set_weights(updated)

    def _model_callbacks(self, model_type):
        cb_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.pprovider.root_join(self.model_name.format(model_type)),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min",
        )

        return [
            cb_model_checkpoint,
            # tf.keras.callbacks.TensorBoard(self.pprovider.root_join(f'QAgent-Tensorboard/{model_type}'))
        ]
