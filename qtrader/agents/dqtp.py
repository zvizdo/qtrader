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
        n_steps_target_update=1000,
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
        self.n_steps_target_update = n_steps_target_update
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

        self.model_name = "QAgent-Model-{}.keras"
        self.model_online = None
        self.model_target = None

        self.load_model(src="online", dst="online")
        self.load_model(src="target", dst="target")

        self.n_steps = 0  # needs to be loaded and persisted
        self.n_updates = 0  # needs to be loaded and persisted
        self.n_updates_target = 0  # needs to be loaded and persisted

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
        self.td_tracker, self.td_tracker_n = 0, 0
        self.loss_tracker, self.loss_tracker_n = 0.0, 0
        self.q_value_tracker, self.q_value_tracker_n = 0.0, 0
        self.reward_tracker, self.reward_tracker_n = 0.0, 0

    def load_config(self):
        try:
            c = self.pprovider.load_dict("QAgent-Params")
            self.expl_rate = c.get("expl_rate", self.expl_rate)
            self.n_steps = c.get("n_steps", 0)
            self.n_updates = c.get("n_updates", 0)
            self.n_updates_target = c.get("n_updates_target", 0)
            self.exp_weighting = c.get("exp_weighting", self.exp_weighting)

            path = self.pprovider.root_join(self.rb_prefix)
            with open(path, "rb") as f:
                self.rb = pickle.load(f)

        except:
            pass

    def save_config(self):
        self.pprovider.persist_dict(
            name="QAgent-Params",
            obj={
                "expl_rate": self.expl_rate,
                "n_steps": self.n_steps,
                "n_updates": self.n_updates,
                "n_updates_target": self.n_updates_target,
                "exp_weighting": self.exp_weighting,
            },
        )
        path = self.pprovider.root_join(self.rb_prefix)
        with open(path, "wb") as f:
            pickle.dump(self.rb, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _possible_actions(self, symbol: str, state: dict):
        return np.ones(len(self.ACTIONS))

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
                # create random action
                p = np.random.rand(len(self.ACTIONS)) * pa
                ai = np.argmax(p)
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

        return actions

    def feedback(
        self, state: dict, action: dict, reward: dict, state_future: dict
    ) -> None:
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

            # Days in current trade (for holding penalty)
            trade = state["state_symbol"][sy]["trade"]
            cdt = to_dt(state["state_global"]["account"]["current_datetime"])
            if pos_prev and trade:
                days_in_trade = (
                    cdt - to_dt(trade[0]["datetime"])
                ).total_seconds() / 86400.0
            else:
                days_in_trade = 0.0
            reward[sy]["days_in_trade"] = days_in_trade

        # update state
        state["reward"] = reward
        state["state_future"] = state_future

        # persist updated state
        if not self.no_full_state:
            dt = to_dt(state["state_global"]["account"]["current_datetime"]).strftime(
                "%Y%m%d%H"
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
            and self.n_steps % self.n_step_update == 0
        ):
            self.exp_weighting = min(1.0, self.exp_weighting + self.exp_w_inc)
            self.n_updates += 1
            run_learn = True

            # print(f"EXPL. RATE: {self.expl_rate}")

        if self.n_updates > 0 and self.n_steps % self.n_steps_target_update == 0:
            self.expl_rate = max(self.expl_min, self.expl_rate * self.expl_decay)
            self.n_updates_target += 1
            self.copy_weights_to_target()
            self.save_model(online=True)
            self.save_model(online=False)

            #print(f"MODEL TARGET UPDATED!")

        self.n_steps += 1
        # print(
        #     f"N_STEP: {self.n_steps} / N_UPDATES: {self.n_updates} / N_UPDATES_TARGET: {self.n_updates_target}"
        # )

        return run_learn

    def _generate_example(self, symbol: str, state: dict) -> list:
        ex = []

        # START: POSITION
        cdt = to_dt(state["state_global"]["account"]["current_datetime"])
        account_value = state["state_global"]["account"]["value"]

        sym_state = state["state_symbol"][symbol]
        pos = sym_state["position"]
        trade = sym_state["trade"]
        ohlcv = sym_state["ohlcv"]
        price = ohlcv["close"][-1]

        cs_open = ohlcv["open"][-1]
        cs_high = ohlcv["high"][-1]
        cs_low = ohlcv["low"][-1]

        # START: TIME
        ex.append(math.cos(math.pi * cdt.weekday() / 3))
        ex.append(math.sin(math.pi * cdt.weekday() / 3))

        # START: POSITION Direction / Exposure Flag
        # This is exactly 1.0 if in position (LONG), 0.0 if not (FLAT)
        ex.append(0.0 if not pos else float(np.sign(pos["size"])))

        # POSITION Profit (Return on current equity)
        if not pos:
            ex.append(0.0)
        else:
            return_on_current_equity = pos["profit"] / (account_value + 1e-8)
            ex.append(
                math.copysign(1.0, return_on_current_equity)
                * math.log1p(100 * abs(return_on_current_equity))
            )

        # Trade duration features
        ex.append(0.0 if not pos else math.log1p(len(trade)))

        if trade:
            trade_end_dt = to_dt(trade[-1]["datetime"])
            days_since_last = abs((cdt - trade_end_dt).total_seconds()) / 86400.0

            ex.append(0.0 if not pos else math.exp(-days_since_last / 28.0))

            if pos:
                trade_start_dt = to_dt(trade[0]["datetime"])
                days_since_start = abs((cdt - trade_start_dt).total_seconds()) / 86400.0
                ex.append(math.exp(-days_since_start / 28.0))
            else:
                ex.append(-1.0 * math.exp(-days_since_last / 28.0))
        else:
            ex.append(0.0)
            ex.append(0.0)

        # START: BRIDGE BANDS
        bb = sym_state["bridge_bnds"]
        bb_w = bb["bridge_bands_width"]
        bb_pos = bb["bridge_bands_pos"]
        bb_hexp = bb["hurst_exp"]

        # Volatility-Normalized OHLC Returns
        vol = abs(bb_w[-1]) + 1e-8
        ex.append((1.0 - cs_open / price) / vol)
        ex.append((1.0 - cs_high / price) / vol)
        ex.append((1.0 - cs_low / price) / vol)

        for i in [1, 3, 7]:
            ex.append(bb_w[-i])
            ex.append(bb_pos[-i])
            ex.append(bb_hexp[-i])

        # START: MACD
        macd_short = sym_state["macd_12_26_9"]
        macd_long = sym_state["macd_50_200_35"]

        ms_macd  = macd_short["macd"]
        ms_hist  = macd_short["macd_hist"]
        ml_macd  = macd_long["macd"]
        ml_hist  = macd_long["macd_hist"]

        for i in [1, 3, 7]:
            ex.append(ms_macd[-i])
            ex.append(ms_hist[-i])
            ex.append(ml_macd[-i])
            ex.append(ml_hist[-i])

        return ex

    def _generate_reward(self, rewards):
        """
        Reward(t+1) = P(t) * Market_Return
                    - Cost * |P(t) - P(t-1)|
                    - P(t) * alpha * (exp(days_in_trade / tau) - 1)

        P(t)             = 1 for ACTION_LONG, 0 for ACTION_FLAT
        Market_Return    = ln(Close(t+1) / Close(t)) * 10
        Cost             = transaction_cost (per position change)
        alpha            = 0.002  (~10% of typical 2% daily BTC log return)
        tau              = 14 days (exponential time constant)
        """
        transaction_cost = 0.01
        alpha = 0.002  # 10% of typical daily BTC log return (~0.02)
        tau = 14.0  # days; penalty doubles roughly every tau days

        def _compute_single_reward(re):
            p_t = 1.0 if re["action_private"] == self.ACTION_LONG else 0.0
            p_t_prev = re.get("position_prev_indicator", 0.0)
            market_return = re.get("market_log_return", 0.0) * 10
            days_in_trade = re.get("days_in_trade", 0.0)

            holding_penalty = p_t * alpha * (np.exp(days_in_trade / tau) - 1)

            return (
                p_t * market_return
                - transaction_cost * abs(p_t - p_t_prev)
                - holding_penalty
            )

        r = np.array([_compute_single_reward(re) for re in rewards])
        return r

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
        self.q_value_tracker_n += 1
        q_values_future[pa == 0] = -np.inf
        q_f_action_index = np.argmax(q_values_future, axis=1)

        if self.n_updates_target > 0:
            q_values_future_t = model_target.predict(
                examples_all[r:], verbose=0, batch_size=1024
            )
            q_t_t = q_values_future_t[rws, q_f_action_index]
        else:
            q_t_t = q_values_future[rws, q_f_action_index]
        q_a_t = reward + self.rl_gamma * q_t_t

        tds = np.abs(q_a_t - q_a) + 1e-6
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
                    kernel_regularizer=(
                        tf.keras.regularizers.l2(self.model_l2_reg)
                        if self.model_l2_reg > 0
                        else None
                    ),
                )
            )

        layers.append(tf.keras.layers.Dense(output_size, activation="linear"))

        m = tf.keras.Sequential(layers)
        m.compile(
            loss=tf.keras.losses.Huber(),  # "mse",  # "mse",  # tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.model_lr,
                clipnorm=1.0
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
        """Copy online model weights to the target model in-memory."""
        if self.model_online is None:
            return
        if self.model_target is None:
            self.model_target = self._create_model(
                input_size=self.model_online.input_shape[1],
                output_size=self.model_online.output_shape[1],
            )
        self.model_target.set_weights(self.model_online.get_weights())

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
