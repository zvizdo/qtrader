import time as tm
import numpy as np
from datetime import datetime
import pickle
from qtrader.agents.base import BaseAgent
from qtrader.rlflow.persistence import BasePersistenceProvider
from typing import Dict, Optional
import pandas as pd
import tensorflow as tf


to_dt = datetime.fromisoformat


class DQAgent(BaseAgent):
    ACTION_DO_NOTHING = "DO_NOTHING"
    ACTION_BUY_1X = "BUY_1X"
    # ACTION_BUY_2X = "BUY_2X"
    # ACTION_BUY_3X = "BUY_3X"
    ACTION_SELL_1X = "SELL_1X"
    # ACTION_SELL_2X = "SELL_2X"
    # ACTION_SELL_3X = "SELL_3X"
    ACTION_CLOSE_POSITION = "CLOSE_POSITION"

    ACTIONS = [
        ACTION_DO_NOTHING,
        ACTION_BUY_1X,  # ACTION_BUY_2X, ACTION_BUY_3X,
        ACTION_SELL_1X,  # ACTION_SELL_2X, ACTION_SELL_3X,
        ACTION_CLOSE_POSITION,
    ]

    def __init__(
        self,
        name: str,
        pprovider: BasePersistenceProvider,
        expl_max=1,
        expl_min=0.01,
        expl_decay=0.9,  # exploration
        invest_pct=0.02,
        invest_multiplier=(1, 1.5, 2),
        invest_max=0.05,
        no_reduce_pct=0.01,
        buy_only=True,  # how much to invest in an order
        n_steps_warmup=1000,
        n_step_update=10,
        n_steps_target_update=1000,
        exp_memory_size=365,
        exp_mini_batch_size=128,
        exp_weighting=0.4,
        exp_w_inc=0.0005,
        exp_alpha=0.8,  # retraining and exp replay
        model_lr=0.0001,
        model_l2_reg=0.001,
        model_layers=[32],
        model_act_func="sigmoid",  # model related
        rl_gamma=0.9,
        rl_reward_type="position-relative-log",
        rl_nudge_reward_pct=0.0,  # rl related,
        no_learn=False,
        no_full_state=True,
    ):
        super(DQAgent, self).__init__()

        self.name = name

        self.pprovider = pprovider
        assert isinstance(self.pprovider, BasePersistenceProvider)

        self.expl_min = expl_min
        self.expl_decay = expl_decay
        self.expl_rate = expl_max  # needs to be loaded and persisted

        self.invest_pct = invest_pct
        self.invest_multiplier = invest_multiplier
        self.reduce_multiplier = (1 / 2, 1 / 2, 1 / 2)  # (1 / 4, 2 / 4, 3 / 4)
        self.invest_max = invest_max
        self.no_reduce_pct = no_reduce_pct
        self.buy_only = buy_only

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
        self.rl_reward_type = rl_reward_type
        self.rl_nudge_reward_pct = rl_nudge_reward_pct

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
        self.rb = ReplayBuffer(
            capacity=self.exp_memory_size, alpha=self.exp_alpha, max_priority=1.0
        )

        self.no_learn = no_learn
        self.no_full_state = no_full_state

        self.learn_timer = dict()
        self.td_tracker, self.td_tracker_n = 0, 0

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
        account_value = state["state_global"]["account"]["value"]
        pos = state["state_symbol"][symbol]["position"]
        invest_max_reached = False
        no_reduce_func = lambda s: (
            s if s > (account_value * self.no_reduce_pct) / pos["price_last"] else 0
        )
        if pos:
            invest_max_reached = (
                False if account_value * self.invest_max > pos["value"] else True
            )

        price = state["state_symbol"][symbol]["ohlcv"]["close"][-1]
        invest_amounts = [
            account_value * self.invest_pct * im for im in self.invest_multiplier
        ]
        stock_amounts = [
            ia / price if not invest_max_reached else 0 for ia in invest_amounts
        ]

        pactions = np.ones(len(self.ACTIONS))
        # is close position possible
        pactions[self.ACTIONS.index(self.ACTION_CLOSE_POSITION)] = 1 if pos else 0
        # are BUYs/SELLs possible
        if not pos:
            pactions[self.ACTIONS.index(self.ACTION_BUY_1X)] = (
                1 if stock_amounts[0] > 0 else 0
            )
            # pactions[self.ACTIONS.index(self.ACTION_BUY_2X)] = 1 if stock_amounts[1] > 0 else 0
            # pactions[self.ACTIONS.index(self.ACTION_BUY_3X)] = 1 if stock_amounts[2] > 0 else 0
            pactions[self.ACTIONS.index(self.ACTION_SELL_1X)] = (
                1 if not self.buy_only and stock_amounts[0] > 0 else 0
            )
            # pactions[self.ACTIONS.index(self.ACTION_SELL_2X)] = 1 if stock_amounts[1] > 0 else 0
            # pactions[self.ACTIONS.index(self.ACTION_SELL_3X)] = 1 if stock_amounts[2] > 0 else 0

        elif pos and pos["size"] > 0:  # BUY position
            pactions[self.ACTIONS.index(self.ACTION_BUY_1X)] = (
                1 if stock_amounts[0] > 0 else 0
            )
            # pactions[self.ACTIONS.index(self.ACTION_BUY_2X)] = 1 if stock_amounts[1] > 0 else 0
            # pactions[self.ACTIONS.index(self.ACTION_BUY_3X)] = 1 if stock_amounts[2] > 0 else 0

            size = abs(pos["size"])
            reduce_position_amounts = [
                no_reduce_func(abs(size * rm)) for rm in self.reduce_multiplier
            ]
            pactions[self.ACTIONS.index(self.ACTION_SELL_1X)] = (
                1 if reduce_position_amounts[0] > 0 else 0
            )
            # pactions[self.ACTIONS.index(self.ACTION_SELL_2X)] = 1 if reduce_position_amounts[1] > 0 else 0
            # pactions[self.ACTIONS.index(self.ACTION_SELL_3X)] = 1 if reduce_position_amounts[2] > 0 else 0

        elif pos and pos["size"] < 0:  # SELL position
            pactions[self.ACTIONS.index(self.ACTION_SELL_1X)] = (
                1 if stock_amounts[0] > 0 else 0
            )
            # pactions[self.ACTIONS.index(self.ACTION_SELL_2X)] = 1 if stock_amounts[1] > 0 else 0
            # pactions[self.ACTIONS.index(self.ACTION_SELL_3X)] = 1 if stock_amounts[2] > 0 else 0

            size = abs(pos["size"])
            reduce_position_amounts = [
                no_reduce_func(abs(size * rm)) for rm in self.reduce_multiplier
            ]
            pactions[self.ACTIONS.index(self.ACTION_BUY_1X)] = (
                1 if reduce_position_amounts[0] > 0 else 0
            )
            # pactions[self.ACTIONS.index(self.ACTION_BUY_2X)] = 1 if reduce_position_amounts[1] > 0 else 0
            # pactions[self.ACTIONS.index(self.ACTION_BUY_3X)] = 1 if reduce_position_amounts[2] > 0 else 0

        return pactions

    def _shape_action(self, a: dict, symbol: str, state: dict):
        account_value = state["state_global"]["account"]["value"]
        pos = state["state_symbol"][symbol]["position"]
        price = state["state_symbol"][symbol]["ohlcv"]["close"][-1]
        invest_amounts = [
            account_value * self.invest_pct * im for im in self.invest_multiplier
        ]
        stock_amounts = [ia / price for ia in invest_amounts]
        size = abs(pos["size"]) if pos else 0
        reduce_position_amounts = [size * rm for rm in self.reduce_multiplier]

        if a["action_private"].startswith("BUY"):
            if not pos or pos["size"] > 0:  # open BUY position # add to BUY position
                a["action"] = "BUY"
                a["type"] = "limit"
                a["price"] = price
                if a["action_private"] == self.ACTION_BUY_1X:
                    a["size"] = stock_amounts[0]
                # if a['action_private'] == self.ACTION_BUY_2X:
                #     a['size'] = stock_amounts[1]
                # if a['action_private'] == self.ACTION_BUY_3X:
                #     a['size'] = stock_amounts[2]

            elif pos and pos["size"] < 0:  # reduce SELL position
                a["action"] = "BUY"
                a["type"] = "market"
                if a["action_private"] == self.ACTION_BUY_1X:
                    a["size"] = reduce_position_amounts[0]
                # if a['action_private'] == self.ACTION_BUY_2X:
                #     a['size'] = reduce_position_amounts[1]
                # if a['action_private'] == self.ACTION_BUY_3X:
                #     a['size'] = reduce_position_amounts[2]

        elif a["action_private"].startswith("SELL"):
            if not pos or pos["size"] < 0:  # open SELL position # add to SELL position
                a["action"] = "SELL"
                a["type"] = "limit"
                a["price"] = price
                if a["action_private"] == self.ACTION_SELL_1X:
                    a["size"] = stock_amounts[0]
                # if a['action_private'] == self.ACTION_SELL_2X:
                #     a['size'] = stock_amounts[1]
                # if a['action_private'] == self.ACTION_SELL_3X:
                #     a['size'] = stock_amounts[2]

            elif pos and pos["size"] > 0:  # reduce BUY position
                a["action"] = "SELL"
                a["type"] = "market"
                if a["action_private"] == self.ACTION_SELL_1X:
                    a["size"] = reduce_position_amounts[0]
                # if a['action_private'] == self.ACTION_SELL_2X:
                #     a['size'] = reduce_position_amounts[1]
                # if a['action_private'] == self.ACTION_SELL_3X:
                #     a['size'] = reduce_position_amounts[2]

        elif a["action_private"] == self.ACTION_CLOSE_POSITION:
            a["action"] = self.ACTION_CLOSE_POSITION

        else:
            a["action"] = a["action_private"]

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

        # adjust reward
        account_value = state["state_global"]["account"]["value"]
        account_cash = state["state_global"]["account"]["cash"]
        for sy in state["state_global"]["symbols"]:
            reward[sy]["invest_min_pct"] = self.invest_pct
            reward[sy]["invest_value"] = account_value
            reward[sy]["invest_cash"] = account_cash
            reward[sy]["action_private"] = action[sy]["action_private"]

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
            self.expl_rate = max(self.expl_min, self.expl_rate * self.expl_decay)
            self.exp_weighting = min(1.0, self.exp_weighting + self.exp_w_inc)
            self.n_updates += 1
            run_learn = True

            print(f"EXPL. RATE: {self.expl_rate}")

        if self.n_updates > 0 and self.n_steps % self.n_steps_target_update == 0:
            self.n_updates_target += 1
            self.save_model(online=True)  # save online
            self.load_model(src="online", dst="target")  # load online to target
            self.save_model(online=False)  # save target

            print(f"MODEL TARGET UPDATED!")

        self.n_steps += 1
        print(
            f"N_STEP: {self.n_steps} / N_UPDATES: {self.n_updates} / N_UPDATES_TARGET: {self.n_updates_target}"
        )

        return run_learn

    def _generate_example__(self, symbol: str, state: dict) -> Optional[tuple]:
        columns = []
        ex = []

        # START: POSITION
        cdt = datetime.fromisoformat(
            state["state_global"]["account"]["current_datetime"]
        )
        account_value = state["state_global"]["account"]["value"]
        pos = state["state_symbol"][symbol]["position"]
        trade = state["state_symbol"][symbol]["trade"]
        trades = state["state_symbol"][symbol]["trades"]
        price = state["state_symbol"][symbol]["ohlcv"]["close"][-1]
        # END: POSITION

        # START: TIME
        columns.append("WeekdayX")
        ex.append(np.cos(np.pi * cdt.weekday() / 3))
        columns.append("WeekdayY")
        ex.append(np.sin(np.pi * cdt.weekday() / 3))
        # END: TIME

        # START: POSITION
        columns.append("Position_Direction")
        ex.append(0.0 if not pos else np.sign(pos["size"]))

        columns.append("Position_Profit")
        ex.append(
            0.0
            if not pos
            else np.sign(pos["profit"])
            * np.log1p(100 * abs(pos["profit"] / account_value))
        )

        columns.append("Trade_Num_Orders")
        ex.append(0.0 if not pos else np.log1p(len(trade)))

        columns.append("Trade_Last_Order_Direction")
        ex.append(0.0 if not pos else np.sign(trade[-1]["size_instruction"]))

        columns.append("Trade_Since_Last_Order")
        ex.append(
            0.0
            if not pos
            else np.log1p(
                abs((cdt - datetime.fromisoformat(trade[-1]["datetime"])).days) / 28
            )
        )

        columns.append("Trade_In_Time")
        ex.append(
            np.log1p(
                abs((cdt - datetime.fromisoformat(trade[0]["datetime"])).days) / 28
            )
            if pos
            else (
                -1
                * np.log1p(
                    abs((cdt - datetime.fromisoformat(trade[-1]["datetime"])).days) / 28
                )
                if trade
                else 0.0
            )
        )

        trade_order_profit = (
            np.array([o["profit"] for o in trade if o["instruction"] == "SELL"])
            if pos
            else None
        )
        columns.append("Trade_Order_Profit_Ratio")
        ex.append(
            2
            * (
                trade_order_profit[trade_order_profit > 0].sum()
                / (np.abs(trade_order_profit).sum() + 1e6)
                - 0.5
            )
            if pos and trade_order_profit.size > 0
            else 0.0
        )

        columns.append("Trade_Order_Profit_Pct")
        ex.append(
            2 * ((trade_order_profit > 0).sum() / trade_order_profit.size - 0.5)
            if pos and trade_order_profit.size > 0
            else 0.0
        )

        columns.append("Trade_Order_Profit_MaxDDwn")
        max_dd = (
            trade_order_profit.cumsum().min()
            if pos and trade_order_profit.size > 0
            else 0.0
        )
        ex.append(
            np.sign(max_dd) * np.log1p(abs(100 * max_dd / account_value))
            if pos and trade_order_profit.size > 0
            else 0.0
        )

        columns.append("Trade_Order_Profit_MaxDUp")
        max_dup = (
            trade_order_profit.cumsum().max()
            if pos and trade_order_profit.size > 0
            else 0.0
        )
        ex.append(
            np.sign(max_dup) * np.log1p(abs(100 * max_dup / account_value))
            if pos
            else 0.0
        )

        columns.append("Trade_Order_Profit_Avg")
        ewm = (
            pd.Series(trade_order_profit.cumsum()).ewm(alpha=0.5).mean().values[-1]
            if pos and trade_order_profit.size > 0
            else 0.0
        )
        ex.append(
            np.sign(ewm) * np.log1p(abs(100 * ewm / account_value)) if pos else 0.0
        )

        # END: POSITION

        # START: Hist. TRADE/ORDER
        trade_order_profit = np.array(
            [o["profit"] for t in trades for o in t if o["instruction"] == "SELL"]
        )
        columns.append("Trade_HOrder_Profit_Ratio")
        ex.append(
            2
            * (
                trade_order_profit[trade_order_profit > 0].sum()
                / (np.abs(trade_order_profit).sum() + 1e6)
                - 0.5
            )
            if trade_order_profit.size > 0
            else 0.0
        )

        columns.append("Trade_HOrder_Profit_Pct")
        ex.append(
            2 * ((trade_order_profit > 0).sum() / trade_order_profit.size - 0.5)
            if trade_order_profit.size > 0
            else 0.0
        )

        columns.append("Trade_HOrder_Profit_MaxDDwn")
        max_dd = (
            trade_order_profit.cumsum().min() if trade_order_profit.size > 0 else 0.0
        )
        ex.append(
            np.sign(max_dd) * np.log1p(abs(100 * max_dd / account_value))
            if trade_order_profit.size > 0
            else 0.0
        )

        columns.append("Trade_HOrder_Profit_MaxDUp")
        max_dup = (
            trade_order_profit.cumsum().max() if trade_order_profit.size > 0 else 0.0
        )
        ex.append(
            np.sign(max_dup) * np.log1p(abs(100 * max_dup / account_value))
            if trade_order_profit.size > 0
            else 0.0
        )

        for a in [0.5, 0.25, 0.1]:
            columns.append(f"Trade_HOrder_Profit_Avg_a{a}")
            ewm = (
                pd.Series(trade_order_profit.cumsum()).ewm(alpha=a).mean().values[-1]
                if trade_order_profit.size > 0
                else 0.0
            )
            ex.append(
                np.sign(ewm) * np.log1p(abs(100 * ewm / account_value))
                if trade_order_profit.size > 0
                else 0.0
            )
        # END: Hist. TRADE/ORDER

        n_lb = 21
        # START: BRIDGE BANDS
        bb = state["state_symbol"][symbol]["bridge_bnds"]
        for i in range(1, n_lb + 1):
            columns.append(f"BB_W_{str(i - 1).zfill(2)}")
            ex.append(bb["bridge_bands_width"][-i])
            columns.append(f"BB_Pos_{str(i - 1).zfill(2)}")
            ex.append(bb["bridge_bands_pos"][-i])
            columns.append(f"BB_HExp_{str(i - 1).zfill(2)}")
            ex.append(bb["hurst_exp"][-i])
        # END: BRIDGE BANDS

        # START: MODEL
        m = state["state_symbol"][symbol]["model_ind"]
        columns.append(f"Model_AUC")
        ex.append(m["params"]["eval"]["auc"])
        for i in range(len(m["preds"]["p"])):
            p = m["preds"]["p"][i]
            if not np.isnan(m["preds"]["target"][i]):  # hist prediction
                t = m["preds"]["target"][i]
                columns.append(f"Model_P_{i}")
                ex.append(-2 * (abs(t - p) - 0.5))

            else:
                columns.append(f"Model_P_{i}")
                ex.append(p)
        # END: MODEL

        return ex

    def _generate_example(self, symbol: str, state: dict) -> Optional[tuple]:
        columns = []
        ex = []

        # START: POSITION
        cdt = to_dt(state["state_global"]["account"]["current_datetime"])
        account_value = state["state_global"]["account"]["value"]
        pos = state["state_symbol"][symbol]["position"]
        trade = state["state_symbol"][symbol]["trade"]
        trades = state["state_symbol"][symbol]["trades"]
        price = state["state_symbol"][symbol]["ohlcv"]["close"][-1]

        cs_open = state["state_symbol"][symbol]["ohlcv"]["open"][-1]
        cs_high = state["state_symbol"][symbol]["ohlcv"]["high"][-1]
        cs_low = state["state_symbol"][symbol]["ohlcv"]["low"][-1]
        # END: POSITION

        # START: TIME
        columns.append("WeekdayX")
        ex.append(np.cos(np.pi * cdt.weekday() / 3))
        columns.append("WeekdayY")
        ex.append(np.sin(np.pi * cdt.weekday() / 3))
        # END: TIME

        # START: POSITION
        columns.append("Position_Direction")
        ex.append(0.0 if not pos else np.sign(pos["size"]))

        columns.append("Position_Profit")
        ex.append(
            0.0
            if not pos
            else np.sign(pos["profit"])
            * np.log1p(100 * abs(pos["profit"] / (account_value - pos["profit"])))
        )

        columns.append("Trade_Num_Orders")
        ex.append(0.0 if not pos else np.log1p(len(trade)))

        columns.append("Trade_Since_Last_Order")
        ex.append(
            0.0
            if not pos
            else np.log1p(
                abs((cdt - to_dt(trade[-1]["datetime"])).total_seconds() / 3600 / 24)
                / 28
            )
        )

        columns.append("Trade_In_Time")
        ex.append(
            np.log1p(
                abs((cdt - to_dt(trade[0]["datetime"])).total_seconds() / 3600 / 24)
                / 28
            )
            if pos
            else (
                -1
                * np.log1p(
                    abs(
                        (cdt - to_dt(trade[-1]["datetime"])).total_seconds() / 3600 / 24
                    )
                    / 28
                )
                if trade
                else 0.0
            )
        )

        trade_order_profit = (
            np.array([o["profit"] for o in trade if o["instruction"] == "SELL"])
            if pos
            else None
        )
        columns.append("Trade_Order_Profit_Ratio")
        ex.append(
            2
            * (
                trade_order_profit[trade_order_profit > 0].sum()
                / (np.abs(trade_order_profit).sum() + 1e6)
                - 0.5
            )
            if pos and trade_order_profit.size > 0
            else 0.0
        )

        # END: POSITION

        columns.append("Rel_Open")
        ex.append(1 - cs_open / price)
        columns.append("Rel_High")
        ex.append(1 - cs_high / price)
        columns.append("Rel_Low")
        ex.append(1 - cs_low / price)

        # n_lb = 3
        # START: BRIDGE BANDS
        bb = state["state_symbol"][symbol]["bridge_bnds"]
        for i in [1, 3, 7]:  # range(1, n_lb + 1):
            columns.append(f"BB_W_{str(i - 1).zfill(2)}")
            ex.append(bb["bridge_bands_width"][-i])
            columns.append(f"BB_Pos_{str(i - 1).zfill(2)}")
            ex.append(bb["bridge_bands_pos"][-i])
            columns.append(f"BB_HExp_{str(i - 1).zfill(2)}")
            ex.append(bb["hurst_exp"][-i])
        # END: BRIDGE BANDS

        # START: MACD

        macd_short = state["state_symbol"][symbol]["macd_12_26_9"]
        macd_long = state["state_symbol"][symbol]["macd_50_200_35"]
        for i in [1, 3, 7]:
            columns.append(f"MACD_Short_{str(i - 1).zfill(2)}")
            ex.append(macd_short["macd"][-i])
            columns.append(f"MACD_Short_Hist_{str(i - 1).zfill(2)}")
            ex.append(macd_short["macd_hist"][-i])

            columns.append(f"MACD_Long_{str(i - 1).zfill(2)}")
            ex.append(macd_long["macd"][-i])
            columns.append(f"MACD_Long_Hist_{str(i - 1).zfill(2)}")
            ex.append(macd_long["macd_hist"][-i])

        # END: MACD

        return ex

    def _generate_reward(self, rl_reward_type, rewards, **kwargs):
        # def sigmoid(x, d=50.):
        #     z = 1 / (1 + np.exp(-x / d))
        #     return 2 * z - 1

        # if rl_reward_type == 'point-simple':
        #     r = np.array([
        #         0. if re['v_curr'] - re['v_prev'] == 0 else 1
        #         if re['v_curr'] - re['v_prev'] >= 0 else -1.
        #         for re in rewards
        #     ])
        #     return r

        # if rl_reward_type == 'position-relative':
        #     r = np.array([
        #         (re['v_curr'] - re['v_prev']) / re['v_pos'] if re['v_pos'] else 0.
        #         for re in rewards
        #     ])
        #
        #     return r

        if rl_reward_type == "portfolio-change-log":
            r = np.array([re["v_po_change"] for re in rewards])

            return np.sign(r) * np.log1p(np.abs(r))

        if rl_reward_type == "position-relative-log":
            r = np.array(
                [
                    (
                        (re["v_curr"] - re["v_prev"]) / re["v_invest"]
                        if re["v_invest"]
                        else 0.0
                    )
                    for re in rewards
                ]
            )

            p = self.rl_nudge_reward_pct
            r_nudge = np.zeros(r.size)
            if p > 0:
                for i, re in enumerate(rewards):
                    a = re["action_private"]

                    # closed trade within 3 to 10 days
                    if (
                        re["v_invest"]
                        and a == DQAgent.ACTION_CLOSE_POSITION
                        and 3 <= re["t_dur"] <= 10
                    ):
                        r_nudge[i] += p * re["v_invest"]

                    # take profit early
                    if (
                        re["v_invest"]
                        and a in [DQAgent.ACTION_CLOSE_POSITION, DQAgent.ACTION_SELL_1X]
                        and (re["v_pnl"] / re["v_invest"]) > self.invest_pct
                    ):
                        r_nudge[i] += p * re["v_invest"]

            r += r_nudge
            return np.sign(r) * np.log1p(np.abs(100 * r))

        if rl_reward_type == "position-relative-action-log":
            r = np.array([100 * re["v_action"] / re["invest_value"] for re in rewards])

            return np.sign(r) * np.log1p(np.abs(r))

        if rl_reward_type == "portfolio-change-plus-action-log":
            r = np.array([100 * re["v_action"] / re["invest_value"] for re in rewards])
            r += np.array([re["v_po_change"] for re in rewards])

            return np.sign(r) * np.log1p(np.abs(r))

        # if rl_reward_type == 'position-relative-log-action':
        #     r_action = np.array([
        #         100 * re['v_action'] / re['v_pos'] if re['v_pos'] else 0.
        #
        #         for re in rewards
        #     ])
        #     r = np.array([
        #         100 * (re['v_curr'] - re['v_prev']) / re['v_pos'] if re['v_pos'] else 0.
        #         for re in rewards
        #     ])
        #
        #     return np.sign(r) * np.log1p(np.abs(r + r_action))

        # if rl_reward_type == 'position-relative-sig':
        #     r = np.array([
        #         (re['v_curr'] - re['v_prev']) / re['v_pos'] if re['v_pos'] else 0.
        #         for re in rewards
        #     ])
        #
        #     return np.sign(r) * sigmoid(np.abs(100 * r), 2)

        # if rl_reward_type == 'position-relative-sig-action':
        #     r_action = np.array([
        #         100 * re['v_action'] / re['v_pos'] if re['v_pos'] else 0.
        #
        #         for re in rewards
        #     ])
        #     r = np.array([
        #         100 * (re['v_curr'] - re['v_prev']) / re['v_pos'] if re['v_pos'] else 0.
        #         for re in rewards
        #     ])
        #
        #     return np.sign(r) * sigmoid(np.abs(r + r_action), 2)

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
        r = self.exp_mini_batch_size
        examples_all = np.zeros((r * 2, columns), dtype=np.float64)
        examples_all[:r] = [s[1] for s in states]
        examples_all[r:] = [s[2] for s in states]
        reward = self._generate_reward(self.rl_reward_type, [s[3] for s in states])

        if self.model_online is None:
            self.model_online = self._create_model(
                input_size=columns,
                output_size=len(self.ACTIONS),
            )
            self.save_model(online=True)  # save online
            self.load_model(src="online", dst="target")  # load online to target
            self.save_model(online=False)  # save target

        self.learn_timer["sampling_dataset_gen"] = round(tm.time() - st, 3)
        st = tm.time()

        model_online = self.model_online
        model_target = self.model_target

        q_values_all = model_online.predict(examples_all, verbose=0, batch_size=1024)
        q_values = q_values_all[:r]
        q_values_future = q_values_all[r:]
        q_values_future_t = model_target.predict(
            examples_all[r:], verbose=0, batch_size=1024
        )

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
        q_values_future[pa == 0] = -np.inf
        q_f_action_index = np.argmax(q_values_future, axis=1)
        q_t_t = (
            q_values_future_t[rws, q_f_action_index]
            if self.n_updates_target > 0
            else q_values_future[rws, q_f_action_index]  # np.zeros(r)
        )
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
            model_online.fit(
                x=X,
                y=y,
                batch_size=self.exp_mini_batch_size,
                epochs=1,
                # validation_data=(X, y),
                # validation_batch_size=self.exp_mini_batch_size,
                sample_weight=weights,
                shuffle=False,
                verbose=2,
                # callbacks=self._model_callbacks(mt)
            )

        except Exception as e:
            print(f"EXCEPTION model_online.fit: {e}")

        self.td_tracker += tds.mean()
        self.td_tracker_n += 1

        self.learn_timer["model_fit"] = round(tm.time() - st, 3)
        self.learn_timer["total"] = round(tm.time() - st_total, 3)

    def _create_model(
        self,
        input_size: int,
        output_size: int,
    ):
        ki = tf.keras.initializers.GlorotUniform(seed=42)

        layers = [tf.keras.Input(shape=(input_size,))]
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
            loss="mse",  # "mse",  # tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.model_lr,
                # clipnorm=self.model_l2_reg
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
        if self.model_online is None:
            return

        tf.keras.models.save_model(
            self.model_online if online else self.model_target,
            self.pprovider.root_join(
                prefix + self.model_name.format("Online" if online else "Target")
            ),
        )

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


class ReplayBuffer(object):
    def __init__(self, capacity, alpha, max_priority):
        """
        ### Initialize
        """
        # We use a power of $2$ for capacity because it simplifies the code and debugging
        self.capacity = capacity
        # $\alpha$
        self.alpha = alpha

        # Maintain segment binary trees to take sum and find minimum over a range
        self.priority_sum = np.zeros(2 * self.capacity, dtype=np.float32)
        self.priority_min = np.ones(2 * self.capacity, dtype=np.float32) * np.inf

        # Current max priority, $p$, to be assigned to new transitions
        self.max_priority = max_priority

        # Arrays for buffer
        self.data = np.zeros(self.capacity, dtype=object)

        # We use cyclic buffers to store data, and `next_idx` keeps the index of the next empty
        # slot
        self.next_idx = 0

        # Size of the buffer
        self.size = 0

    def add(self, state, priority=None):
        """
        ### Add sample to queue
        """

        # Get next available slot
        idx = self.next_idx

        # store in the queue
        self.data[idx] = state

        # Increment next available slot
        self.next_idx = (idx + 1) % self.capacity
        # Calculate the size
        self.size = min(self.capacity, self.size + 1)

        # $p_i^\alpha$, new samples get `max_priority`
        priority_alpha = (
            self.max_priority**self.alpha if priority is None else priority**self.alpha
        )
        # Update the two segment trees for sum and minimum
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):
        """
        #### Set priority in binary segment tree for minimum
        """

        # Leaf of the binary tree
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the minimum of it's two children
            self.priority_min[idx] = min(
                self.priority_min[2 * idx], self.priority_min[2 * idx + 1]
            )

    def _set_priority_sum(self, idx, priority):
        """
        #### Set priority in binary segment tree for sum
        """

        # Leaf of the binary tree
        idx += self.capacity
        # Set the priority at the leaf
        self.priority_sum[idx] = priority

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the sum of it's two children
            self.priority_sum[idx] = (
                self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]
            )

    def _sum(self):
        """
        #### $\sum_k p_k^\alpha$
        """

        # The root node keeps the sum of all values
        return self.priority_sum[1]

    def _min(self):
        """
        #### $\min_k p_k^\alpha$
        """

        # The root node keeps the minimum of all values
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        """
        #### Find largest $i$ such that $\sum_{k=1}^{i} p_k^\alpha  \le P$
        """

        # Start from the root
        idx = 1
        while idx < self.capacity:
            # If the sum of the left branch is higher than required sum
            if self.priority_sum[idx * 2] > prefix_sum:
                # Go to left branch of the tree
                idx = 2 * idx
            else:
                # Otherwise go to right branch and reduce the sum of left
                #  branch from required sum
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        # We are at the leaf node. Subtract the capacity by the index in the tree
        # to get the index of actual value
        return idx - self.capacity

    def sample(self, batch_size, beta):
        """
        ### Sample from buffer
        """

        # Initialize samples
        rnd = np.random.random(batch_size) * self._sum()
        weights = np.zeros(shape=batch_size, dtype=np.float32)
        indexes = np.array(
            [self.find_prefix_sum_idx(rnd[i]) for i in range(batch_size)],
            dtype=np.int32,
        )

        # $\min_i P(i) = \frac{\min_i p_i^\alpha}{\sum_k p_k^\alpha}$
        prob_min = self._min() / self._sum()
        # $\max_i w_i = \bigg(\frac{1}{N} \frac{1}{\min_i P(i)}\bigg)^\beta$
        max_weight = (prob_min * self.size) ** (-beta)

        states = [None] * batch_size
        for i in range(batch_size):
            idx = indexes[i]
            # $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            # $w_i = \bigg(\frac{1}{N} \frac{1}{P(i)}\bigg)^\beta$
            weight = (prob * self.size) ** (-beta)
            # Normalize by $\frac{1}{\max_i w_i}$,
            #  which also cancels off the $\frac{1}{N}$ term
            weights[i] = weight / max_weight
            states[i] = self.data[idx]

        return states, weights, indexes

    def update_priorities(self, indexes, priorities):
        """
        ### Update priorities
        """

        for idx, priority in zip(indexes, priorities):
            # Set current max priority
            self.max_priority = max(self.max_priority, priority)

            # Calculate $p_i^\alpha$
            priority_alpha = priority**self.alpha
            # Update the trees
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_full(self):
        """
        ### Whether the buffer is full
        """
        return self.capacity == self.size
