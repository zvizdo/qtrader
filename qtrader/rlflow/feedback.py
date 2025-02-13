import numpy as np
from datetime import datetime
from qtrader.agents.base import BaseAgent
from qtrader.rlflow import BaseTask
from qtrader.environments.base import BaseMarketEnv
from qtrader.rlflow.persistence import BasePersistenceProvider


class FeedbackTask(BaseTask):

    def __init__(
        self,
        env: BaseMarketEnv,
        pprovider: BasePersistenceProvider,
        agent: BaseAgent,
        **kwargs,
    ):
        super(FeedbackTask, self).__init__(env, pprovider, name="FEEDBACK", **kwargs)
        self.agent = agent
        assert isinstance(agent, BaseAgent)

        self.ttl_reward = 0
        self.num_feedbacks = 0

    def run(self, state_prev: dict, state: dict, **kwargs) -> None:
        if state_prev is None:
            return None

        agent = self.agent
        to_dt = datetime.fromisoformat

        cdt = to_dt(state["state_global"]["account"]["current_datetime"])
        reward = {}

        symbols = state_prev["state_global"]["symbols"]
        for sy in symbols:
            pos_prev = state_prev["state_symbol"][sy]["position"]
            pos = state["state_symbol"][sy]["position"]

            price = state["state_symbol"][sy]["ohlcv"]["close"][-1]
            price_prev = state["state_symbol"][sy]["ohlcv"]["close"][-2]

            trade = state["state_symbol"][sy]["trade"]
            order_last = None
            trade_direction = None
            trade_days_in = None
            if trade:
                order_last, order_last_profit = trade[-1], trade[-1]["profit"]
                trade_direction = trade[0]["instruction"]
                trade_days_in = (
                    (cdt - to_dt(trade[0]["datetime"])).total_seconds() / 3600 / 24
                    if pos or pos_prev
                    else None
                )

            # calc reward values
            if pos_prev is None and pos is None:
                # no change, no position was opened, no new opened
                reward[sy] = {"v_prev": 0, "v_curr": 0, "v_action": 0, "t_dur": None}

            elif pos_prev is not None and pos is None:
                # position was closed
                v_prev = pos_prev["size"] * price_prev
                v_curr = pos_prev["size"] * order_last["price"]
                reward[sy] = {
                    "v_prev": v_prev,
                    "v_curr": v_curr,
                    "v_action": order_last_profit,
                    "t_dur": trade_days_in,
                }

            elif pos_prev is None and pos is not None:
                # position was just opened
                v_prev = 0
                v_curr = (pos["size"] * price) - (pos["size"] * order_last["price"])
                reward[sy] = {
                    "v_prev": v_prev,
                    "v_curr": v_curr,
                    "v_action": 0,
                    "t_dur": trade_days_in,
                }

            elif pos_prev["size"] == pos["size"]:
                # no change, holding same position
                v_prev = pos_prev["size"] * price_prev
                v_curr = pos["size"] * price
                reward[sy] = {
                    "v_prev": v_prev,
                    "v_curr": v_curr,
                    "v_action": 0,
                    "t_dur": trade_days_in,
                }

            elif pos["size"] > 0 and pos_prev["size"] < pos["size"]:
                # BUY position, but more was added
                size_change = pos["size"] - pos_prev["size"]
                v_prev = pos_prev["size"] * price_prev
                v_curr = ((pos["size"] - size_change) * price) + (
                    (size_change * price) - (size_change * order_last["price"])
                )
                reward[sy] = reward[sy] = {
                    "v_prev": v_prev,
                    "v_curr": v_curr,
                    "v_action": 0,
                    "t_dur": trade_days_in,
                }

            elif pos["size"] < 0 and pos_prev["size"] > pos["size"]:
                # SELL position, but more was added
                size_change = pos["size"] - pos_prev["size"]
                v_prev = pos_prev["size"] * price_prev
                v_curr = ((pos["size"] - size_change) * price) + (
                    (size_change * price) - (size_change * order_last["price"])
                )
                reward[sy] = {
                    "v_prev": v_prev,
                    "v_curr": v_curr,
                    "v_action": 0,
                    "t_dur": trade_days_in,
                }

            elif 0 < pos["size"] < pos_prev["size"]:
                # BUY position, but reduced
                size_change = pos_prev["size"] - pos["size"]
                v_prev = pos_prev["size"] * price_prev
                v_curr = (pos["size"] * price) + (
                    size_change * order_last["price"]
                )  # - (size_change * price_prev)
                reward[sy] = {
                    "v_prev": v_prev,
                    "v_curr": v_curr,
                    "v_action": order_last_profit,
                    "t_dur": trade_days_in,
                }

            elif 0 > pos["size"] > pos_prev["size"]:
                # SELL position, but reduced
                size_change = pos_prev["size"] - pos["size"]
                v_prev = pos_prev["size"] * price_prev
                v_curr = (pos["size"] * price) + (
                    size_change * order_last["price"]
                )  # - (size_change * price_prev)
                reward[sy] = {
                    "v_prev": v_prev,
                    "v_curr": v_curr,
                    "v_action": order_last_profit,
                    "t_dur": trade_days_in,
                }

            # calc current position value
            reward[sy]["v_po_change"] = 100 * (
                (
                    state["state_global"]["account"]["value"]
                    - state_prev["state_global"]["account"]["value"]
                )
                / state_prev["state_global"]["account"]["value"]
            )
            reward[sy]["v_pos"] = None
            reward[sy]["v_invest"] = None
            reward[sy]["v_pnl"] = None
            if pos:
                reward[sy]["v_pos"] = pos["size"] * price
                reward[sy]["v_invest"] = reward[sy]["v_pos"] - pos["profit"]
                reward[sy]["v_pnl"] = pos["profit"]

            elif pos_prev:
                reward[sy]["v_pos"] = pos_prev["size"] * order_last["price"]
                reward[sy]["v_invest"] = reward[sy]["v_pos"] - pos_prev["profit"]
                reward[sy]["v_pnl"] = pos_prev["profit"]

            self.env.log(
                f"[{sy}][{trade_direction if pos else 'n/a'}] PnL: {round(pos['profit'] if pos else 0, 2)}; R: {[f'{k}: {round(v, 2) if v is not None else np.nan}' for k, v in reward[sy].items()]}"
            )

        self.ttl_reward += sum([reward[sy]["v_po_change"] for sy in symbols])
        self.num_feedbacks += 1
        agent.feedback(state_prev, state_prev["action"], reward, state)
