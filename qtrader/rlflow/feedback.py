import numpy as np
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
        reward = {}

        symbols = state_prev["state_global"]["symbols"]
        for sy in symbols:
            pos = state["state_symbol"][sy]["position"]

            reward[sy] = {}
            reward[sy]["v_po_change"] = 100 * (
                (
                    state["state_global"]["account"]["value"]
                    - state_prev["state_global"]["account"]["value"]
                )
                / state_prev["state_global"]["account"]["value"]
            )

            self.env.log(
                f"[{sy}] PnL: {round(pos['profit'] if pos else 0, 2)}; "
                f"Portfolio Change: {round(reward[sy]['v_po_change'], 4)}%"
            )

        self.ttl_reward += sum([reward[sy]["v_po_change"] for sy in symbols])
        self.num_feedbacks += 1
        agent.feedback(state_prev, state_prev["action"], reward, state)
