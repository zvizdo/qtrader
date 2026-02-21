from qtrader.rlflow import BaseTask
from qtrader.environments.base import BaseMarketEnv
from qtrader.rlflow.persistence import BasePersistenceProvider
from qtrader.agents.base import BaseAgent


class ActTask(BaseTask):

    def __init__(
        self,
        env: BaseMarketEnv,
        pprovider: BasePersistenceProvider,
        agent: BaseAgent,
        **kwargs,
    ):
        super(ActTask, self).__init__(env, pprovider, name="ACT", **kwargs)
        self.agent = agent

    def run(self, state: dict, **kwargs):
        env = self.env
        agent = self.agent
        assert isinstance(agent, BaseAgent)

        actions = agent.act(state)
        assert isinstance(actions, dict)

        state["action"] = actions

        # Execute actions
        for sy, act in actions.items():
            a = act.get("action")
            env.log(f"\tAction [{sy}]: {a}")

            if a == "DO_NOTHING":
                pass
            elif a == "BUY":
                order_type = act.get("type", "market")
                if order_type == "market":
                    env.execute_buy_market(symbol=sy, size=act["size"])
                elif order_type == "limit":
                    env.execute_buy_limit(
                        symbol=sy, size=act["size"], price=act["price"]
                    )
            elif a == "CLOSE_POSITION":
                env.execute_close_position(symbol=sy)

        return actions, state
