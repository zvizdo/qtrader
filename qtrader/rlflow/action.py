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
        super(ActTask, self).__init__(env, pprovider, name="ACT", nout=2, **kwargs)
        self.agent = agent

    def run(self, state: dict, **kwargs):
        env = self.env
        agent = self.agent
        assert isinstance(agent, BaseAgent)

        actions = agent.act(state)
        assert isinstance(actions, dict)

        state["action"] = actions

        return actions, state


class ShapeActionsForMappingTask(BaseTask):

    def __init__(
        self, env: BaseMarketEnv, pprovider: BasePersistenceProvider, **kwargs
    ):
        super(ShapeActionsForMappingTask, self).__init__(
            env, pprovider, name="ShapeActionsForMapping", **kwargs
        )

    def run(self, actions: dict) -> list:
        action_list = []
        for sy, act in actions.items():
            act.update({"symbol": sy})
            action_list.append(act)

        return action_list


class ExecuteActionTask(BaseTask):

    def __init__(
        self, env: BaseMarketEnv, pprovider: BasePersistenceProvider, **kwargs
    ):
        super(ExecuteActionTask, self).__init__(
            env, pprovider, name="ExecuteAction", **kwargs
        )

    def run(self, action: dict):
        env = self.env
        env.log(f"\tAction: {action}")

        a = action.get("action")
        order_type = action.get("type", None)
        if a == "DO_NOTHING":
            pass

        elif a == "BUY" and order_type == "market":
            env.execute_buy_market(symbol=action["symbol"], size=action["size"])

        elif a == "BUY" and order_type == "limit":
            env.execute_buy_limit(
                symbol=action["symbol"], size=action["size"], price=action["price"]
            )

        elif a == "SELL" and order_type == "market":
            env.execute_sell_market(symbol=action["symbol"], size=action["size"])

        elif a == "SELL" and order_type == "limit":
            env.execute_sell_limit(
                symbol=action["symbol"], size=action["size"], price=action["price"]
            )

        elif a == "CLOSE_POSITION":
            env.execute_close_position(symbol=action["symbol"])
