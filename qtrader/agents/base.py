import numpy as np
from typing import Dict


class BaseAgent(object):

    def act(self, state: dict) -> Dict:
        """
        Implement the action of the agent
        :param state: Current state
        :return: Dict: keys are symbols
                action: BUY, SELL, CLOSE_POSITION
                type: market, limit
                size: number
                price: [if type limit] price limit to buy
        """
        raise NotImplementedError()

    def feedback(
        self, state: dict, action: dict, reward: dict, state_future: dict
    ) -> None:
        """
        Accept feedback/reward from previous state
        :param state: Previous state
        :param action: Previous action
        :param reward: Reward
        :param state_future: Current state
        :return: None
        """
        pass

    def ready_to_learn(self, state: dict) -> bool:
        """
        Is the agent ready to learn. If True, learn() will be called
        :return: bool
        """
        return False

    def learn(self) -> None:
        """
        Agent should learn from experience.
        :return:
        """
        raise NotImplementedError()


class RandomAgent(BaseAgent):

    def act(self, state: dict) -> Dict:
        symbols = state["state_global"]["symbols"]
        account_value = state["state_global"]["account"]["value"]

        actions = {}
        for sy in symbols:
            pos = state["state_symbol"][sy]["position"]
            price = state["state_symbol"][sy]["ohlcv"]["close"][-1]

            actions_possible = ["DO_NOTHING", "BUY"]

            if pos:
                actions_possible.append("SELL")
                actions_possible.append("CLOSE_POSITION")

            rnd = np.random.randint(0, len(actions_possible))
            a = actions_possible[rnd]
            action = {"action": a}

            if a not in ["DO_NOTHING", "CLOSE_POSITION"]:
                action["type"] = "market"
                action["size"] = (account_value * 0.1) / price

            actions[sy] = action

        return actions
