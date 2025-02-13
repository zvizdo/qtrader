# TO-DO DELETE
from qtrader.agents.base import BaseAgent
from qtrader.rlflow import BaseTask, INSTANCE_AGENT


class ReadyToLearnTask(BaseTask):

    def __init__(self, **kwargs):
        super(ReadyToLearnTask, self).__init__(name="ReadyToLearn", **kwargs)

    def run(self, state: dict, **kwargs) -> bool:
        agent = self._get_instance(INSTANCE_AGENT)
        assert isinstance(agent, BaseAgent)

        return agent.ready_to_learn(state)


class LearnTask(BaseTask):

    def __init__(self, **kwargs):
        super(LearnTask, self).__init__(name="LEARN", **kwargs)

    def run(self, **kwargs) -> None:
        agent = self._get_instance(INSTANCE_AGENT)
        assert isinstance(agent, BaseAgent)

        agent.learn()
