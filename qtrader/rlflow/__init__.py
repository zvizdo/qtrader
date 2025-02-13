# import prefect
# from prefect import Task

# INSTANCE_ENV = 'env'
# INSTANCE_AGENT = 'agent'
# INSTANCE_PERSISTENCE = 'persistence'

from qtrader.environments.base import BaseMarketEnv
from qtrader.rlflow.persistence import BasePersistenceProvider


class BaseTask(object):

    def __init__(
        self, env: BaseMarketEnv, pprovider: BasePersistenceProvider, **kwargs
    ):
        self.name = kwargs.get("name", str(self.__class__))

        self.env = env
        assert isinstance(env, BaseMarketEnv)

        self.pprovider = pprovider
        assert isinstance(pprovider, BasePersistenceProvider)
