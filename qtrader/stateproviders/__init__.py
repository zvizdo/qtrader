from qtrader.environments.base import BaseMarketEnv
from qtrader.rlflow.persistence import BasePersistenceProvider


class BaseStateProvider(object):

    def __init__(self, env: BaseMarketEnv, **kwargs):
        self.env = env

    def load_config(self, pprovider: BasePersistenceProvider):
        pass

    def provide(self):
        raise NotImplementedError()

    def save_config(self, pprovider: BasePersistenceProvider):
        pass


class BaseSymbolStateProvider(BaseStateProvider):

    def __init__(self, env: BaseMarketEnv, symbol: str, **kwargs):
        super(BaseSymbolStateProvider, self).__init__(env, **kwargs)
        self.symbol = symbol
