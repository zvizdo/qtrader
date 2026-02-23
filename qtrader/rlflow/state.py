import time
import json
import hashlib

from datetime import datetime
from qtrader.environments.base import BaseMarketEnv
from qtrader.rlflow import BaseTask
from qtrader.stateproviders.basic import BaseStateProvider, BaseSymbolStateProvider
from qtrader.rlflow.persistence import BasePersistenceProvider


class StateProviderTask(BaseTask):

    def __init__(
        self,
        env: BaseMarketEnv,
        pprovider: BasePersistenceProvider,
        cls_state_provider,
        params: dict = None,
        allow_cache: bool = False,
        **kwargs,
    ):
        super(StateProviderTask, self).__init__(env, pprovider, **kwargs)
        self.cls_state_provider = cls_state_provider
        self.params = params or {}
        self.params_hashed = hashlib.md5(
            json.dumps(self.params, sort_keys=True).encode()
        ).hexdigest()
        self.allow_cache = allow_cache

    def _key(self, symbol: str, cdt: datetime):
        return f"Flow-State-{symbol}-{cdt.strftime('%Y%m%d%H%M')}-{self.cls_state_provider.__name__}-{self.params_hashed}"

    def run(self, symbol: str = None, cache_enabled: bool = False, **kwargs) -> dict:
        st = time.time()

        env = self.env
        pprovider = self.pprovider

        if self.allow_cache and cache_enabled:
            ckey = self._key(symbol, env.get_current_market_datetime())

            try:
                r = pprovider.load_dict(ckey)
                env.log(
                    f"\tStateProviderTask[{self.cls_state_provider}][{symbol}] - Time: {round(time.time() - st, 3)}s (cache)"
                )
                return r

            except Exception:
                pass

        # Compute (cache miss)
        sp = None
        if issubclass(self.cls_state_provider, BaseSymbolStateProvider):
            sp = self.cls_state_provider(env, symbol, **self.params)

        elif issubclass(self.cls_state_provider, BaseStateProvider):
            sp = self.cls_state_provider(env, **self.params)

        assert isinstance(sp, BaseStateProvider)

        sp.load_config(pprovider)
        rslt = sp.provide()
        sp.save_config(pprovider)

        if self.allow_cache and cache_enabled:
            pprovider.persist_dict(ckey, rslt)

        env.log(
            f"\tStateProviderTask[{sp.__class__}][{symbol}] - Time: {round(time.time() - st, 3)}s (compute)"
        )
        return rslt


class StateAggregatorTask(BaseTask):

    def __init__(
        self, env: BaseMarketEnv, pprovider: BasePersistenceProvider, **kwargs
    ):
        super(StateAggregatorTask, self).__init__(
            env, pprovider, name="StateAgg", **kwargs
        )

    def run(self, symbols: list, states_global: list, states_symbol: list) -> dict:
        state = {
            "state_global": {"symbols": list(symbols)},
            "state_symbol": {},
        }

        for s in states_global:
            state["state_global"].update(s)

        for sy in symbols:
            state["state_symbol"][sy] = {}

        for state_provider in states_symbol:
            for i, sy in enumerate(symbols):
                state["state_symbol"][sy].update(state_provider[i])

        # check invalid states
        symbol_invalid_states = set()
        for sy in symbols:
            for sspn in state["state_symbol"][sy].keys():
                if sspn != "position" and state["state_symbol"][sy][sspn] is None:
                    symbol_invalid_states.add(sy)

        # clean invalid states
        for sy in symbol_invalid_states:
            state["state_global"]["symbols"].remove(sy)
            state["state_symbol"].pop(sy, None)

        return state
