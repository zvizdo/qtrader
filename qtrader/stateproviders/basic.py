from datetime import timedelta
from qtrader.environments.base import BaseMarketEnv
from qtrader.stateproviders import BaseStateProvider, BaseSymbolStateProvider


class AccountInfoStateProvider(BaseStateProvider):

    def provide(self):
        return {
            "account": {
                "value": self.env.get_account_value(),
                "cash": self.env.get_account_cash(),
                "current_datetime": self.env.get_current_market_datetime().isoformat(),
            }
        }


class PositionSymbolStateProvider(BaseSymbolStateProvider):

    def provide(self):
        return {"position": self.env.get_position(self.symbol)}


class TradeSymbolStateProvider(BaseSymbolStateProvider):

    def provide(self):
        cdt = self.env.get_current_market_datetime()

        return {
            "trade": self.env.get_trade(self.symbol),
            "trades": self.env.get_trades(
                self.symbol, dt_since=cdt - timedelta(days=4 * 7 * 3)
            ),
        }


class OHLCVSymbolStateProvider(BaseSymbolStateProvider):

    def __init__(self, env: BaseMarketEnv, symbol: str, days_ago: int = 365, **kwargs):
        super(OHLCVSymbolStateProvider, self).__init__(env, symbol, **kwargs)
        self.days_ago = days_ago

    def provide(self):
        cdt = self.env.get_current_market_datetime()
        data = self.env.get_ohlcv(
            self.symbol, dt_from=cdt - timedelta(days=self.days_ago), dt_to=cdt
        )
        if len(data) == 0:
            return {"ohlcv": None}

        data["datetime"] = data["datetime"].apply(lambda x: x.isoformat())

        return {"ohlcv": data.to_dict(orient="list")}
