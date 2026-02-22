from AlgorithmImports import OrderProperties, TimeInForce, OrderDirection, OrderStatus

import pandas as pd
from datetime import datetime, timezone, timedelta
from qtrader.environments.base import BaseMarketEnv
from qtrader.rlflow.persistence import BasePersistenceProvider


class LeanMarketEnv(BaseMarketEnv):

    def __init__(self, qcl, pprovider: BasePersistenceProvider, verbose=True):
        super().__init__()

        self.qcl = qcl  # QuantConnect Lean Env

        self.pprovider = pprovider
        self.verbose = verbose
        self._comm_cache = {}  # order_id -> commission (filled orders don't change)

    # START: LeanMarketEnv

    def get_current_market_datetime(self):
        return self.qcl.time  # self.qcl.utc_time.replace(tzinfo=None)

    def get_account_value(self):
        return self.qcl.portfolio.total_portfolio_value

    def get_account_cash(self):
        return self.qcl.portfolio.cash_book["USD"].amount

    def get_ohlcv(self, symbol, dt_from, dt_to=None):
        sy = self.qcl.portfolio[symbol].symbol
        cols = ["datetime", "open", "high", "low", "close", "volume"]
        if not dt_to:
            dt_to = self.get_current_market_datetime()

        df = self.qcl.history(sy, dt_from, dt_to)
        df["datetime"] = df.index.get_level_values("time")
        df = df[cols]

        return df.reset_index(drop=True)

    def execute_buy_market(self, symbol, size):
        self.qcl.market_order(symbol, size)

    def execute_sell_market(self, symbol, size):
        self.execute_buy_market(symbol, -1 * size)

    def execute_buy_limit(self, symbol, size, price):
        dt = self.get_current_market_datetime()
        op = OrderProperties()
        op.time_in_force = TimeInForce.good_til_date(dt + timedelta(hours=36))
        self.qcl.limit_order(symbol, size, price, order_properties=op)

    def execute_sell_limit(self, symbol, size, price):
        self.execute_buy_limit(symbol, -1 * size, price)

    def execute_close_position(self, symbol):
        self.qcl.liquidate(symbol)

    def get_position(self, symbol):
        pos = self.qcl.portfolio[symbol]
        if not pos.invested:
            return None

        return {
            "size": pos.quantity,
            "price_last": pos.price,
            "value": pos.price * pos.quantity,  # pos.average_price * pos.quantity,
            "profit": pos.unrealized_profit,  # pos.quantity * (pos.price - pos.average_price)
        }

    def get_trades(self, symbol, dt_since):
        def map_order(o):
            fill_time = o.last_fill_time.replace(tzinfo=None)

            # Extract actual commission, cached per order ID
            if o.id not in self._comm_cache:
                comm = 0.0
                try:
                    ticket = self.qcl.transactions.get_order_ticket(o.id)
                    for evt in ticket.order_events:
                        comm += abs(float(evt.order_fee.value.amount))
                except Exception:
                    pass
                self._comm_cache[o.id] = comm

            return {
                "id": o.id,
                "symbol": symbol,
                "ts": fill_time.isoformat(),
                "datetime": fill_time.isoformat(),
                "price": o.price,
                "size": o.absolute_quantity,
                "instruction": "BUY" if o.quantity > 0 else "SELL",
                "size_instruction": o.quantity,
                "comm": self._comm_cache[o.id],
            }

        sy = self.qcl.portfolio[symbol].symbol
        trades = [t for t in self.qcl.trade_builder.closed_trades if t.symbol == sy]
        orders = self.qcl.transactions.get_orders(
            lambda x: x.symbol == sy and x.status == OrderStatus.FILLED
        )

        orders = sorted(orders, key=lambda x: x.last_fill_time)

        ti = 0
        to_list = []
        t_curr = []
        for o in orders:
            od = map_order(o)

            # Advance past any closed trades whose exit is before this order's fill time
            while ti < len(trades) and o.last_fill_time > trades[ti].exit_time:
                if t_curr:
                    to_list.append(t_curr)
                t_curr = []
                ti += 1

            t_curr.append(od)
            od["profit"] = BaseMarketEnv.get_order_pnl(t_curr)

        if t_curr:
            to_list.append(t_curr)

        return [
            t for t in to_list if datetime.fromisoformat(t[0]["datetime"]) >= dt_since
        ]

    def get_trade(self, symbol):
        trades = self.get_trades(symbol, datetime(2010, 1, 1))
        if len(trades) > 0:
            return trades[-1]

        return []

    def log(self, msg):
        if self.verbose:
            self.qcl.debug(msg)

    # END: LeanMarketEnv
