import json

import backtrader as bt
import numpy as np
import math
import pandas as pd
import prefect
from backtrader import Analyzer
from datetime import datetime, timedelta
from prefect.executors import LocalDaskExecutor
from qtrader.agents.base import BaseAgent
from qtrader.environments.base import BaseMarketEnv
from qtrader.rlflow.persistence import BasePersistenceProvider
from typing import Optional, List


class BacktestingMarketEnv(BaseMarketEnv):

    def __init__(self, bt_strategy_ref: bt.Strategy,
                 symbols: list):
        self.bt_strategy_ref = bt_strategy_ref
        self.symbols = symbols

    def get_current_market_datetime(self) -> datetime:
        return self.bt_strategy_ref.datas[0].datetime.datetime(0)

    def get_account_value(self) -> float:
        return self.bt_strategy_ref.broker.get_value()

    def get_account_cash(self) -> float:
        return self.bt_strategy_ref.broker.get_cash()

    def get_ohlcv(self, symbol: str, dt_from: datetime, dt_to: Optional[datetime] = None) -> pd.DataFrame:
        ticks_from_now = None
        if not dt_to:
            ticks_from_now = 0
            dt_to = self.get_current_market_datetime()

        ticks_between = (dt_to.date() - dt_from.date()).days
        ticks_from_now = ticks_from_now if ticks_from_now else (
                self.get_current_market_datetime().date() - dt_to.date()).days
        data = {
            'datetime': list(self.bt_strategy_ref.dnames[symbol].datetime.get(size=ticks_between, ago=ticks_from_now)),
            'open': list(self.bt_strategy_ref.dnames[symbol].open.get(size=ticks_between, ago=ticks_from_now)),
            'high': list(self.bt_strategy_ref.dnames[symbol].high.get(size=ticks_between, ago=ticks_from_now)),
            'low': list(self.bt_strategy_ref.dnames[symbol].low.get(size=ticks_between, ago=ticks_from_now)),
            'close': list(self.bt_strategy_ref.dnames[symbol].close.get(size=ticks_between, ago=ticks_from_now)),
            'volume': list(self.bt_strategy_ref.dnames[symbol].volume.get(size=ticks_between, ago=ticks_from_now))
        }
        data['datetime'] = [bt.num2date(dt) for dt in data['datetime']]
        df = pd.DataFrame(data=data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df = df[df.datetime >= dt_from]
        df = df[df.datetime <= dt_to]

        return df

    def execute_buy_market(self, symbol: str, size: int) -> None:
        self.bt_strategy_ref.buy(
            data=self.bt_strategy_ref.dnames[symbol],
            exectype=bt.Order.Market,
            size=size,
            # valid=bt.Order.DAY
        )

    def execute_sell_market(self, symbol: str, size: int) -> None:
        self.bt_strategy_ref.sell(
            data=self.bt_strategy_ref.dnames[symbol],
            exectype=bt.Order.Market,
            size=size,
            # valid=bt.Order.DAY
        )

    def execute_buy_limit(self, symbol: str, size: int, price: float) -> None:
        self.bt_strategy_ref.buy(
            data=self.bt_strategy_ref.dnames[symbol],
            exectype=bt.Order.Limit,
            size=size,
            price=price,
            valid=timedelta(days=1)
        )

    def execute_sell_limit(self, symbol: str, size: int, price: float) -> None:
        self.bt_strategy_ref.sell(
            data=self.bt_strategy_ref.dnames[symbol],
            exectype=bt.Order.Limit,
            size=size,
            price=price,
            valid=timedelta(days=1)
        )

    def get_position(self, symbol: str) -> Optional[dict]:
        pos = self.bt_strategy_ref.getposition(self.bt_strategy_ref.dnames[symbol])
        if not pos:
            return None

        comminfo = self.bt_strategy_ref.broker.getcommissioninfo(self.bt_strategy_ref.dnames[symbol])
        pnl = comminfo.profitandloss(pos.size, pos.price, self.bt_strategy_ref.dnames[symbol].close[0])

        # trade = self.get_trade(symbol)
        # t_size, t_price_avg = BacktestingMarketEnv.get_trade_avg_price(trade)

        return {
            'size': pos.size,
            'price_last': self.bt_strategy_ref.dnames[symbol].close[0],
            'value': self.bt_strategy_ref.dnames[symbol].close[0] * pos.size,
            'profit': pnl
        }

    def execute_close_position(self, symbol: str) -> None:
        self.bt_strategy_ref.close(self.bt_strategy_ref.dnames[symbol])

    def get_trade(self, symbol: str) -> List[dict]:
        current_trade = self.bt_strategy_ref.current_trade[symbol]

        if len(current_trade) > 0:
            # current trade ongoing; return
            return current_trade

        # return last closed trade
        return self.bt_strategy_ref.trade_orders[symbol][-1] if self.bt_strategy_ref.trade_orders[symbol] else []

    def get_trades(self, symbol: str, dt_since: datetime) -> List[List[dict]]:
        trades = []
        to = [o for o in self.bt_strategy_ref.trade_orders[symbol]]

        ct = self.bt_strategy_ref.current_trade[symbol]
        if len(ct) > 0:
            to.append(ct)

        for n in range(1, len(to) + 1):
            trade = to[-1 * n]
            if datetime.fromisoformat(trade[0]['datetime']) < dt_since:
                trades.append(
                    [o for o in trade if datetime.fromisoformat(o['datetime']) >= dt_since]
                )
                break

            trades.append(trade)

        trades = trades[::-1]
        return trades


class RLFLowBacktestingStrategy(bt.Strategy):

    def __init__(self, symbols: List[str], agent: BaseAgent, pprovider: BasePersistenceProvider,
                 fromdate: datetime, todate: datetime,
                 use_dask: bool = False):
        self.env = BacktestingMarketEnv(self, symbols=symbols)

        self.agent = agent
        assert isinstance(self.agent, BaseAgent)

        self.pprovider = pprovider
        assert isinstance(self.pprovider, BasePersistenceProvider)

        self.fromdate = fromdate
        self.todate = todate
        self.use_dask = use_dask

        self.trades = {sy: [] for sy in symbols}
        self.orders = {sy: [] for sy in symbols}
        self.current_trade = {sy: [] for sy in symbols}
        self.trade_orders = {sy: [] for sy in symbols}
        self.state_prev = None

        from qtrader.rlflow.flow import rl_flow
        self.flow = rl_flow

    def next(self):
        if self.env.get_current_market_datetime().date() < self.fromdate.date():
            return

        # Simply log the closing price of the series from the reference
        print(f"Date: {self.env.get_current_market_datetime()}; Value: {self.env.get_account_value()}")

        if (self.todate.date() - self.env.get_current_market_datetime().date()).days <= 5:
            for sy in self.env.symbols:
                self.env.execute_close_position(sy)

            print('LAST DAY!')
            return

        with prefect.context({
            'env': self.env,
            'agent': self.agent,
            'persistence': self.pprovider
        }):
            r = self.flow.run(parameters={
                'symbols': self.env.symbols,
                'state_prev': self.state_prev
            },
                executor=LocalDaskExecutor(scheduler='threads', num_workers=16) if self.use_dask else None
            )
            if r.is_failed():
                raise Exception("RL-flow failed!")
            self.state_prev = r.result[[t for t in self.flow.tasks if t.name == "ACT"][0]].result[1]

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'[{order.p.data.params.name}] BUY EXECUTED at {order.executed.price}')
            elif order.issell():
                print(f'[{order.p.data.params.name}] SELL EXECUTED at {order.executed.price}')

            sy = order.p.data.params.name
            self.orders[sy].append(order)
            o = {
                'datetime': bt.num2date(order.dteos).isoformat(),
                'price': order.executed.price,
                'size': abs(order.executed.size),
                'instruction': "BUY" if order.executed.size > 0 else "SELL",
                'size_instruction': order.executed.size,
                'comm': order.executed.comm
            }
            self.current_trade[sy].append(o)
            self.current_trade[sy][-1]['profit'] = BaseMarketEnv.get_order_pnl(self.current_trade[sy])

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print('Order Canceled/Margin/Rejected')

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        sy = trade.data.params.name
        self.trades[sy].append(trade)
        # move current trade orders to final trades
        trade_current = self.current_trade[sy]
        size_in = sum([o['size_instruction'] for o in trade_current if o['instruction'] == 'BUY'])
        size_out = sum([o['size_instruction'] for o in trade_current if o['instruction'] == 'SELL'])

        self.trade_orders[sy].append(trade_current)
        self.current_trade[sy] = []

        print(f"TRADE FINISHED: In: {abs(round(size_in, 4))} == Out: {abs(round(size_out, 4))} == {math.isclose(abs(size_in), abs(size_out), abs_tol=1e-6)}")
        print(' OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))


class QTraderEvaluationAnalyzer(Analyzer):
    alias = ('QTraderEvaluation',)

    def create_analysis(self):
        '''Replace default implementation to instantiate an AutoOrdereDict
        rather than an OrderedDict'''
        from backtrader.utils import AutoOrderedDict

        self.rets = AutoOrderedDict()

    def start(self):
        super(QTraderEvaluationAnalyzer, self).start()
        symbols = ['BTCUSD']

        self.trades = []
        self.current_trade = {}
        self.trade_orders = {}

    def notify_trade(self, trade):
        if trade.status == trade.Closed:
            self.trades.append(trade)

            # move current trade orders to final trades
            sy = trade.data.params.name
            trade_current = self.current_trade[sy]
            self.trade_orders[sy].append(trade_current)
            self.current_trade[sy] = []

            pnls = np.array(sorted([t.pnlcomm for t in self.trades]))
            print(
                f"NofT: {len(self.trades)}; W/L: {(pnls >= 0).sum()}/{(pnls < 0).sum()}; PnL: {pnls.mean()}"
            )

    def notify_order(self, order):
        if order.status in [order.Completed]:
            o = {
                'datetime': bt.num2date(order.dteos).isoformat(),
                'price': order.executed.price,
                'size': abs(order.executed.size),
                'instruction': "BUY" if order.executed.size > 0 else "SELL",
                'size_instruction': order.executed.size,
                'comm': order.executed.comm
            }
            sy = order.p.data.params.name
            if sy not in self.current_trade:
                self.current_trade[sy] = []

            if sy not in self.trade_orders:
                self.trade_orders[sy] = []

            self.current_trade[sy].append(o)
            self.current_trade[sy][-1]['profit'] = BaseMarketEnv.get_order_pnl(self.current_trade[sy])

    def stop(self):
        sqn_trades = None
        sqn_orders = None

        if self.trades and len(self.trades) >= 30:
            pnls = np.array([t.pnlcomm for t in self.trades])
            pnl_av = np.mean(pnls)
            pnl_stddev = pnls.std()
            try:
                sqn_trades = np.sqrt(pnls.size if pnls.size < 100 else 100.0) * pnl_av / pnl_stddev
            except ZeroDivisionError:
                sqn_trades = None
        else:
            sqn_trades = None

        num_orders = sum([len(t) for sy in self.trade_orders for t in self.trade_orders[sy]])
        if num_orders >= 30:
            trades = []
            for sy in self.trade_orders:
                for t in self.trade_orders[sy]:
                    trades.append(t)

            for t in trades:
                comm = 0
                for o in t:
                    if o['instruction'] == 'BUY':
                        comm += o['comm']

                    else:
                        o['comm'] += comm
                        comm = 0

            orders = [o for t in trades for o in t if o['instruction'] == 'SELL']
            pnls = np.array([o['profit'] - o['comm'] for o in orders])
            pnl_av = np.mean(pnls)
            pnl_stddev = pnls.std()
            try:
                sqn_orders = np.sqrt(pnls.size if pnls.size < 100 else 100.0) * pnl_av / pnl_stddev
            except ZeroDivisionError:
                sqn_orders = None
        else:
            sqn_orders = None

        self.rets.sqn_trades = sqn_trades
        self.rets.trades = self.trades

        self.rets.sqn_orders = sqn_orders
        self.rets.trade_orders = self.trade_orders


class CryptoSpotCommissionInfo(bt.CommissionInfo):
    '''Commission scheme for cryptocurrency spot market.

        Required Args:
            commission: commission fee in percentage, between 0.0 and 1.0
    '''
    params = (
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),  # apply % commission
    )

    def __init__(self):
        assert abs(self.p.commission) < 1.0  # commission is a percentage
        assert self.p.mult == 1.0
        assert self.p.margin is None
        assert self.p.commtype == bt.CommInfoBase.COMM_PERC
        assert self.p.stocklike
        assert self.p.percabs
        assert self.p.leverage == 1.0
        assert self.p.automargin == False

        super().__init__()

    def getsize(self, price, cash):
        '''Support fractional size.

            More details at https://www.backtrader.com/blog/posts/2019-08-29-fractional-sizes/fractional-sizes/.
        '''
        return self.p.leverage * (cash / price)


def run_backtest_strict(data_symbols: list,
                        agent: BaseAgent, pprovider: BasePersistenceProvider,
                        cash_starting: float = 100000.0,
                        dt_from: datetime = None, dt_to: datetime = None,
                        use_dask=False,
                        crypto=True):
    print(data_symbols)
    # Create a Data Feeds
    data_feed = {}
    for s in data_symbols:
        data_feed[s[0]] = pd.read_csv(s[1], parse_dates=True, index_col='datetime')

    params = {
        'symbols': list(data_feed.keys()),
        'agent': agent,
        'pprovider': pprovider,
        'fromdate': dt_from,
        'todate': dt_to,
        'use_dask': use_dask
    }

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash_starting)
    if crypto:
        cerebro.broker.addcommissioninfo(CryptoSpotCommissionInfo(commission=0.005))
    else:
        cerebro.broker.setcommission(commission=0.001)

    cerebro.broker.set_slippage_perc(0.005)

    # Add the analyzers we are interested in
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
    cerebro.addanalyzer(QTraderEvaluationAnalyzer, _name="score")

    for sy, df in data_feed.items():
        cerebro.adddata(
            bt.feeds.PandasData(dataname=df, name=sy, todate=dt_to)
        )

    cerebro.addstrategy(
        RLFLowBacktestingStrategy,
        **params
    )

    strategies = cerebro.run()
    rl = strategies[0]

    ta = rl.analyzers.ta.get_analysis()
    score = rl.analyzers.score.get_analysis()

    to = score['trade_orders']
    rslt = {
        'profit': ta['pnl']['net']['total'] if 'pnl' in ta else 0,
        'cash_start': cash_starting,
        'cash_end': cerebro.broker.getvalue(),
        'eval': {
            'sqn_trades': score['sqn_trades'],
            'sqn_orders': score['sqn_orders'],
            'num_trades': len(score['trades']),
            'num_orders': sum([len(t) for sy in to for t in to[sy]])
        }
    }
    print(json.dumps(rslt, indent=2))
    pprovider.persist_dict('Trade-Orders', to)
    pprovider.persist_dict('Score', rslt)

    return rslt
