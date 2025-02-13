import hashlib
import hmac
import time
import requests
import uuid
import logging
import math
import pandas as pd
from typing import Optional, List
from urllib.parse import urlencode
from datetime import datetime, timezone, timedelta
from qtrader.environments.base import BaseMarketEnv
from qtrader.rlflow.persistence import BasePersistenceProvider


class BitstampMarketEnv(BaseMarketEnv):

    def __init__(self, key: str, secret: str, pprovider: BasePersistenceProvider,
                 step_candle_sec: int = 86400,
                 name: str = None):
        super().__init__()

        self.key = key
        self.secret = secret
        self.pprovider = pprovider
        self.step_candle_sec = step_candle_sec
        self.name = name

        self.pprov_prefix = 'BitstampMarketEnv'
        if name is None:
            self.prefix_trades = self.pprov_prefix + "-{}-Trades"
            self.prefix_ord_open = self.pprov_prefix + "-{}-Open"

        else:
            self.prefix_trades = self.pprov_prefix + f"-{self.name}" + "-{}-Trades"
            self.prefix_ord_open = self.pprov_prefix + f"-{self.name}" + "-{}-Open"

    def _call(self, endpoint: str, payload: dict = None):
        r = requests.get(
            f'https://www.bitstamp.net/api/v2/{endpoint}',
            payload
        )

        if not r.ok:
            raise Exception(r.content.decode())

        return r.json()

    def _call_auth(self, endpoint: str, payload: dict):
        timestamp = str(int(round(time.time() * 1000)))
        nonce = str(uuid.uuid4())
        content_type = 'application/x-www-form-urlencoded' if len(payload.keys()) > 0 else ''
        payload_string = urlencode(payload)

        # '' (empty string) in message represents any query parameters or an empty string in case there are none
        message = 'BITSTAMP ' + self.key + \
                  'POST' + \
                  'www.bitstamp.net' + \
                  f'/api/v2/{endpoint}' + \
                  '' + \
                  content_type + \
                  nonce + \
                  timestamp + \
                  'v2' + \
                  payload_string
        message = message.encode('utf-8')
        signature = hmac.new(self.secret.encode(), msg=message, digestmod=hashlib.sha256).hexdigest()
        headers = {
            'X-Auth': 'BITSTAMP ' + self.key,
            'X-Auth-Signature': signature,
            'X-Auth-Nonce': nonce,
            'X-Auth-Timestamp': timestamp,
            'X-Auth-Version': 'v2',
            'Content-Type': content_type
        }

        r = requests.post(
            f'https://www.bitstamp.net/api/v2/{endpoint}',
            headers=headers,
            data=payload_string
        )
        if not r.status_code == 200:
            raise Exception('Status code not 200')

        string_to_sign = (nonce + timestamp + r.headers.get('Content-Type')).encode('utf-8') + r.content
        signature_check = hmac.new(self.secret.encode(), msg=string_to_sign, digestmod=hashlib.sha256).hexdigest()
        if not r.headers.get('X-Server-Auth-Signature') == signature_check:
            raise Exception('Signatures do not match')

        return r.json()

    def _log(self, func, msg, payload):
        cls_name = self.__class__.__name__
        func(
            f"{cls_name}[{self.name if self.name else 'PRIMARY'}]/{msg}/{payload}"
        )

    def _store_open_order(self, symbol, o):
        if 'status' in o and o['status'] == 'error':
            logging.warning(f"BitstampMarketEnv - Order Creation Error: {o}")
            return

        ts = datetime.fromisoformat(o['datetime'])
        candle_delta = timedelta(seconds=self.step_candle_sec)
        delta = ts.replace(tzinfo=timezone.utc).timestamp() % candle_delta.total_seconds()
        cdt_next = datetime.utcfromtimestamp(ts.replace(tzinfo=timezone.utc).timestamp() - delta)

        order = {
            'id': o['id'],
            'symbol': symbol,
            'ts': ts.isoformat(),
            'datetime': (cdt_next - timedelta(seconds=1)).isoformat(),
            'price': float(o['price']),
            'size': float(o['amount']),
            'instruction': "BUY" if o['type'] == '0' else 'SELL',
            'size_instruction': float(o['amount']),
            'comm': 0
        }
        logging.info(f"BitstampMarketEnv - Order Created: {order}")

        self.pprovider.persist_dict(name=self.prefix_ord_open.format(symbol), obj=order)

    def _notify_order(self, order: dict):
        logging.info(f"BitstampMarketEnv - Order Filled: {order}")

        sy = order['symbol']
        id_prefix = self.prefix_trades.format(sy)
        try:
            trades = self.pprovider.load_dict(name=id_prefix)

        except Exception as e:
            trades = []

        # first ever order?
        if len(trades) == 0:
            trades.append([order])
            logging.info(f"BitstampMarketEnv - New Trade Started")

        # new trade?
        elif math.isclose(sum([o['size_instruction'] for o in trades[-1]]), 0, abs_tol=1e-8):
            trades.append([order])
            logging.info(f"BitstampMarketEnv - New Trade Started")

        else:
            trades[-1].append(order)
            logging.info(f"BitstampMarketEnv - Append Order to Existing Trade")

        trades[-1][-1]['profit'] = BitstampMarketEnv.get_order_pnl(trades[-1])
        self.pprovider.persist_dict(name=id_prefix, obj=trades)

    def get_current_ticker_time(self) -> datetime:
        r = self._call('ticker/btcusd/')
        return datetime.utcfromtimestamp(int(r['timestamp']))

    def get_cached_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        id = f"{self.pprov_prefix}-{symbol}"
        try:
            df = self.pprovider.load_dict(id)
            df = pd.DataFrame(df)
            df['datetime'] = df['datetime'].apply(lambda x: datetime.fromisoformat(x))
            df[[c for c in list(df.columns) if c != 'datetime']].apply(lambda s: s.astype(float))

            return df

        except Exception as e:
            return None

    def refresh_ohlcv(self, symbol: str, dt_from: datetime = datetime(2015, 1, 1)):
        id = f"{self.pprov_prefix}-{symbol}"
        df = self.get_cached_ohlcv(symbol)

        date_start = int(dt_from.replace(tzinfo=timezone.utc).timestamp()) \
            if df is None \
            else int(df.iloc[-1]['datetime'].timestamp()) + 1
        date_end = int((datetime.utcnow() - timedelta(days=1)).timestamp())  # int(datetime(2300, 1, 1).timestamp())

        end = date_end

        data_new = []
        while date_start <= end:
            rsp = self._call(f"ohlc/{symbol.lower()}/", payload={
                "step": self.step_candle_sec,
                "limit": 28,
                "start": date_start,
                "end": end
            })

            nd = rsp['data']['ohlc']
            data_new += [d for d in nd if int(d['timestamp']) >= date_start]
            if nd:
                end = int(nd[0]['timestamp']) - 1

            else:
                end = 0

        if not data_new:
            return

        df_new = pd.DataFrame.from_records(data_new)
        df_new['datetime'] = pd.to_datetime(df_new['timestamp'], unit='s', utc=True)
        df_new.sort_values(by=['datetime'], inplace=True)
        df_new.drop('timestamp', inplace=True, axis=1)
        for c in [c for c in list(df_new.columns) if c != 'datetime']:
            df_new[c] = df_new[c].astype(float)

        if df is not None:
            df = pd.concat([df, df_new], ignore_index=True)

        else:
            df = df_new

        df_save = df.copy()
        df_save['datetime'] = df_save['datetime'].apply(lambda x: x.isoformat())
        self.pprovider.persist_dict(id, df_save.to_dict(orient='list'))

    def refresh_orders(self, symbol: str):
        try:
            order = self.pprovider.load_dict(name=self.prefix_ord_open.format(symbol))
            o = self._call_auth('order_status/', payload={
                'id': order['id'],
                'omit_transactions': False
            })

        except Exception as e:
            logging.info(f"BitstampMarketEnv - Order Refresh Error: {e}")
            return

        logging.info(f"BitstampMarketEnv - Order Refresh: {o}")
        if o['status'] == 'Finished':
            sy_one = symbol[:3].lower()
            # sy_two = symbol[3:].lower()
            inst = 1 if order['instruction'] == 'BUY' else -1
            size = sum([float(t[sy_one]) for t in o['transactions']])
            p_avg = sum([(float(t[sy_one]) / size) * float(t['price']) for t in o['transactions']])

            order.update({
                'price': p_avg,
                'size': size,
                'size_instruction': inst * size,
                'comm': sum([float(t['fee']) for t in o['transactions']])
            })
            self._notify_order(order)
            self.pprovider.delete(self.prefix_ord_open.format(symbol))

    # START: BaseMarketEnv Impl
    def get_current_market_datetime(self) -> datetime:
        df = self.get_cached_ohlcv(symbol='BTCUSD')
        return df.iloc[-1]['datetime'].to_pydatetime().replace(tzinfo=None)

    def get_account_value(self) -> float:
        r = self._call_auth('balance/btcusd/', payload={})
        usd = float(r['usd_available'])
        btc = float(r['btc_available'])
        r = self._call('ticker/btcusd/')

        return usd + btc * float(r['last'])

    def get_account_cash(self) -> float:
        r = self._call_auth('balance/btcusd/', payload={})
        return float(r['usd_available'])

    def get_ohlcv(self, symbol: str, dt_from: datetime, dt_to: Optional[datetime] = None) -> pd.DataFrame:
        df = self.get_cached_ohlcv(symbol)

        df = df[df.datetime >= dt_from.replace(tzinfo=timezone.utc)]
        if dt_to:
            df = df[df.datetime <= dt_to.replace(tzinfo=timezone.utc)]
        return df

    def execute_buy_market(self, symbol: str, size: float) -> None:
        o = self._call_auth(f'buy/market/{symbol.lower()}/', payload={
            'amount': round(size, 8)
        })
        self._store_open_order(symbol, o)

    def execute_sell_market(self, symbol: str, size: float) -> None:
        o = self._call_auth(f'sell/market/{symbol.lower()}/', payload={
            'amount': round(size, 8)
        })
        self._store_open_order(symbol, o)

    def execute_buy_limit(self, symbol: str, size: float, price: float) -> None:
        o = self._call_auth(f'buy/{symbol.lower()}/', payload={
            'amount': round(size, 8),
            'price': price,
            'daily_order': True
        })
        self._store_open_order(symbol, o)

    def execute_sell_limit(self, symbol: str, size: float, price: float) -> None:
        o = self._call_auth(f'sell/{symbol.lower()}/', payload={
            'amount': round(size, 8),
            'price': price,
            'daily_order': True
        })
        self._store_open_order(symbol, o)

    def execute_close_position(self, symbol: str) -> None:
        pos = self.get_position(symbol)
        if pos is not None:
            self.execute_sell_market(symbol, abs(pos['size']))

    def get_position(self, symbol: str) -> Optional[dict]:
        r = self._call_auth(f'balance/{symbol.lower()}/', payload={})
        btc = float(r['btc_available'])

        if btc == 0:
            return None

        data = self.get_cached_ohlcv(symbol)
        price_last = data.iloc[-1]['close']
        trade = self.get_trade(symbol)
        t_size, t_price_avg = BitstampMarketEnv.get_trade_avg_price(trade)

        return {
            'size': btc,
            'price_last': price_last,
            'value': price_last * t_size,
            'profit': t_size * (price_last - t_price_avg)
        }

    def get_trade(self, symbol: str) -> List[dict]:
        id_prefix = self.prefix_trades.format(symbol)
        try:
            trades = self.pprovider.load_dict(name=id_prefix)

        except Exception as e:
            trades = []

        if len(trades) > 0:
            return trades[-1]

        return []

    def get_trades(self, symbol: str, dt_since: datetime) -> List[List[dict]]:
        id_prefix = self.prefix_trades.format(symbol)
        try:
            trades = self.pprovider.load_dict(name=id_prefix)

        except Exception as e:
            trades = []

        t = []
        for n in range(1, len(trades) + 1):
            trade = trades[-1 * n]
            if datetime.fromisoformat(trade[0]['datetime']) < dt_since:
                t.append(
                    [o for o in trade if datetime.fromisoformat(o['datetime']) >= dt_since]
                )
                break

            t.append(trade)

        t = t[::-1]
        return t


class MirroredBitstampMarketEnv(BitstampMarketEnv):

    def __init__(self, key: str, secret: str, pprovider: BasePersistenceProvider,
                 mirrored_accounts: List[BitstampMarketEnv],
                 step_candle_sec: int = 86400,
                 name: str = None,):
        super().__init__(key, secret, pprovider, step_candle_sec, name)

        self.order_min_usd = 10.25
        self.mirrored_accounts = mirrored_accounts
        if len([ma for ma in mirrored_accounts if ma.name is None]) > 0:
            raise Exception('Each mirrored account MUST have a name!')

    def get_current_ticker(self, symbol: str) -> datetime:
        r = self._call(f'ticker/{symbol.lower()}/')
        return {
            'timestamp': datetime.utcfromtimestamp(int(r['timestamp'])),
            'price': float(r['last'])
        }

    def refresh_orders(self, symbol: str):
        super().refresh_orders(symbol)

        for ma in self.mirrored_accounts:
            try:
                ma.refresh_orders(symbol)

            except Exception as e:
                logging.warning(f'MirroredBitstampMarketEnv[{ma.name}] / refresh_orders / {e}')

    def _get_order_params(self, symbol, size):
        order_min_usd = self.order_min_usd
        ticker = self.get_current_ticker(symbol)
        value_pa = self.get_account_value()
        size_pct_value_pa = size / (value_pa / ticker['price'])
        order_min_size = order_min_usd / ticker['price']

        return ticker['price'], size_pct_value_pa, order_min_size

    def execute_buy_market(self, symbol: str, size: float) -> None:
        price_last, size_pct_value_pa, order_min_size = self._get_order_params(symbol, size)

        for ma in self.mirrored_accounts:
            try:
                value_ma = ma.get_account_value()
                size_ma = (size_pct_value_pa * value_ma) / price_last
                size_ma = size_ma if size > order_min_size else order_min_size

                # buy either at min size, or the full fraction of PA account
                ma.execute_buy_market(symbol, size_ma)

            except Exception as e:
                logging.warning(f'MirroredBitstampMarketEnv[{ma.name}] / execute_buy_market / {e}')

        # execute primary account action
        super().execute_buy_market(symbol, size)

    def execute_sell_market(self, symbol: str, size: float) -> None:
        price_last, size_pct_value_pa, order_min_size = self._get_order_params(symbol, size)

        for ma in self.mirrored_accounts:
            try:
                value_ma = ma.get_account_value()
                size_ma = (size_pct_value_pa * value_ma) / price_last

                # attempt to sell at full fraction of PA account
                ma.execute_sell_market(symbol, size_ma)

            except Exception as e:
                logging.warning(f'MirroredBitstampMarketEnv[{ma.name}] / execute_sell_market / {e}')

        # execute primary account action
        super().execute_sell_market(symbol, size)

    def execute_buy_limit(self, symbol: str, size: float, price: float) -> None:
        price_last, size_pct_value_pa, order_min_size = self._get_order_params(symbol, size)

        for ma in self.mirrored_accounts:
            try:
                value_ma = ma.get_account_value()
                size_ma = (size_pct_value_pa * value_ma) / price_last
                size_ma = size_ma if size > order_min_size else order_min_size

                # buy either at min size, or the full fraction of PA account
                ma.execute_buy_limit(symbol, size_ma, price)

            except Exception as e:
                logging.warning(f'MirroredBitstampMarketEnv[{ma.name}] / execute_buy_limit / {e}')

        # execute primary account action
        super().execute_buy_limit(symbol, size, price)

    def execute_sell_limit(self, symbol: str, size: float, price: float) -> None:
        price_last, size_pct_value_pa, order_min_size = self._get_order_params(symbol, size)

        for ma in self.mirrored_accounts:
            try:
                value_ma = ma.get_account_value()
                size_ma = (size_pct_value_pa * value_ma) / price_last

                # attempt to sell at full fraction of PA account
                ma.execute_sell_limit(symbol, size_ma, price)

            except Exception as e:
                logging.warning(f'MirroredBitstampMarketEnv[{ma.name}] / execute_sell_market / {e}')

        # execute primary account action
        super().execute_sell_limit(symbol, size, price)

    def execute_close_position(self, symbol: str) -> None:
        super().execute_close_position(symbol)

        for ma in self.mirrored_accounts:
            try:
                ma.execute_close_position(symbol)

            except Exception as e:
                logging.warning(f'MirroredBitstampMarketEnv[{ma.name}] / execute_close_position / {e}')