import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Tuple


class BaseMarketEnv(object):

    def get_current_market_datetime(self) -> datetime:
        """
        Returns datetime that represents current market date and time
        :return: datetime
        """
        raise NotImplementedError()

    def get_account_value(self) -> float:
        """
        Returns the total account value, all cash and all open positions value
        :return: float
        """
        raise NotImplementedError()

    def get_account_cash(self) -> float:
        """
        Returns the amount of cash available in the account
        :return: float
        """
        raise NotImplementedError()

    def get_ohlcv(
        self, symbol: str, dt_from: datetime, dt_to: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Return a dataframe with OHLCV values for a symbols
        :param symbol: What symbol the data should be retried for
        :param dt_from: From what datetime on should the data be retrieved
        :param dt_to: To what datetime should the data be retrieved
        :return: Pandas DataFrame with results
        """
        raise NotImplementedError()

    def execute_buy_market(self, symbol: str, size: float) -> None:
        """
        Execute a BUY Market order
        :param symbol: Symbol to buy
        :param size: Amount of stocks to buy
        :return: None
        """
        raise NotImplementedError()

    def execute_sell_market(self, symbol: str, size: float) -> None:
        """
        Execute a SELL Market order
        :param symbol: Symbol to sell
        :param size: Amount of stocks to buy
        :return: None
        """
        raise NotImplementedError()

    def execute_buy_limit(self, symbol: str, size: float, price: float) -> None:
        """
        Execute a BUY Limit order
        :param symbol: Symbol to buy
        :param size: Amount of stocks to buy
        :param price: Price at which to trigger the limit order
        :return: None
        """
        raise NotImplementedError()

    def execute_sell_limit(self, symbol: str, size: float, price: float) -> None:
        """
        Execute a SELL Limit order
        :param symbol: Symbol to sell
        :param size: Amount of stocks to sell
        :param price: Price at which to trigger the limit order
        :return: None
        """
        raise NotImplementedError()

    def get_position(self, symbol: str) -> Optional[dict]:
        """
        Gets the status of the current position of a given symbol
        :param symbol: Symbol
        :return: dict or None (if no open positions)
        """
        raise NotImplementedError()

    def execute_close_position(self, symbol: str) -> None:
        """
        Closes position for a symbol
        :param symbol: Symbol
        :return: None
        """
        raise NotImplementedError()

    def get_trade(self, symbol: str) -> List[dict]:
        """
        A Trade is open when the a position in a instrument goes from 0 to a size X which may positive/negative
        for long/short positions)
        A Trade is closed when a position goes from X to 0.
        :param symbol:
        :return: List or orders belonging to the current ongoing trade or last trade
        """
        raise NotImplementedError()

    def get_trades(self, symbol: str, dt_since: datetime) -> List[List[dict]]:
        """
        Returns all trades for a symbol takinginto account orders since dt_since. Including the ongoing trade
        :return [[o1, o2], [o3, o4]]
        """
        raise NotImplementedError()

    def log(self, msg: str):
        print(msg)

    @staticmethod
    def get_last_trade_(orders: List[dict], pos_current_size) -> Optional[List[dict]]:
        """
        From list of orders, reconstruct the trade
        o = {
            'price': orders[-1].executed.price,
            'size': abs(orders[-1].executed.size),
            'instruction': "BUY" / "SELL",
            'size_instruction': orders[-1].executed.size,
        }
        pos_current_size - 0 if no position open,pos/net size if open
        """

        trade = []
        pos_size = pos_current_size

        # take the last order and subtract the position size
        trade.append(orders[-1])
        pos_size -= orders[-1]["size_instruction"]

        # go thru the rest of the orders
        for i in range(2, len(orders) + 1):
            if math.isclose(pos_size, 0, abs_tol=1e-6):  # pos_size == 0:
                break

            trade.append(orders[-1 * i])
            pos_size -= orders[-1 * i]["size_instruction"]

        if not math.isclose(pos_size, 0, abs_tol=1e-6):
            # not enough orders to trace all the way back to the trade beginning
            return None

        trade = trade[::-1]
        for n, o in enumerate(trade):
            o["profit"] = BaseMarketEnv.get_order_pnl(trade[: n + 1])

        return trade

    @staticmethod
    def get_trade_avg_price(trade: List[dict]) -> Tuple[float, float]:
        """
        Function to calculate pnl for a trade
        trade is a dictionary of:
        {
            'price': orders[-1].executed.price,
            'size': abs(orders[-1].executed.size),
            'instruction': "BUY" / "SELL",
        }
        """
        trade_direction = trade[0]["instruction"]

        orders_same_direction = [
            [o["size"], o["price"]]
            for o in trade
            if o["instruction"] == trade_direction
        ]
        orders_opp_direction = [
            (o["size"], o["price"])
            for o in trade
            if o["instruction"] != trade_direction
        ]

        for o_opp in orders_opp_direction:
            o_opp_size = o_opp[0]
            for i in range(len(orders_same_direction)):
                if orders_same_direction[i][0] <= 0:  # order size has been depleted
                    continue

                if (
                    orders_same_direction[i][0] - o_opp_size <= 0
                ):  # all depleted but carry over
                    o_opp_size -= orders_same_direction[i][0]
                    orders_same_direction[i][0] = 0

                if orders_same_direction[i][0] - o_opp_size > 0:  # just takes awau some
                    orders_same_direction[i][0] -= o_opp_size
                    o_opp_size = 0

                if o_opp_size <= 0:
                    break

        trade_price = np.array(orders_same_direction)
        trade_size = trade_price[:, 0].sum()
        trade_price_avg = (trade_price[:, 0] * trade_price[:, 1]).sum() / trade_size
        return trade_size, trade_price_avg

    @staticmethod
    def get_order_pnl(trade: List[dict]) -> float:
        """
        Calculates PnL for the last order using FIFO matching against opening orders.
        """
        if not trade:
            return 0.0

        order_last = trade[-1]
        trade_direction = trade[0]["instruction"]

        # If adding to position (Buy -> Buy), no realized PnL
        if order_last["instruction"] == trade_direction:
            return 0.0

        # 1. Calculate the 'skip' volume (Total previous closed volume)
        # This replaces the specific list comprehension you had
        qty_to_skip = 0.0
        for o in trade[:-1]:
            if o["instruction"] != trade_direction:
                qty_to_skip += o["size"]

        qty_needed = order_last["size"]
        total_cost = 0.0
        total_filled = 0.0

        # 2. Iterate only opening orders to find matches
        for order in trade:
            # Skip unrelated orders (previous closes or the current close)
            if order["instruction"] != trade_direction:
                continue

            order_size = order["size"]

            # FIFO Logic:
            # If we still have 'skip' volume, consume this order's size first
            if qty_to_skip >= order_size:
                qty_to_skip -= order_size
                continue  # This order was fully closed previously

            # If we are here, we have exhausted previous closes (or partially exhausted this one)
            # The available meat on this order is original size minus whatever we needed to skip
            available_on_order = order_size - qty_to_skip
            qty_to_skip = 0.0  # We have now paid our "skip debt"

            # Take what we need, or whatever is left on this order
            take = min(qty_needed, available_on_order)

            total_cost += take * order["price"]
            total_filled += take
            qty_needed -= take

            # Optimization: Stop iterating if we filled the order
            if qty_needed <= 1e-9:
                break

        # 3. Calculate PnL
        if total_filled == 0:
            return 0.0

        avg_entry_price = total_cost / total_filled

        # Direction multiplier: 1 if Long (Sell - Buy), -1 if Short (Buy - Sell)
        # Logic: (Exit - Entry) * Size * (1 if Buy else -1)
        # Simplified: (Exit - Entry) * Size for Long
        #             (Entry - Exit) * Size for Short

        direction_mult = 1 if trade_direction == "BUY" else -1

        # Note: We calculate PnL on the *order_last['size']*
        # If we closed more than we opened (flip), this assumes the flipped portion
        # shares the same entry price (or you might want to limit pnl to total_filled)
        pnl = (
            direction_mult
            * order_last["size"]
            * (order_last["price"] - avg_entry_price)
        )

        return pnl

    # @staticmethod
    # def get_order_pnl(trade: List[dict]) -> float:
    #     """
    #     Function to extract pnl from the last order in the trade
    #     trade is a dictionary of:
    #     {
    #         'price': orders[-1].executed.price,
    #         'size': abs(orders[-1].executed.size),
    #         'instruction': "BUY" / "SELL",
    #     }
    #     """
    #     trade_direction = trade[0]["instruction"]
    #     order_last = trade[-1]

    #     if trade_direction == order_last["instruction"]:
    #         return 0

    #     trade_price_avg = []
    #     order_last_size = order_last["size"]
    #     trade_oppdirection_size = np.sum(
    #         [o["size"] for o in trade[:-1] if o["instruction"] != trade_direction]
    #     )
    #     for order in [o for o in trade if o["instruction"] == trade_direction]:
    #         if order["size"] - trade_oppdirection_size > 0:
    #             size_to_take = min(
    #                 order["size"] - trade_oppdirection_size, order_last_size
    #             )
    #             trade_price_avg.append((order["price"], size_to_take))
    #             order_last_size -= size_to_take
    #             trade_oppdirection_size = min(trade_oppdirection_size - size_to_take, 0)

    #         else:
    #             trade_oppdirection_size -= order["size"]

    #         if order_last_size <= 1e-6:  # 0:
    #             break

    #     if len(trade_price_avg) == 0:
    #         return 0.0

    #     trade_price_avg = np.array(trade_price_avg)
    #     trade_price_avg = (
    #         trade_price_avg[:, 0] * trade_price_avg[:, 1]
    #     ).sum() / trade_price_avg[:, 1].sum()
    #     size_direction = 1 if trade_direction == "BUY" else -1
    #     pnl = (
    #         size_direction
    #         * order_last["size"]
    #         * (order_last["price"] - trade_price_avg)
    #     )
    #     return pnl
