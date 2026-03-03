
# region Imports
from QuantConnect.Indicators import RollingWindow
from AlgorithmImports import (
    AccountType,
    BrokerageName,
    FillGroupingMethod,
    FillMatchingMethod,
    QCAlgorithm,
    Resolution,
    Slice,
    TradeBuilder,
    TradeBarConsolidator,
    TradeBar,
)

import json
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime, timedelta
from qtrader.rlflow.persistence import (
    BasePersistenceProvider,
    NoPersistenceProvider,
    DiskIndexPersistenceProvider,
    LeanDiskIndexPersistenceProvider
)
from qtrader.environments.base import BaseMarketEnv
from qtrader.environments.lean import LeanMarketEnv
from qtrader.agents.base import RandomAgent

# from qtrader.agents.dq import DQAgent
from qtrader.agents.dqtp import DQTPAgent

from qtrader.rlflow.state import StateProviderTask, StateAggregatorTask
from qtrader.rlflow.action import ActTask
from qtrader.rlflow.feedback import FeedbackTask

from qtrader.stateproviders.basic import (
    AccountInfoStateProvider,
    PositionSymbolStateProvider,
    TradeSymbolStateProvider,
    OHLCVSymbolStateProvider,
)
from qtrader.stateproviders.indicators import (
    BridgeBandsSymbolStateProvider,
    MACDSymbolStateProvider,
)

# endregion


class QTraderAlgorithm(QCAlgorithm):

    BAR_PERIOD = timedelta(minutes=15)

    def _load_run_params(self):
        params = {}
        try:
            params_path = self.object_store.get_file_path(f"{self.name}/params.json")
            with open(params_path, "rb") as f:
                params = json.load(f)

        except:
            pass
        finally:
            return params

    def _create_agent(self, name, pprovider, params):
        model_n_layers = int(params.get("model_n_layers", 2))
        model_fl_size = int(params.get("model_fl_size", 128))
        model_shape = params.get("model_shape", "flat")

        agent = DQTPAgent(
            name=name,
            pprovider=pprovider,
            expl_max=1,
            expl_min=float(params.get("expl_min", 0.01)),
            expl_decay=float(params.get("expl_decay", 0.9995)),
            invest_pct=float(params.get("invest_pct", 0.05)),
            n_steps_warmup=int(params.get("n_steps_warmup", 1024)),
            n_step_update=int(params.get("n_step_update", 4)),
            n_steps_checkpoint=int(params.get("n_steps_checkpoint", 5000)),
            exp_memory_size=int(params.get("exp_memory_size", 365 * 100_000)),
            exp_mini_batch_size=int(params.get("exp_mini_batch_size", 32)),
            exp_weighting=float(params.get("exp_weighting", 0.4)),
            exp_w_inc=float(params.get("exp_w_inc", 0.00005)),
            exp_alpha=float(params.get("exp_alpha", 0.8)),
            model_lr=float(params.get("model_lr", 1e-4)),
            model_l2_reg=float(params.get("model_l2_reg", 0)),
            model_layers=[
                (
                    int(model_fl_size / np.power(2, i))
                    if model_shape == "cone"
                    else model_fl_size
                )
                for i in range(model_n_layers)
            ],  # model related
            rl_gamma=float(params.get("rl_gamma", 0.9)),
        )

        return agent

    def initialize(self):
        run_params = self._load_run_params()
        base_name = self.name.split("/")[0]
        self.debug(run_params)

        seed_value = run_params.get("seed", 42)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

        run_type = "LIVE" if self.live_mode else run_params.get("run_type", "TRAIN")
        date_start = datetime.fromisoformat(run_params.get("date_start", "2018-04-05"))
        date_end = datetime.fromisoformat(run_params.get("date_end", "2018-04-06"))

        self.set_start_date(date_start.year, date_start.month, date_start.day)
        self.set_end_date(date_end.year, date_end.month, date_end.day)

        self.set_account_currency("USD")
        self.set_cash(1000)  # Set Strategy Cash

        self.set_brokerage_model(BrokerageName.COINBASE, AccountType.CASH)
        self.set_trade_builder(
            TradeBuilder(FillGroupingMethod.FLAT_TO_FLAT, FillMatchingMethod.FIFO)
        )

        self.exchange = self.add_crypto("BTCUSD", Resolution.MINUTE)
        self.symbol = self.exchange.symbol
        self.set_time_zone(self.exchange.exchange.time_zone)

        # Create rolling window for the maximum lookback (approx 36 days of 15-min bars)
        self.history_window = RollingWindow[TradeBar](3500)

        # Consolidator: 1-min → BAR_PERIOD bars
        self._consolidator = TradeBarConsolidator(self.BAR_PERIOD)
        self._consolidator.data_consolidated += lambda sender, bar: self.history_window.add(bar)

        # -- WARMUP --
        # Fetch minute bars and push them through the consolidator
        self.debug("Warming up consolidator...")
        history_bars = self.history[TradeBar](self.symbol, timedelta(days=36), Resolution.MINUTE)
        for bar in history_bars:
            self._consolidator.update(bar)
        self.debug(f"Warmup complete. Window size: {self.history_window.count}")

        # Now that memory is warmed up, attach RL logic and register with Lean
        self._consolidator.data_consolidated += self._on_consolidated_bar
        self.subscription_manager.add_consolidator(self.symbol, self._consolidator)

        # persistance
        golden_cache_dir = Path(self.object_store.get_file_path("cache")).resolve()
        has_cache = golden_cache_dir.exists()

        if run_type == "WARMUP":
            self.debug(f"{run_type}: Using isolated lean store to avoid Docker volume SQLite concurrency.")
            self.pprovider = LeanDiskIndexPersistenceProvider(prefix=base_name, lean_obj_store=self.object_store)
        else:
            cache_prov = DiskIndexPersistenceProvider(str(golden_cache_dir)) if run_type in ("TRAIN", "EVAL") and has_cache else None
            msg = f"read-through cache from {golden_cache_dir}" if cache_prov else "isolated lean store"
            self.debug(f"{run_type}: Using {msg}")
            self.pprovider = LeanDiskIndexPersistenceProvider(
                prefix=base_name, lean_obj_store=self.object_store, cache_provider=cache_prov
            )
            
        assert isinstance(self.pprovider, BasePersistenceProvider)

        # market env
        self.menv = LeanMarketEnv(
            qcl=self, pprovider=self.pprovider,
            bar_period=self.BAR_PERIOD,
            verbose=(run_type == "LIVE"),
        )
        assert isinstance(self.menv, BaseMarketEnv)

        self.p_run_type = run_type

        # region Agent (skip for WARMUP — no agent needed)

        if run_type == "WARMUP":
            self.agent = None
        else:
            agent = self._create_agent(
                base_name, self.pprovider, run_params.get("hyperparams", {})
            )
            assert isinstance(agent, DQTPAgent)
            agent.no_learn = True if run_type in ("EVAL", "LIVE") else False
            agent.no_full_state = False if run_type == "LIVE" else True
            agent.load_config()
            self.agent = agent

        # endregion

        # region State providers

        self.task_sp_acc_info = StateProviderTask(
            self.menv, self.pprovider, AccountInfoStateProvider, name="AccountInfo"
        )
        self.task_ssp_position = StateProviderTask(
            self.menv,
            self.pprovider,
            PositionSymbolStateProvider,
            name="PositionSymbol",
        )
        self.task_ssp_trade = StateProviderTask(
            self.menv, self.pprovider, TradeSymbolStateProvider, name="TradeSymbol"
        )
        self.task_ssp_ohlcv = StateProviderTask(
            self.menv,
            self.pprovider,
            OHLCVSymbolStateProvider,
            params={"days_ago": 4},
            name="OHLCVSymbol",
            allow_cache=True,
        )

        # -- Bridge Bands: micro (3.5h structure) --
        self.task_ssp_bb_micro = StateProviderTask(
            self.menv,
            self.pprovider,
            BridgeBandsSymbolStateProvider,
            params={"days_ago": 4, "bridge_range_length": 14,
                    "bollinger_bands_length": 14, "hurst_exp_length": 14,
                    "state_key": "bridge_bnds_micro"},
            name="BBMicro",
            allow_cache=True,
        )
        # -- Bridge Bands: daily (1-day volatility regime) --
        self.task_ssp_bb_daily = StateProviderTask(
            self.menv,
            self.pprovider,
            BridgeBandsSymbolStateProvider,
            params={"days_ago": 10, "bridge_range_length": 96,
                    "bollinger_bands_length": 96, "hurst_exp_length": 96,
                    "state_key": "bridge_bnds_daily"},
            name="BBDaily",
            allow_cache=True,
        )
        # -- Bridge Bands: weekly (5-day trend regime) --
        self.task_ssp_bb_weekly = StateProviderTask(
            self.menv,
            self.pprovider,
            BridgeBandsSymbolStateProvider,
            params={"days_ago": 30, "bridge_range_length": 480,
                    "bollinger_bands_length": 480, "hurst_exp_length": 480,
                    "state_key": "bridge_bnds_weekly"},
            name="BBWeekly",
            allow_cache=True,
        )

        # -- MACD: micro (intraday momentum, 12/26/9 on 15m bars) --
        self.task_ssp_macd_micro = StateProviderTask(
            self.menv,
            self.pprovider,
            MACDSymbolStateProvider,
            params={
                "days_ago": 4,
                "ema_short_length": 12,
                "ema_long_length": 26,
                "signal_length": 9,
            },
            name="MACDMicro",
            allow_cache=True,
        )
        # -- MACD: daily (swing momentum, ~1d/3d) --
        self.task_ssp_macd_daily = StateProviderTask(
            self.menv,
            self.pprovider,
            MACDSymbolStateProvider,
            params={
                "days_ago": 10,
                "ema_short_length": 96,
                "ema_long_length": 288,
                "signal_length": 96,
            },
            name="MACDDaily",
            allow_cache=True,
        )
        # -- MACD: weekly (macro momentum, ~5d/10d) --
        self.task_ssp_macd_weekly = StateProviderTask(
            self.menv,
            self.pprovider,
            MACDSymbolStateProvider,
            params={
                "days_ago": 30,
                "ema_short_length": 480,
                "ema_long_length": 960,
                "signal_length": 192,
            },
            name="MACDWeekly",
            allow_cache=True,
        )

        self.task_state_agg = StateAggregatorTask(self.menv, self.pprovider)

        # endregion

        # region Action & Feedback (skip for WARMUP)

        if run_type != "WARMUP":
            self.task_act = ActTask(self.menv, self.pprovider, agent=self.agent)
            self.task_feedback = FeedbackTask(self.menv, self.pprovider, agent=self.agent)

        # endregion

        self.p_symbols = [self.symbol.value]
        self.p_state_prev = None
        self.p_cache_enabled = True

    def on_data(self, data: Slice):
        """Minute-level tick handler — all logic lives in _on_consolidated_bar."""
        pass

    def _on_consolidated_bar(self, sender, bar):
        """Fires every BAR_PERIOD with a consolidated TradeBar."""
        cur_date = self.menv.get_current_market_datetime()
        self.menv.log(f"BAR: {cur_date}")
        if cur_date + self.BAR_PERIOD >= self.end_date:  # last bar, close all
            if self.p_run_type != "WARMUP":
                for sy in self.p_symbols:
                    self.menv.execute_close_position(sy)

            return

        # region Create state (runs for ALL modes including WARMUP)

        rslt_task_sp_acc_info = self.task_sp_acc_info.run(
            cache_enabled=self.p_cache_enabled
        )

        rslt_task_ssp_position = [
            self.task_ssp_position.run(symbol=sy, cache_enabled=self.p_cache_enabled)
            for sy in self.p_symbols
        ]
        rslt_task_ssp_trade = [
            self.task_ssp_trade.run(symbol=sy, cache_enabled=self.p_cache_enabled)
            for sy in self.p_symbols
        ]
        rslt_task_ssp_ohlcv = [
            self.task_ssp_ohlcv.run(symbol=sy, cache_enabled=self.p_cache_enabled)
            for sy in self.p_symbols
        ]
        rslt_task_ssp_bb_micro = [
            self.task_ssp_bb_micro.run(symbol=sy, cache_enabled=self.p_cache_enabled)
            for sy in self.p_symbols
        ]
        rslt_task_ssp_bb_daily = [
            self.task_ssp_bb_daily.run(symbol=sy, cache_enabled=self.p_cache_enabled)
            for sy in self.p_symbols
        ]
        rslt_task_ssp_bb_weekly = [
            self.task_ssp_bb_weekly.run(symbol=sy, cache_enabled=self.p_cache_enabled)
            for sy in self.p_symbols
        ]
        rslt_task_ssp_macd_micro = [
            self.task_ssp_macd_micro.run(symbol=sy, cache_enabled=self.p_cache_enabled)
            for sy in self.p_symbols
        ]
        rslt_task_ssp_macd_daily = [
            self.task_ssp_macd_daily.run(symbol=sy, cache_enabled=self.p_cache_enabled)
            for sy in self.p_symbols
        ]
        rslt_task_ssp_macd_weekly = [
            self.task_ssp_macd_weekly.run(symbol=sy, cache_enabled=self.p_cache_enabled)
            for sy in self.p_symbols
        ]

        state = self.task_state_agg.run(
            symbols=self.p_symbols,
            states_global=[rslt_task_sp_acc_info],
            states_symbol=[
                rslt_task_ssp_position,
                rslt_task_ssp_trade,
                rslt_task_ssp_ohlcv,
                rslt_task_ssp_bb_micro,
                rslt_task_ssp_bb_daily,
                rslt_task_ssp_bb_weekly,
                rslt_task_ssp_macd_micro,
                rslt_task_ssp_macd_daily,
                rslt_task_ssp_macd_weekly,
            ],
        )

        # endregion

        # WARMUP mode: only compute state (already cached above), skip agent
        if self.p_run_type == "WARMUP":
            return

        # region Create action

        actions, state = self.task_act.run(state)
        self.menv.log(
            "\n".join(
                [
                    f"{s}: {[round(p, 2) for p in v['predictions'].tolist()]}"
                    for s, v in actions.items()
                    if "predictions" in v
                ]
            )
        )

        # endregion

        self.task_feedback.run(self.p_state_prev, state)
        self.menv.log(
            f"\tMODEL STATS: N_STEP: {self.agent.n_steps} / N_UPDATES: {self.agent.n_updates} / N_CHECKPOINTS: {self.agent.n_checkpoints}"
        )
        self.menv.log(
            f"\tMODEL STATS: EXPL_RATE: {self.agent.expl_rate} / EXP_W: {self.agent.exp_weighting}"
        )

        if self.agent.ready_to_learn(state):
            self.agent.learn()
            self.menv.log(f"\t{self.agent.learn_timer}")

        self.p_state_prev = state

    def on_order_event(self, order_event):
        self.debug("{} {}".format(self.time, order_event.to_string()))

    def on_end_of_algorithm(self):
        if self.agent and not self.agent.no_learn:
            self.agent.save_config()
            self.agent.save_model(online=True)
            self.agent.save_model(online=False)

        self.pprovider.close()

        if self.p_run_type == "WARMUP":
            return

        agent = self.agent

        # -- DQN convergence metrics --
        if hasattr(agent, "td_tracker_n") and agent.td_tracker_n > 0:
            self.set_summary_statistic(
                "mean_td_error", agent.td_tracker / agent.td_tracker_n
            )

        # if hasattr(agent, "td_rb_max_priority") and agent.td_rb_max_priority > 0:
        #     self.set_summary_statistic(
        #         "mean_td_max_priority", agent.td_rb_max_priority / agent.td_tracker_n
        #     )

        if hasattr(agent, "loss_tracker_n") and agent.loss_tracker_n > 0:
            self.set_summary_statistic(
                "mean_loss", agent.loss_tracker / agent.loss_tracker_n
            )

        if hasattr(agent, "q_value_tracker_n") and agent.q_value_tracker_n > 0:
            self.set_summary_statistic(
                "mean_q_value", agent.q_value_tracker / agent.q_value_tracker_n
            )

        if hasattr(agent, "q_value_diff_tracker") and agent.q_value_tracker_n > 0:
            self.set_summary_statistic(
                "mean_q_value_diff", agent.q_value_diff_tracker / agent.q_value_tracker_n
            )

        # -- Reward metrics --
        if hasattr(agent, "reward_tracker_n") and agent.reward_tracker_n > 0:
            self.set_summary_statistic(
                "mean_shaped_reward", agent.reward_tracker / agent.reward_tracker_n
            )

        # if self.task_feedback.num_feedbacks > 0:
        #     self.set_summary_statistic(
        #         "mean_portfolio_change",
        #         self.task_feedback.ttl_reward / self.task_feedback.num_feedbacks,
        #     )

        if hasattr(agent, "market_log_change_tracker_n") and agent.market_log_change_tracker_n > 0:
            self.set_summary_statistic(
                "mean_market_log_change", agent.market_log_change_tracker / agent.market_log_change_tracker_n
            )

        # -- Agent state --
        self.set_summary_statistic("exploration_rate", agent.expl_rate)
        # self.set_summary_statistic("n_updates", agent.n_updates)
        # self.set_summary_statistic("replay_buffer_size", agent.rb.size)

        self.log(
            "{} - TotalPortfolioValue: {}".format(
                self.time, self.portfolio.total_portfolio_value
            )
        )
        self.log("{} - CashBook: {}".format(self.time, self.portfolio.cash_book))
