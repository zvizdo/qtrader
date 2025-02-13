from prefect import Flow, Parameter, case, unmapped
from qtrader.rlflow.state import StateProviderTask, StateAggregatorTask
from qtrader.rlflow.action import ActTask, ShapeActionsForMappingTask, ExecuteActionTask
from qtrader.rlflow.feedback import FeedbackTask
from qtrader.rlflow.learn import ReadyToLearnTask, LearnTask

# START: Set up BaseStateProviders

from qtrader.stateproviders.basic import AccountInfoStateProvider

task_sp_account_info = StateProviderTask(AccountInfoStateProvider, name="AccountInfo")

# END: Set up BaseStateProviders

# START: Set up BaseSymbolStateProviders

from qtrader.stateproviders.basic import (
    PositionSymbolStateProvider,
    TradeSymbolStateProvider,
    OHLCVSymbolStateProvider,
)

task_ssp_position = StateProviderTask(
    PositionSymbolStateProvider, name="PositionSymbol"
)
task_ssp_trade = StateProviderTask(TradeSymbolStateProvider, name="TradeSymbol")
task_ssp_ohlcv = StateProviderTask(
    OHLCVSymbolStateProvider,
    params={"days_ago": 365},
    name="OHLCVSymbol",
    allow_cache=False,
)

from qtrader.stateproviders.indicators import (
    BridgeBandsSymbolStateProvider,
    TrendlinesSymbolStateProvider,
)

task_ssp_bb = StateProviderTask(
    BridgeBandsSymbolStateProvider,
    params={"days_ago": 365},
    name="BridgeBandsSymbol",
    allow_cache=True,
)

task_ssp_tl = StateProviderTask(
    TrendlinesSymbolStateProvider, name="TrendlinesSymbol", allow_cache=True
)

from qtrader.stateproviders.model import IndicatorModelSymbolStateProvider

task_ssp_indm = StateProviderTask(
    IndicatorModelSymbolStateProvider,
    params={"refresh_rate_days": 35, "days_ago": 28 * 12 * 3, "num_trials": 50},
    name="IndicatorModelSymbol",
    allow_cache=True,
)

# END: Set up BaseStateProviders

task_state_agg = StateAggregatorTask()

task_act = ActTask()
task_shape_actions = ShapeActionsForMappingTask()
task_action_exec = ExecuteActionTask()

task_feedback = FeedbackTask()

task_ready_to_learn = ReadyToLearnTask()
task_learn = LearnTask()

with Flow("rl-flow") as rl_flow:
    # START: Flow Parameters
    p_symbols = Parameter("symbols", required=True, default=[])
    p_state_prev = Parameter("state_prev", required=False, default=None)
    p_cache_enabled = Parameter("cache_enabled", required=False, default=True)
    # END: Flow Parameters

    # START: Global State
    rslt_task_sp_account_info = task_sp_account_info(cache_enabled=p_cache_enabled)
    # END: Global State

    # START: Symbol State
    rslt_task_ssp_position = task_ssp_position.map(
        symbol=p_symbols, cache_enabled=unmapped(p_cache_enabled)
    )
    rslt_task_ssp_trade = task_ssp_trade.map(
        symbol=p_symbols, cache_enabled=unmapped(p_cache_enabled)
    )
    rslt_task_ssp_ohlcv = task_ssp_ohlcv.map(
        symbol=p_symbols, cache_enabled=unmapped(p_cache_enabled)
    )
    rslt_task_ssp_bb = task_ssp_bb.map(
        symbol=p_symbols, cache_enabled=unmapped(p_cache_enabled)
    )
    rslt_task_ssp_indm = task_ssp_indm.map(
        symbol=p_symbols, cache_enabled=unmapped(p_cache_enabled)
    )
    # rslt_task_ssp_tl = task_ssp_tl.map(symbol=p_symbols, cache_enabled=unmapped(p_cache_enabled))
    # END: Symbol State

    state = task_state_agg(
        symbols=p_symbols,
        states_global=[rslt_task_sp_account_info],
        states_symbol=[
            rslt_task_ssp_position,
            rslt_task_ssp_trade,
            rslt_task_ssp_ohlcv,
            rslt_task_ssp_bb,
            rslt_task_ssp_indm,
            # rslt_task_ssp_tl,
        ],
    )

    # ACTION
    actions, state_with_actions = task_act(state)
    action_list = task_shape_actions(actions)
    task_action_exec.map(action_list)

    # FEEDBACK
    feedback = task_feedback(
        state_prev=p_state_prev,
        state=state_with_actions,
    )

    # LEARN
    with case(
        task_ready_to_learn(state=state_with_actions, upstream_tasks=[feedback]), True
    ):
        task_learn()
