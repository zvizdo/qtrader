import sys
sys.path.insert(0, '.')

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
import utils
from qtrader.rlflow.persistence import SQLitePersistenceProvider

st.set_page_config(
    page_title="Smart Trader Dashboard",
    layout="wide"
)

utils.download_db()
ppprovider = SQLitePersistenceProvider(root='.')
states = sorted(ppprovider.list(prefix="QAgent-State"), reverse=True)
iters = sorted(list(set(['-'.join(s.split('-')[:3]) for s in states])), reverse=True)

st.title('Smart Trader Dashboard')

# SIDEBAR
iter = st.sidebar.selectbox("Iteration", iters)
dates = sorted(list(set([s.split('-')[3] for s in states if iter in s])), reverse=True)
dates.insert(0, (datetime.strptime(dates[0], "%Y%m%d%H") + timedelta(days=1)).strftime("%Y%m%d%H"))

if f"{iter}-Date" not in st.session_state:
    st.session_state[f"{iter}-Date"] = datetime.strptime(dates[0], "%Y%m%d%H")

col1, col2 = st.sidebar.columns(2)
with col1:
    date_dec = st.button('Day-1')
    if date_dec:
        st.session_state[f"{iter}-Date"] = st.session_state[f"{iter}-Date"] + timedelta(days=-1)
with col2:
    date_inc = st.button('Day+1')
    if date_inc:
        st.session_state[f"{iter}-Date"] = st.session_state[f"{iter}-Date"] + timedelta(days=1)

date = st.sidebar.date_input(
    label="Date", value=st.session_state[f"{iter}-Date"],
    min_value=datetime.strptime(dates[-1], "%Y%m%d%H"),
    max_value=datetime.strptime(dates[0], "%Y%m%d%H")
)
st.session_state[f"{iter}-Date"] = date

show_latest = False
state_name = [s for s in states if iter in s and st.session_state[f'{iter}-Date'].strftime("%Y%m%d") in s]
if not state_name:
    show_latest = True
    state_name = [s for s in states if iter in s and dates[1] in s]

state_name = state_name[0]

@st.cache_data
def get_state(name):
    return ppprovider.load_dict(name=name)


state = get_state(state_name)
if show_latest:
    state = state['state_future']

sy = st.selectbox("Symbol", state['state_global']['symbols'])
state_sy = state['state_symbol'][sy]

st.markdown("## General")
acc, pos, action, reward = st.columns(4)
with acc:
    st.markdown('#### Account')
    st.markdown(f"**Value:** ${state['state_global']['account']['value']}")
    st.markdown(f"**Available:** ${state['state_global']['account']['cash']}")

with pos:
    st.markdown("#### Position")
    if state_sy['position']:
        st.json(state_sy['position'])
    else:
        st.caption('No position.')

with action:
    st.markdown("### Action")
    st.markdown(f"**Action:** {state['action'][sy]['action_private']}")
    st.markdown(f"**Action Values:**")
    st.dataframe(
        pd.DataFrame((zip(['HOLD', 'BUY', 'SELL', 'CLOSE'], state['action'][sy]['predictions'])), columns=['Action', 'Value'])
    )

with reward:
    st.markdown("### Reward")
    if 'reward' in state:
        st.markdown(f"**Action Reward:** {round(state['reward'][sy]['v_action'], 3)}")
        st.markdown(f"**State Reward:** {round(state['reward'][sy]['v_curr'] - state['reward'][sy]['v_prev'], 3)}")
    else:
        st.caption('No reward for current date!')

st.markdown(f"## Activity")
st.caption(f"{'Showing Current Trade.' if state_sy['position'] else 'Showing Previous Trade, no current open trade.'}")

df = pd.DataFrame(state_sy['ohlcv'])
df_t = pd.DataFrame(state_sy['trade'])
df['ts'] = df['datetime'].apply(lambda dt: datetime.fromisoformat(dt) if isinstance(dt, str) else dt)
if len(df_t):
    df_t['ts'] = df_t['ts'].apply(lambda dt: datetime.fromisoformat(dt))

dt_anchor = (df_t.ts.min() if len(df_t) else df.ts.max()).replace(tzinfo=timezone.utc)
df = df[df.ts >= (dt_anchor - timedelta(days=28))]
fig = utils.plot(df)
if len(df_t):
    fig = utils.add_trade(fig, df_t)
st.plotly_chart(fig, use_container_width=True)

st.markdown("## Bridge Bands")
df = pd.DataFrame(state_sy['bridge_bnds']).iloc[-28*3:]
fig = utils.plot_bridge_bands(df)

st.plotly_chart(fig, use_container_width=True)


st.markdown("## Indicator Model")

df = pd.DataFrame(state_sy['ohlcv'])
df = df.iloc[-10:]
df = pd.concat([
    df.reset_index(drop=True),
    pd.DataFrame(state_sy['model_ind']['preds'])[['target', 'p']].reset_index(drop=True)
], axis=1)
fig = utils.plot(df)
fig = utils.add_model_ind(fig, df)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Model Params", expanded=False):
    st.json(state_sy['model_ind']['params'])
