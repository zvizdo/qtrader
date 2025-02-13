import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from datetime import datetime
from google.cloud import storage
import numpy as np

@st.cache_resource(show_spinner="Loading data...")
def download_db():
    gcs = storage.Client()
    bucket = gcs.bucket('as-dev-anze-qtrader')
    blob = bucket.blob('db.sqlite')
    blob.download_to_filename('db.sqlite')


def plot(df, with_volume=True):
    # Create figure with secondary y-axis
    df['ts'] = df['datetime'].apply(lambda dt: datetime.fromisoformat(dt) if isinstance(dt, str) else dt)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0
        # specs=[[{"secondary_y": True}]]
    )

    # include candlestick with rangeselector
    fig.add_trace(go.Candlestick(
        x=df['ts'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Symbol'
    ),
        row=1, col=1
    )

    if with_volume:
        # include a go.Bar trace for volumes
        fig.add_trace(go.Bar(
                x=df['ts'],
                y=df['volume'],
                opacity=0.5,
                marker_color=list(df.apply(lambda r: 'rgb(0, 255, 0)' if r['close'] >= r['open'] else 'rgb(255, 0, 0)', axis=1)), #'rgb(0, 0, 255)'
                name='Volume'
            ),
            row=2, col=1
        )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        showlegend=False,
        autosize=True,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0
        ),
        # paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='#fff',  # 'rgba(0,0,0,0)',
        # height=224 * 4,
    )

    return fig


def plot_bridge_bands(df, with_volume=True):
    fig = plot(df, with_volume)
    fig.add_trace(go.Scatter(
        x=df['ts'],
        y=df['bridge_bands_upper'],
        opacity=0.5,
        mode='lines',
        name='Upper BB'
    ),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(
        x=df['ts'],
        y=df['bridge_bands_lower'],
        opacity=0.5,
        mode='lines',
        name='Lower BB'
    ),
        row=1, col=1
    )

    return fig


def plot_trendlines(df, with_volume=True):
    fig = plot(df, with_volume)
    for i in range(4):
        fig.add_trace(go.Scatter(
            x=df['ts'],
            y=df[f'line_{i}'],
            opacity=0.75,
            mode='lines',
            name=f'SR{i + 1}',
            line=dict(color="Blue")
        ),
            row=1, col=1
        )

    return fig


def add_pivot_points(fig, df):
    df['ts'] = df['datetime'].apply(lambda dt: datetime.fromisoformat(dt))

    fig.add_trace(go.Scatter(
        x=df[df.type == "MAX"]['ts'],
        y=df[df.type == "MAX"]['close'],
        opacity=1,
        mode='markers',
        marker=dict(
            color='Green',
            size=6,
            line=dict(color='Black', width=2)
        ),
        name='MAX'
    ),
        row=1, col=1
    )

    fig.add_trace(go.Scatter(
        x=df[df.type == "MIN"]['ts'],
        y=df[df.type == "MIN"]['close'],
        opacity=1,
        mode='markers',
        marker=dict(
            color='Red',
            size=6,
            line=dict(color='Black', width=2)
        ),
        name='MIN'
    ),
        row=1, col=1
    )

    return fig


def add_model_ind(fig, df):
    y = np.zeros(len(df))
    y[df.p.values >= 0.5] = df[df.p >= 0.5].low.values * 0.975
    y[df.p.values < 0.5] = df[df.p < 0.5].high.values * 1.025

    fig.add_trace(go.Scatter(
        x=df['ts'],
        y=y,
        mode="markers+text",
        name="Predictions",
        text=df.apply(lambda r: f"P: {round(r.p, 2)}: T: {r.target}", axis=1),
        marker_symbol=['triangle-up' if p >= 0.5 else 'triangle-down' for p in df.p.values],
        marker_color=['Green' if p >= 0.5 else 'Red' for p in df.p.values],
        marker_size=15,
        textposition=["bottom center" if p >= 0.5 else "top center" for p in df.p.values]
    ))

    return fig


def add_trade(fig, df):
    y = np.zeros(len(df))
    y[df.instruction.values == "BUY"] = df[df.instruction == "BUY"].price.values * 0.975
    y[df.instruction.values == "SELL"] = df[df.instruction == "SELL"].price.values * 1.025

    fig.add_trace(go.Scatter(
        x=df['ts'],
        y=y,
        mode="markers+text",
        name="Trade",
        text=df.apply(lambda r: f"{r.instruction} / ${round(r['size'] * r['price'], 2)}", axis=1),
        marker_symbol=['triangle-up' if ins == "BUY" else 'triangle-down' for ins in df.instruction.values],
        marker_color=['Green' if ins == "BUY" else 'Red' for ins in df.instruction.values],
        marker_size=15,
        textposition=["bottom center" if ins == "BUY" else "top center" for ins in df.instruction.values]
    ))

    fig.add_trace(go.Scatter(
        x=df['ts'],
        y=df.price,
        mode="markers+text",
        name="Trade",
        # text=df.apply(lambda r: f"{r.instruction} / ${round(r['size'] * r['price'], 2)}", axis=1),
        marker_symbol=['diamond-open-dot' if ins == "BUY" else 'diamond-open-dot' for ins in df.instruction.values],
        marker_color=['Blue' if ins == "BUY" else 'Red' for ins in df.instruction.values],
        marker_size=15,
        # textposition=["bottom center" if ins == "BUY" else "top center" for ins in df.instruction.values]
    ))

    return fig
