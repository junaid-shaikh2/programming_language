from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import settings
from src.data_sources import APIError, AlphaVantageClient
from src.model import load_model
from src.service import run_training_pipeline


st.set_page_config(
    page_title="Forex Direction Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("ML-Based Forex Direction Prediction")
st.caption("Train a machine learning model on forex price history and predict the next-day direction.")


with st.sidebar:
    st.header("Configuration")
    base_currency = st.selectbox("Base currency", ["EUR", "GBP", "USD", "AUD", "JPY"], index=0)
    quote_currency = st.selectbox("Quote currency", ["USD", "JPY", "CHF", "CAD", "EUR"], index=0)
    train_clicked = st.button("Train and Predict", type="primary", use_container_width=True)

    st.markdown("### API status")
    st.write(f"Alpha Vantage key: {'Configured' if settings.alpha_vantage_api_key else 'Missing'}")
    st.write(f"NewsAPI key: {'Configured' if settings.newsapi_key else 'Optional / Missing'}")


def render_price_chart(price_df: pd.DataFrame, pair: str) -> None:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=price_df.index,
                open=price_df["open"],
                high=price_df["high"],
                low=price_df["low"],
                close=price_df["close"],
                name=pair,
            )
        ]
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_backtest_chart(signals: list[dict[str, object]]) -> None:
    if not signals:
        return

    backtest_df = pd.DataFrame(signals)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=backtest_df["date"],
            y=backtest_df["strategy_return_pct"],
            name="Strategy Return %",
            marker_color="#0f766e",
        )
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)


if train_clicked:
    try:
        with st.spinner("Fetching market data, training the model, and generating a prediction..."):
            result = run_training_pipeline(base_currency, quote_currency)

        pair = result["pair"]
        prediction = result["training"]["latest_prediction"]
        context = result["market_context"]
        backtest = result["training"]["backtest"]

        st.success(f"Model trained successfully for {pair}")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Predicted direction", prediction["prediction"])
        col2.metric("Probability UP", f"{prediction['probability_up']:.2%}")
        col3.metric("Model accuracy", f"{result['training']['accuracy']:.2%}")
        col4.metric("Precision", f"{result['training']['precision']:.2%}")
        col5.metric("Recall", f"{result['training']['recall']:.2%}")

        overview_tab, backtest_tab, market_tab, raw_tab = st.tabs(
            ["Overview", "Backtest", "Market Context", "Raw Payload"]
        )

        with overview_tab:
            price_df = AlphaVantageClient(settings.alpha_vantage_api_key).get_fx_daily(base_currency, quote_currency)
            st.subheader("Recent Price Action")
            render_price_chart(price_df.tail(90), pair)
            st.write(
                f"Model trained on {result['training']['rows_used']} feature rows. "
                f"Prediction date reference: {prediction['prediction_for_date']}."
            )

        with backtest_tab:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Strategy Return", f"{backtest['strategy_return_pct']:.2f}%")
            col_b.metric("Buy & Hold Return", f"{backtest['buy_and_hold_return_pct']:.2f}%")
            col_c.metric("Winning Signal Rate", f"{backtest['winning_signal_rate']:.2f}%")
            st.write(
                f"Test window: {backtest['test_period_start']} to {backtest['test_period_end']}"
            )
            render_backtest_chart(backtest["signals"])
            if backtest["signals"]:
                st.dataframe(pd.DataFrame(backtest["signals"]), use_container_width=True)
            st.write("Confusion matrix `[DOWN, UP]` by predicted class:")
            st.write(result["training"]["confusion_matrix"])

        with market_tab:
            st.subheader("Market Context")
            st.write(f"Sentiment score: {context['sentiment_score']}")
            st.write(f"Recent headline count: {context['headline_count']}")
            if context["headlines"]:
                st.write("Recent headlines")
                for headline in context["headlines"]:
                    st.write(f"- {headline}")

        with raw_tab:
            st.code(json.dumps(result, indent=2), language="json")

    except APIError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.exception(exc)
else:
    st.info("Choose a forex pair from the sidebar and click 'Train and Predict' to begin.")

    try:
        saved = load_model()
        st.subheader("Last Saved Model Snapshot")
        st.json(saved["metadata"])
    except FileNotFoundError:
        st.write("No saved model yet. Train the model to generate predictions.")
