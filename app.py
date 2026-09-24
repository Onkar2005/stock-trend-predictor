"""
app.py
------
Streamlit UI for the Stock Market Trend Predictor.

Run with:
    streamlit run app.py

Two ways to get a prediction:
  1. Manual Input  - type in OHLC + technical indicator values yourself.
  2. From CSV      - upload recent OHLCV data (needs at least ~35 rows so
                      rolling indicators like SMA_20 / RSI_14 can be computed)
                      and the app calculates the indicators for you and
                      predicts the trend for the most recent day.
"""

import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from feature_engineering import FEATURE_COLUMNS, TREND_LABELS, add_technical_indicators

try:
    from database.db_utils import fetch_history, log_prediction, test_connection
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False  # mysql-connector-python not installed - app still works without it

st.set_page_config(page_title="Stock Trend Predictor", page_icon="📈", layout="centered")

MODEL_PATH = "models/xgb_trend_model.pkl"
SCALER_PATH = "models/scaler.pkl"
META_PATH = "models/metadata.json"


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, scaler, meta


def predict_trend(model, scaler, feature_row: pd.DataFrame):
    X_scaled = scaler.transform(feature_row[FEATURE_COLUMNS])
    pred_class = int(model.predict(X_scaled)[0])
    probs = model.predict_proba(X_scaled)[0]
    return pred_class, probs


def render_result(pred_class, probs):
    label = TREND_LABELS[pred_class]
    emoji = {"Up": "🟢", "Down": "🔴", "Sideways": "🟡"}[label]
    confidence = probs[pred_class] * 100

    st.markdown("### Prediction")
    st.markdown(f"## {emoji} {label}  —  {confidence:.1f}% confidence")

    prob_df = pd.DataFrame(
        {
            "Trend": [TREND_LABELS[i] for i in sorted(TREND_LABELS)],
            "Probability (%)": [probs[i] * 100 for i in sorted(TREND_LABELS)],
        }
    ).set_index("Trend")
    st.bar_chart(prob_df)
    st.caption(
        "Probabilities are the model's confidence across all three classes "
        "and sum to 100%."
    )


st.title("📈 Stock Market Trend Predictor")
st.caption(
    "Predicts the likely stock trend over the next few trading days "
    "(Up / Sideways / Down) using an XGBoost model trained on OHLC data "
    "and technical indicators."
)

try:
    model, scaler, meta = load_artifacts()
except FileNotFoundError:
    st.error(
        "No trained model found in `models/`. Run `python train_model.py` "
        "first (see README.md), then reload this page."
    )
    st.stop()

with st.sidebar:
    st.header("Model info")
    st.metric("Test accuracy", f"{meta['test_accuracy'] * 100:.1f}%")
    st.write(f"Trained on {meta['n_train']} rows, tested on {meta['n_test']} rows")
    st.markdown("---")

    st.header("Database")
    db_connected = DB_AVAILABLE and test_connection()
    if db_connected:
        st.success("MySQL connected — predictions will be logged")
    elif DB_AVAILABLE:
        st.warning("mysql-connector-python installed but can't reach the DB "
                    "(check your .env / DB_* settings). Predictions won't be logged.")
    else:
        st.info("MySQL logging not set up. See database/README section in README.md "
                 "to enable it.")
    log_to_db = st.checkbox("Log predictions to MySQL", value=db_connected, disabled=not db_connected)

    st.markdown("---")
    st.caption(
        "⚠️ This is an educational project, not financial advice. The "
        "sample model is trained on synthetic data unless you retrained "
        "it on real market data."
    )

mode = st.radio("Input method", ["From CSV (auto-compute indicators)", "Manual input"], horizontal=False)

st.markdown("---")

if mode == "From CSV (auto-compute indicators)":
    st.subheader("Upload recent OHLCV data")
    st.write(
        "CSV needs columns `Date, Open, High, Low, Close, Volume`, sorted "
        "oldest to newest, with at least **35 rows** so indicators like "
        "SMA_20 / RSI_14 can be computed for the most recent day."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        raw = pd.read_csv(uploaded, parse_dates=["Date"])
        raw.columns = [c.strip().capitalize() if c.lower() != "date" else "Date" for c in raw.columns]

        if len(raw) < 35:
            st.warning("Please upload at least 35 rows of data for reliable indicators.")
        else:
            df_feat = add_technical_indicators(raw)
            latest = df_feat.dropna(subset=FEATURE_COLUMNS).tail(1)

            if latest.empty:
                st.warning("Not enough non-missing data to compute all indicators. Try a longer file.")
            else:
                st.write("Most recent day used for prediction:")
                st.dataframe(latest[["Date"] + FEATURE_COLUMNS].reset_index(drop=True))

                if st.button("Predict trend", type="primary"):
                    pred_class, probs = predict_trend(model, scaler, latest)
                    render_result(pred_class, probs)
                    if log_to_db:
                        log_prediction(
                            input_source="csv_upload",
                            feature_row=latest,
                            predicted_trend=TREND_LABELS[pred_class],
                            probs=probs,
                            trend_labels=TREND_LABELS,
                        )
                        st.caption("✅ Logged to MySQL (work_history)")

else:
    st.subheader("Enter today's values")
    col1, col2 = st.columns(2)
    with col1:
        open_ = st.number_input("Open", value=500.0, step=1.0)
        high_ = st.number_input("High", value=510.0, step=1.0)
        low_ = st.number_input("Low", value=495.0, step=1.0)
        close_ = st.number_input("Close", value=505.0, step=1.0)
        volume_ = st.number_input("Volume", value=1_000_000, step=1000)
        sma_10 = st.number_input("SMA_10", value=502.0, step=1.0)
        sma_20 = st.number_input("SMA_20", value=498.0, step=1.0)
        ema_12 = st.number_input("EMA_12", value=503.0, step=1.0)
        ema_26 = st.number_input("EMA_26", value=499.0, step=1.0)
    with col2:
        macd = st.number_input("MACD", value=2.0, step=0.1)
        macd_signal = st.number_input("MACD_Signal", value=1.5, step=0.1)
        rsi_14 = st.slider("RSI_14", min_value=0.0, max_value=100.0, value=55.0)
        stoch_k = st.slider("Stoch_%K", min_value=0.0, max_value=100.0, value=60.0)
        stoch_d = st.slider("Stoch_%D", min_value=0.0, max_value=100.0, value=58.0)
        momentum_10 = st.number_input("Momentum_10", value=5.0, step=0.5)
        bb_width = st.number_input("BB_Width", value=0.08, step=0.01, format="%.4f")
        atr_14 = st.number_input("ATR_14", value=12.0, step=0.5)
        ret_std_10 = st.number_input("Daily_Return_Std_10", value=0.012, step=0.001, format="%.4f")

    if st.button("Predict trend", type="primary"):
        feature_row = pd.DataFrame(
            [{
                "Open": open_, "High": high_, "Low": low_, "Close": close_, "Volume": volume_,
                "SMA_10": sma_10, "SMA_20": sma_20, "EMA_12": ema_12, "EMA_26": ema_26,
                "MACD": macd, "MACD_Signal": macd_signal, "RSI_14": rsi_14,
                "Stoch_%K": stoch_k, "Stoch_%D": stoch_d, "Momentum_10": momentum_10,
                "BB_Width": bb_width, "ATR_14": atr_14, "Daily_Return_Std_10": ret_std_10,
            }]
        )
        pred_class, probs = predict_trend(model, scaler, feature_row)
        render_result(pred_class, probs)
        if log_to_db:
            log_prediction(
                input_source="manual",
                feature_row=feature_row,
                predicted_trend=TREND_LABELS[pred_class],
                probs=probs,
                trend_labels=TREND_LABELS,
            )
            st.caption("✅ Logged to MySQL (work_history)")

st.markdown("---")
if DB_AVAILABLE and db_connected:
    with st.expander("📜 Work history (from MySQL)"):
        history_df = fetch_history(limit=50)
        if history_df.empty:
            st.write("No predictions logged yet.")
        else:
            st.dataframe(history_df, use_container_width=True)
