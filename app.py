from flask import Flask, jsonify
import ccxt
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import pytz

from ta.trend import ema_indicator, macd_diff        # type: ignore
from ta.momentum import rsi                          # type: ignore
from ta.volatility import average_true_range         # type: ignore
from ta.volume import on_balance_volume              # type: ignore

# ========== Constants ==========
MODEL_PATH = "new_model.joblib"
TIMEFRAME = "5m"
SYMBOL = "BTC/USDT"
FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "ret_lag1", "ret_lag2", "ret_lag3",
    "volatility_5", "volatility_15",
    "price_vs_ema9", "price_vs_ema21",
    "rsi", "macd", "atr", "obv",
    "candle_body", "candle_range",
    "hour", "dayofweek"
]

india_tz = pytz.timezone("Asia/Kolkata")
model = joblib.load(MODEL_PATH)

exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# ========== Feature Engineering ==========
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ret_1"]         = df["Close"].pct_change()
    df["ret_3"]         = df["Close"].pct_change(3)
    df["ret_6"]         = df["Close"].pct_change(6)
    df["ret_12"]        = df["Close"].pct_change(12)
    df["ret_lag1"]      = df["ret_1"].shift(1)
    df["ret_lag2"]      = df["ret_1"].shift(2)
    df["ret_lag3"]      = df["ret_1"].shift(3)
    df["volatility_5"]  = df["Close"].rolling(5).std()
    df["volatility_15"] = df["Close"].rolling(15).std()

    df["ema_9"]         = ema_indicator(df["Close"], window=9)
    df["ema_21"]        = ema_indicator(df["Close"], window=21)
    df["price_vs_ema9"] = df["Close"] / df["ema_9"] - 1
    df["price_vs_ema21"]= df["Close"] / df["ema_21"] - 1

    df["rsi"]           = rsi(df["Close"], window=14)
    df["macd"]          = macd_diff(df["Close"])
    df["atr"]           = average_true_range(df["High"], df["Low"], df["Close"])
    df["obv"]           = on_balance_volume(df["Close"], df["Volume"])

    df["candle_body"]   = df["Close"] - df["Open"]
    df["candle_range"]  = df["High"] - df["Low"]

    df["Datetime"]      = pd.to_datetime(df["Datetime"])
    df["hour"]          = df["Datetime"].dt.hour
    df["dayofweek"]     = df["Datetime"].dt.dayofweek

    return df

# ========== Fetch OHLCV ==========
def fetch_recent_data(symbol=SYMBOL, timeframe=TIMEFRAME, limit=100):
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(candles, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="ms")
    return df

# ========== Make Prediction ==========
def make_live_prediction():
    df = fetch_recent_data()
    df = compute_features(df).dropna()
    X_latest = df[FEATURE_COLUMNS].iloc[[-1]]
    prediction = int(model.predict(X_latest)[0])
    confidence = float(model.predict_proba(X_latest)[0][prediction])
    signal = "BUY 🟢" if prediction == 1 else "SELL 🔴"
    timestamp = datetime.now(india_tz).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "signal": signal,
        "confidence": round(confidence, 2),
        "timestamp": timestamp,
        "current_price": float(df["Close"].iloc[-1]),
        "timeframe": TIMEFRAME
    }

# ========== Flask App ==========
app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    try:
        result = make_live_prediction()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== Run ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
