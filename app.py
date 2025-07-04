# ==========================================================
# Flask API for Real-Time BTCUSDT Prediction
# ==========================================================
from flask import Flask, jsonify
import pandas as pd
import ccxt
import ta
import joblib
import pytz
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# Feature engineering (must match training)
# ==========================================================
def compute_features(df):
    df["ret_1"] = df["Close"].pct_change()
    df["ret_3"] = df["Close"].pct_change(3)
    df["ret_6"] = df["Close"].pct_change(6)
    df["ret_12"] = df["Close"].pct_change(12)

    df["ret_lag1"] = df["ret_1"].shift(1)
    df["ret_lag2"] = df["ret_1"].shift(2)
    df["ret_lag3"] = df["ret_1"].shift(3)
    df["ret_lag6"] = df["ret_1"].shift(6)

    df["mean_close_5"] = df["Close"].rolling(window=5).mean()
    df["volatility_5"] = df["Close"].rolling(window=5).std()
    df["mean_close_15"] = df["Close"].rolling(window=15).mean()
    df["volatility_15"] = df["Close"].rolling(window=15).std()

    df["ema_9"] = ta.trend.ema_indicator(df["Close"], window=9)
    df["ema_21"] = ta.trend.ema_indicator(df["Close"], window=21)
    df["price_vs_ema9"] = df["Close"] / df["ema_9"] - 1
    df["price_vs_ema21"] = df["Close"] / df["ema_21"] - 1

    df["rsi"] = ta.momentum.rsi(df["Close"], window=14)
    df["macd"] = ta.trend.macd_diff(df["Close"])
    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])
    df["obv"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])

    df["candle_body"] = df["Close"] - df["Open"]
    df["candle_range"] = df["High"] - df["Low"]
    df["upper_wick"] = df["High"] - df[["Close", "Open"]].max(axis=1)
    df["lower_wick"] = df[["Close", "Open"]].min(axis=1) - df["Low"]

    df["minute"] = df["Datetime"].dt.minute
    df["hour"] = df["Datetime"].dt.hour
    df["dayofweek"] = df["Datetime"].dt.dayofweek

    return df

# ==========================================================
# Features matching the training model
# ==========================================================
features = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag6",
    "mean_close_5", "volatility_5",
    "mean_close_15", "volatility_15",
    "ema_9", "ema_21", "price_vs_ema9", "price_vs_ema21",
    "rsi", "macd", "atr", "obv",
    "candle_body", "candle_range", "upper_wick", "lower_wick",
    "minute", "hour", "dayofweek"
]

# ==========================================================
# Load trained model and scaler
# ==========================================================
model_data = joblib.load("btc_model_xgb_confident_v1.joblib")
model = model_data["model"]
scaler = model_data["scaler"]
print("✅ Model and scaler loaded successfully!")

# Binance exchange
exchange = ccxt.binance()
symbol = 'BTC/USDT'

# Indian timezone
india_tz = pytz.timezone("Asia/Kolkata")

# ==========================================================
# Create Flask app
# ==========================================================
app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Fetch last 100 candles (~500 min)
        since = exchange.milliseconds() - 100 * 5 * 60 * 1000
        candles = exchange.fetch_ohlcv(symbol, timeframe='5m', since=since, limit=100)
        df = pd.DataFrame(candles, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit="ms")

        # Compute features
        df = compute_features(df)
        df = df.dropna().reset_index(drop=True)

        if len(df) < 20:
            return jsonify({"error": "Not enough candles to compute all features."}), 400

        # Take the last row
        X_live = df[features].iloc[-1:]
        X_scaled = scaler.transform(X_live)

        # Predict probabilities
        probas = model.predict_proba(X_scaled)[0]
        y_pred_class = int(probas.argmax())
        confidence = float(probas[y_pred_class])
        signal = "BUY" if y_pred_class == 1 else "SELL"

        dt_utc = df["Datetime"].iloc[-1]
        dt_india = dt_utc.tz_localize("UTC").astimezone(india_tz)
        dt_str = dt_india.strftime("%Y-%m-%d %H:%M:%S")

        # Return as JSON
        return jsonify({
            "time_india": dt_str,
            "signal": signal,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================================
# Run app
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
