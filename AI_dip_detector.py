import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import streamlit as st
import tensorflow as tf
import joblib
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ta  # For technical analysis indicators

# Telegram Bot Credentials
TELEGRAM_BOT_TOKEN = "7861934178:AAElJz-VhGupuW72gqfhJPF_0Qbx-O5k5LU"
TELEGRAM_CHAT_ID = "1280751310"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload)

def fetch_stock_data(ticker, period="30d", interval="5m", retries=5):
    attempt = 0
    while attempt < retries:
        try:
            stock = yf.download(ticker, period=period, interval=interval)
            if stock.empty:
                st.error(f"No data available for {ticker}. Skipping...")
                return None
            return stock
        except yf.utils.exceptions.YFRateLimitError:
            wait_time = 2 ** attempt  # Exponential backoff
            st.warning(f"Rate limit exceeded for {ticker}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            attempt += 1
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            return None
    st.error(f"Failed to fetch data for {ticker} after multiple retries.")
    return None

def prepare_lstm_data(data):
    if data is None or data.empty:
        return None, None, None
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

def train_lstm_model(X_train, y_train, ticker, scaler):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(60,1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)
    os.makedirs('trained', exist_ok=True)
    model.save(f"trained/{ticker}_model.keras")
    joblib.dump(scaler, f"trained/{ticker}_scaler.pkl")
    return model

def lstm_prediction(model, data, scaler):
    if model is None or data is None or scaler is None:
        return None
    
    last_60_days = data['Close'].values[-60:].reshape(-1,1)
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = np.array([last_60_days_scaled])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_price = model.predict(X_test)
    return scaler.inverse_transform(predicted_price)[0,0]

def detect_dip(data):
    if data is None or data.empty:
        return False
    
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    latest_rsi = data['RSI'].iloc[-1]
    latest_price = data['Close'].iloc[-1]
    latest_sma_50 = data['SMA_50'].iloc[-1]
    latest_sma_200 = data['SMA_200'].iloc[-1]
    
    if latest_rsi < 30 and latest_price < latest_sma_50 and latest_price < latest_sma_200:
        return True
    return False

def load_trained_model(ticker):
    model_path = f"trained/{ticker}_model.keras"
    scaler_path = f"trained/{ticker}_scaler.pkl"
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

def real_time_predictions(tickers, model_dict, scaler_dict):
    predictions = []
    for ticker in tickers:
        stock_data = fetch_stock_data(ticker, period="7d", interval="1m")
        if stock_data is None:
            continue
        predicted_price = lstm_prediction(model_dict.get(ticker), stock_data, scaler_dict.get(ticker))
        if predicted_price is not None:
            last_close = stock_data['Close'].iloc[-1].item()
            recommendation = "BUY" if predicted_price > last_close else "SELL" if predicted_price < last_close else "HOLD"
            message = f"{ticker} - Predicted Price: {predicted_price}, Last Close: {last_close}, Recommendation: {recommendation}"
            send_telegram_message(message)
            predictions.append({"Ticker": ticker, "Recommendation": recommendation, "Price": predicted_price})
    return predictions

def main():
    st.title("AI Stock Prediction Bot - Day Trading Mode")
    tickers_input = st.text_input("Enter stock tickers separated by commas (e.g., AAPL,MSFT,GOOGL):")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
    
    if not tickers:
        st.warning("Please enter at least one ticker to proceed.")
        return
    
    model_dict, scaler_dict = {}, {}
    st.write("Checking for pre-trained models...")
    
    for ticker in tickers:
        model, scaler = load_trained_model(ticker)
        if model is not None and scaler is not None:
            st.success(f"Loaded pre-trained model for {ticker}")
            model_dict[ticker] = model
            scaler_dict[ticker] = scaler
    
    st.write("AI models are ready!")
    if st.button("Start Real-Time Predictions"):
        predictions = real_time_predictions(tickers, model_dict, scaler_dict)
        df = pd.DataFrame(predictions)
        st.write(df)

if __name__ == "__main__":
    main()
