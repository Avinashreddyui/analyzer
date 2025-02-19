import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import streamlit as st
import tensorflow as tf
import joblib
import os
import logging
import asyncio
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)

# Telegram Bot Credentials
TELEGRAM_BOT_TOKEN = "7861934178:AAElJz-VhGupuW72gqfhJPF_0Qbx-O5k5LU"
TELEGRAM_CHAT_ID = "1280751310"

def send_telegram_message(message):
    """Sends a message to Telegram."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, json=payload)
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")

def fetch_stock_data(ticker, period="30d", interval="5m"):
    """Fetches stock data from Yahoo Finance."""
    try:
        stock = yf.download(ticker, period=period, interval=interval)
        stock.dropna(inplace=True)
        return stock
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

def prepare_lstm_data(data):
    """Prepares LSTM data."""
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

def build_lstm_model():
    """Builds an LSTM model."""
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(60,1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(X_train, y_train, ticker, scaler):
    """Trains and saves the LSTM model."""
    model = build_lstm_model()
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)
    os.makedirs('trained', exist_ok=True)
    model.save(f"trained/{ticker}_model.keras")
    joblib.dump(scaler, f"trained/{ticker}_scaler.pkl")
    return model

def load_trained_model(ticker):
    """Loads a trained model if available."""
    try:
        if os.path.exists(f"trained/{ticker}_model.keras") and os.path.exists(f"trained/{ticker}_scaler.pkl"):
            model = load_model(f"trained/{ticker}_model.keras")
            scaler = joblib.load(f"trained/{ticker}_scaler.pkl")
            return model, scaler
    except Exception as e:
        logging.error(f"Error loading model for {ticker}: {e}")
    return None, None

def lstm_prediction(model, data, scaler):
    """Generates a stock price prediction."""
    last_60_days = data['Close'].values[-60:].reshape(-1,1)
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = np.array([last_60_days_scaled])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_price = model.predict(X_test)
    return scaler.inverse_transform(predicted_price)[0,0]

async def real_time_predictions(tickers, model_dict, scaler_dict):
    """Runs real-time stock price predictions asynchronously."""
    while True:
        predictions = []
        for ticker in tickers:
            stock_data = fetch_stock_data(ticker, period="7d", interval="1m")
            if stock_data is not None and len(stock_data) > 60:
                predicted_price = lstm_prediction(model_dict[ticker], stock_data, scaler_dict[ticker])
                last_close = stock_data['Close'].iloc[-1]
                recommendation = "BUY" if predicted_price > last_close else "SELL" if predicted_price < last_close else "HOLD"
                message = f"{ticker} - Predicted Price: {predicted_price}, Last Close: {last_close}, Recommendation: {recommendation}"
                send_telegram_message(message)
                predictions.append({"Ticker": ticker, "Recommendation": recommendation, "Price": predicted_price})
        await asyncio.sleep(60)

def main():
    """Streamlit UI for user interaction."""
    st.title("AI Stock Prediction Bot")
    tickers = st.text_input("Enter stock tickers (comma separated):", "TSLA,AAPL,NFLX,AMZN,LLY").split(',')
    interval = st.selectbox("Select time interval:", ["1m", "5m", "15m", "1h", "1d"])
    model_dict, scaler_dict = {}, {}
    
    st.write("Checking for pre-trained models...")
    for ticker in tickers:
        model, scaler = load_trained_model(ticker)
        if model is None or scaler is None:
            st.warning(f"Training model for {ticker}...")
            stock_data = fetch_stock_data(ticker, period="30d", interval=interval)
            if stock_data is not None:
                X, y, scaler = prepare_lstm_data(stock_data)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                model = train_lstm_model(X_train, y_train, ticker, scaler)
                model_dict[ticker] = model
                scaler_dict[ticker] = scaler
    
    st.write("AI models are ready!")
    if st.button("Start Real-Time Predictions"):
        asyncio.run(real_time_predictions(tickers, model_dict, scaler_dict))

if __name__ == "__main__":
    main()
