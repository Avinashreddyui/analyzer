import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from datetime import datetime
import threading

# ✅ Load Sentiment Analysis Model
sentiment_analyzer = pipeline("sentiment-analysis")

# ✅ Function to Fetch News Headlines and Analyze Sentiment
@st.cache_data
def get_sentiment_score(ticker):
    search_url = f"https://www.google.com/search?q={ticker}+stock+news&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    headlines = [item.get_text() for item in soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")][:5]
    
    if not headlines:
        return 0  # Neutral sentiment if no headlines
    
    scores = []
    
    def analyze_sentiment(headline):
        try:
            result = sentiment_analyzer(headline)[0]
            sentiment = result['label']
            score = result['score']
            scores.append(score if sentiment == "POSITIVE" else -score if sentiment == "NEGATIVE" else 0)
        except Exception as e:
            print(f"Sentiment Analysis Error: {e}")
            return 0
    
    threads = [threading.Thread(target=analyze_sentiment, args=(headline,)) for headline in headlines]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    return np.mean(scores) if scores else 0

# ✅ Fetch NIFTY 50 Stock List
@st.cache_data
def get_nifty50_stocks():
    nifty50_tickers = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "BHARTIARTL.NS",
        "ITC.NS", "ASIANPAINT.NS", "HCLTECH.NS", "LT.NS", "MARUTI.NS", "AXISBANK.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
        "M&M.NS", "ONGC.NS", "POWERGRID.NS", "TATASTEEL.NS", "TECHM.NS", "HDFCLIFE.NS", "COALINDIA.NS", "INDUSINDBK.NS", "JSWSTEEL.NS", "NTPC.NS",
        "NESTLEIND.NS", "GRASIM.NS", "BPCL.NS", "HINDALCO.NS", "CIPLA.NS", "ADANIPORTS.NS", "DRREDDY.NS", "DIVISLAB.NS", "BRITANNIA.NS", "HEROMOTOCO.NS",
        "EICHERMOT.NS", "APOLLOHOSP.NS", "BAJAJFINSV.NS", "TATAMOTORS.NS", "SBILIFE.NS", "UPL.NS", "DLF.NS", "ICICIPRULI.NS", "PIDILITIND.NS", "GAIL.NS"
    ]
    return sorted(nifty50_tickers)

# ✅ Streamlit UI
st.title("Fast Stock Price Prediction with and without Sentiment Analysis")

# User input for stock ticker
ticker = st.selectbox("Select a NIFTY 50 Stock", get_nifty50_stocks(), index=0)

# Date input fields
start_date = st.date_input("Select Start Date", datetime(2020, 1, 1))
end_date = st.date_input("Select End Date", datetime(2025, 2, 1))

prediction_date = st.date_input("Select Prediction Date", datetime(2025, 2, 2))

if st.button("Predict Stock Price"):
    with st.spinner("Fetching stock data..."):
        df = yf.download(ticker, start=start_date, end=end_date)

        if df.empty:
            st.error("No data available for this stock. Try another ticker.")
            st.stop()

        # Save downloaded data as CSV
        df.to_csv("downloaded_stock_data.csv")
        st.success("Stock data downloaded successfully!")
        
        # Process and clean data
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])

        df['sentiment_score'] = get_sentiment_score(ticker) if prediction_date else 0

        # Display sentiment analysis
        st.subheader("Sentiment Analysis")
        sentiment_value = df['sentiment_score'].iloc[0]
        if sentiment_value > 0.05:
            st.write("Overall Sentiment: **Positive**")
        elif sentiment_value < -0.05:
            st.write("Overall Sentiment: **Negative**")
        else:
            st.write("Overall Sentiment: **Neutral**")

        # Train XGBoost model
        features = ['Open', 'High', 'Low', 'Close', 'sentiment_score']
        X = df[features]
        y = df[['Open', 'High', 'Low', 'Close']]
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        @st.cache_resource
        def train_xgboost(X_train, y_train):
            model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            return model

        xgb_model = train_xgboost(X_scaled[:-1], y_scaled[:-1])
        next_day_features = X_scaled[-1].reshape(1, -1)
        predicted_next_day_ohlc = xgb_model.predict(next_day_features)
        predicted_next_day_ohlc = scaler_y.inverse_transform(predicted_next_day_ohlc.reshape(1, -1))

        st.subheader(f"Predicted OHLC for {prediction_date.strftime('%Y-%m-%d')}")
        st.write(f"**Open:** {predicted_next_day_ohlc[0, 0]:.2f}")
        st.write(f"**High:** {predicted_next_day_ohlc[0, 1]:.2f}")
        st.write(f"**Low:** {predicted_next_day_ohlc[0, 2]:.2f}")
        st.write(f"**Close:** {predicted_next_day_ohlc[0, 3]:.2f}")

st.write("Developed by [Your Name]")
