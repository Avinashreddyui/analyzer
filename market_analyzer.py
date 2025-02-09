import requests
import pandas as pd
from textblob import TextBlob
from bs4 import BeautifulSoup
from googlesearch import search
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import json

# Function to fetch Google search results
def get_google_news(query, num_results=5):
    search_results = search(query + " news", num_results=num_results)
    return list(search_results)

# Function to fetch Yahoo Finance news
def get_yahoo_news(ticker):
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}&newsCount=5"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        news_data = response.json()
        articles = [article['link'] for article in news_data.get('news', [])]
        return articles
    except Exception as e:
        return []

# Function to fetch economic data from FRED API
def get_economic_data(series_id):
    api_key = "cca80011714bb0c446df5ccae205964c"  # Replace with your FRED API key
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
    try:
        response = requests.get(url)
        data = response.json()
        observations = data.get("observations", [])
        return pd.DataFrame(observations)
    except Exception as e:
        return pd.DataFrame()

# Function to extract article content
def extract_article_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.text for p in paragraphs])
    except Exception as e:
        return ""

# Sentiment Analysis Function
def analyze_sentiment(text):
    sentiment_score = TextBlob(text).sentiment.polarity
    if sentiment_score <= -0.5:
        return "Negative"
    elif -0.5 < sentiment_score < -0.1:
        return "Weak Negative"
    elif -0.1 <= sentiment_score <= 0.1:
        return "Neutral"
    elif 0.1 < sentiment_score < 0.5:
        return "Weak Positive"
    else:
        return "Positive"

# Main function
def analyze_market_news(keyword):
    news_urls = get_google_news(keyword) + get_yahoo_news(keyword)
    data = []
    for url in news_urls:
        content = extract_article_content(url)
        if content:
            sentiment = analyze_sentiment(content)
            data.append([url, sentiment])
    df = pd.DataFrame(data, columns=['URL', 'Sentiment'])
    return df

# Streamlit Dashboard
def main():
    st.title("Market News & Economic Data Sentiment Analysis")
    keyword = st.text_input("Enter keyword (e.g., economic news, AAPL stock):", "economic news")
    economic_indicator = st.selectbox("Select Economic Indicator:", ["GDP", "CPI", "Unemployment Rate"])
    
    if st.button("Analyze"):
        df = analyze_market_news(keyword)
        st.write(df)
        
        # Plot Sentiment Analysis
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
        
        # Fetch and Display Economic Data
        indicator_mapping = {"GDP": "GDP", "CPI": "CPIAUCSL", "Unemployment Rate": "UNRATE"}
        econ_data = get_economic_data(indicator_mapping[economic_indicator])
        if not econ_data.empty:
            st.subheader(f"Economic Data: {economic_indicator}")
            st.line_chart(econ_data.set_index("date")['value'].astype(float))
        else:
            st.write("No economic data available.")

if __name__ == "__main__":
    main()
