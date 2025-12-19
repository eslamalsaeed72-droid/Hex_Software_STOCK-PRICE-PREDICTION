import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# Load model and scaler (adjust path if in 'models/' folder)
@st.cache_resource
def load_artifacts():
    model = load_model('models/lstm_stock_model.h5')  # غير المسار لو الملف مش في models
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

st.title("Stock & Crypto Price Prediction using LSTM")
st.markdown("### Predict the next day's closing price using historical data from Yahoo Finance")

# Default to BTC-USD, user can change it
ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL, TSLA, BTC-USD)", value="BTC-USD").upper()

days = st.slider("Number of historical days for context", 365, 2000, 1000)

if st.button("Predict Next Day Closing Price"):
    with st.spinner("Fetching data and generating prediction..."):
        try:
            # Download recent closing prices
            data = yf.download(ticker, period=f"{days}d")['Close']
            
            if data.empty or len(data) < 60:
                st.error("No data available for this ticker or insufficient history (need at least 60 days).")
            else:
                # Prepare last 60 days for prediction
                scaled_data = scaler.transform(data.values.reshape(-1, 1))
                last_60 = scaled_data[-60:].reshape(1, 60, 1)
                
                # Predict
                pred_scaled = model.predict(last_60, verbose=0)
                prediction = scaler.inverse_transform(pred_scaled)[0][0]
                
                # Get latest price (convert to Python float to avoid formatting issues)
                latest_price = float(data.iloc[-1])
                prediction = float(prediction)
                
                # Display results
                st.success(f"Predicted Next Day Close Price for {ticker}: **${prediction:.2f}**")
                st.write(f"Latest Close Price: ${latest_price:.2f}")
                
                change_pct = ((prediction - latest_price) / latest_price) * 100
                st.write(f"Expected Change: **{change_pct:+.2f}%**")
                
                # Plot historical data with prediction
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(data.index, data.values, label='Historical Close Price', color='blue')
                ax.axvline(data.index[-1], color='gray', linestyle='--', label='Today')
                ax.scatter(data.index[-1], latest_price, color='blue', s=100, label='Latest Price')
                ax.scatter(pd.Timestamp.today(), prediction, color='red', s=100, label='Predicted Next Day')
                ax.set_title(f"{ticker} Price History & Next Day Prediction")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (USD)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
        except Exception as e:
            st.error("An error occurred while processing the request. Please check the ticker symbol.")
            st.info("Common issues: Invalid ticker, no internet/data, or market closed.")
