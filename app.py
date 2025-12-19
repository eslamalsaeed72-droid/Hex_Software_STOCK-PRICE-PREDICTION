import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# Load the model and scaler
@st.cache_resource
def load_artifacts():
    model = load_model('models/lstm_stock_model.h5')  
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

st.title("Stock Price Prediction using LSTM")
st.markdown("### Enter a stock ticker to predict the next day's closing price")

ticker = st.text_input("Stock Ticker (e.g., AAPL, TSLA, MSFT)", value="AAPL").upper()
days = st.slider("Number of historical days to display", 365, 2000, 1000)

if st.button("Predict Next Day Price"):
    with st.spinner("Fetching data and predicting..."):
        # Fetch data
        data = yf.download(ticker, period=f"{days}d")['Close']
        
        if data.empty:
            st.error("Invalid ticker or no data available!")
        else:
            # Prepare last 60 days
            scaled_data = scaler.transform(data.values.reshape(-1, 1))
            last_60 = scaled_data[-60:].reshape(1, 60, 1)
            
            # Make prediction
            pred_scaled = model.predict(last_60)
            prediction = scaler.inverse_transform(pred_scaled)[0][0]
            
            latest_price = data.iloc[-1]
            
            st.success(f"Predicted Next Day Close Price for {ticker}: **${prediction:.2f}**")
            st.write(f"Latest Close Price: ${latest_price:.2f}")
            st.write(f"Expected Change: **{((prediction - latest_price)/latest_price*100):.2f}%**")
            
            # Plot historical data
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data.values, label='Historical Close')
            ax.axvline(data.index[-1], color='gray', linestyle='--')
            ax.scatter([data.index[-1]], [latest_price], color='blue', s=100, label='Latest')
            ax.scatter([datetime.today()], [prediction], color='red', s=100, label='Predicted Next Day')
            ax.set_title(f"{ticker} Stock Price + Next Day Prediction")
            ax.legend()
            st.pyplot(fig)
