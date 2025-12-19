# 🔮 Stock Price Prediction using LSTM

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_LINK_IF_DEPLOYED)  
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)  
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📌 Project Overview

This project implements a **Long Short-Term Memory (LSTM)** neural network to predict stock closing prices. It uses historical data from **Yahoo Finance** via `yfinance`, preprocesses the data, trains a deep learning model, evaluates performance, and provides an interactive **Streamlit web application** for real-time next-day price predictions.

Perfect for learning time series forecasting, deep learning, and building end-to-end ML projects.

### Key Features
- Real-time stock data fetching (AAPL by default, supports any ticker like TSLA, MSFT, GOOGL, BTC-USD)
- Data preprocessing with Min-Max scaling
- Multi-layer LSTM model with Dropout to prevent overfitting
- Early Stopping and model evaluation using RMSE
- Interactive Streamlit app for user input and visualization
- Next-day closing price prediction with percentage change estimate

## 📊 Model Performance
- Trained on ~5 years of historical data
- RMSE on test set typically in the range of actual price volatility (varies per stock)
- Example on AAPL: RMSE ~3-8 USD (depending on market conditions)

## 🚀 Quick Start (Google Colab Recommended)

1. Open the notebook in Colab:  
  (https://colab.research.google.com/drive/1Ky4rDVkaexEVGJhbLD4lHrjRNn_pLM28#scrollTo=3ByCSVZyeY3K)

2. Run cells sequentially:
   - Install dependencies
   - Fetch data
   - Train the LSTM model
   - Save model and scaler
   - Launch Streamlit app locally in Colab

3. For local deployment:
   ```bash
   git clone https://github.com/YOUR_USERNAME/stock-price-prediction-lstm.git
   cd stock-price-prediction-lstm
   pip install -r requirements.txt
   streamlit run app.py
