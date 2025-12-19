# Stock Price Prediction using LSTM

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app)  
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)  
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional deep learning project for next-day stock and cryptocurrency price prediction using Long Short-Term Memory (LSTM) networks. The application fetches real-time historical data from Yahoo Finance, trains an LSTM model, and provides an interactive web interface for live predictions.

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app)  

Open the deployed app directly on Streamlit Community Cloud.

**Default ticker**: BTC-USD (Bitcoin) – fully supports any Yahoo Finance ticker (e.g., AAPL, TSLA, ETH-USD, GOOGL, MSFT).

## 📊 Project Overview

- **Objective**: Forecast the next trading day's closing price based on historical data.
- **Model Architecture**: Multi-layer LSTM neural network with Dropout for regularization.
- **Data Source**: Real-time data from Yahoo Finance via `yfinance`.
- **Framework**: TensorFlow / Keras.
- **Deployment**: Interactive web application using Streamlit.
- **Key Features**:
  - Live data fetching and preprocessing
  - Robust LSTM model with early stopping
  - Next-day price prediction with percentage change
  - Interactive chart visualization
  - Full support for stocks and cryptocurrencies

## 📁 Repository Structure
Hex_Software_STOCK-PRICE-PREDICTION/
├── models/
│   ├── lstm_stock_model.h5      # Trained LSTM model
│   └── scaler.pkl               # Fitted MinMaxScaler
├── app.py                       # Streamlit web application
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
└── .streamlit/                  # Optional Streamlit config
text## 🛠️ Local Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/eslamalsaeeed72-droid/Hex_Software_STOCK-PRICE-PREDICTION.git
   cd Hex_Software_STOCK-PRICE-PREDICTION

Install dependencies:Bashpip install -r requirements.txt
Run the application locally:Bashstreamlit run app.py

The app will launch at http://localhost:8501.
📈 How to Use

Enter a valid Yahoo Finance ticker symbol (default: BTC-USD).
Adjust the historical data range using the slider.
Click "Predict Next Day Closing Price" to view:
Predicted closing price for the next day
Current latest closing price
Expected percentage change
Historical price chart with prediction marker


📊 Model Performance

Trained on approximately 5 years of daily closing prices
Evaluated using Root Mean Squared Error (RMSE)
Designed to handle volatility in both traditional stocks and cryptocurrencies

🔮 Future Improvements

Multi-step forecasting (predict next 5–10 days)
Integration of technical indicators (RSI, MACD, Moving Averages)
Comparison with alternative models (GRU, Transformer, Prophet)
Confidence intervals and uncertainty quantification
Prediction history and user dashboard

👤 Author

Name: Eslam Alsaeeed
GitHub: @eslamalsaeeed72-droid
LinkedIn: eslamalsaeed72@gmail.com

📄 License
This project is open source and licensed under the MIT License.
