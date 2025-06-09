# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import timedelta
import numpy as np
import os

from predict import predict
from automation import run_automation

# --- Streamlit Page Config ---
st.set_page_config(page_title=" Stock Price Predictor", layout="centered")

# --- UI Header ---
st.title(" Stock Price Predictor")
st.markdown("Predict the next 5 days of stock closing prices using Machine Learning.")

# --- User Input ---
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, MSFT):", "").upper().strip()

if ticker:
    with st.spinner("Fetching data and predicting..."):
        try:
            data = yf.download(ticker, period='30d', interval='1d')
        except Exception as e:
            st.error(f"Failed to download stock data: {e}")
            st.stop()

        if data.empty:
            st.error("❌ No data found for that ticker. Please check and try again.")
        else:
            try:
                predictions = predict(ticker, data)
                predicted = predictions[-5:]
                last_date = data.index[-1]
                hist_prices = data['Close'].values[-5:]
                hist_dates = data.index[-5:]
                future_dates = [last_date + timedelta(days=i) for i in range(1, 6)]

                # --- Display Predictions ---
                st.subheader("🔮 Predictions")
                for i, price in enumerate(predicted, 1):
                    st.write(f"Day {i}: **${price:.2f}**")

                # --- Plotting ---
                fig, ax = plt.subplots()
                ax.plot(hist_dates, hist_prices, marker='o', label='Historical', color='orange')
                ax.plot(future_dates, predicted, marker='o', label='Predicted', color='blue')
                ax.set_title(f"{ticker} - Closing Price Forecast")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

                # --- Save CSV ---
                df = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted_Close": predicted
                })

                os.makedirs("predictions", exist_ok=True)
                csv_path = f"predictions/predictions_{ticker}.csv"
                df.to_csv(csv_path, index=False)

                st.download_button(
                    " Download CSV",
                    data=df.to_csv(index=False),
                    file_name=f"{ticker}_prediction.csv",
                    mime="text/csv"
                )

                # --- Run Automation (Optional) ---
                if st.button(" Run Full Automation"):
                    run_automation(ticker)
                    st.success(f" Automation completed for {ticker}.")

            except Exception as e:
                st.error(f"An error occurred while predicting: {e}")
