import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import timedelta
from predict import predict

def save_predictions(future_dates, predicted, ticker):
    df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': predicted
    })
    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    df.to_csv(f'predictions/predictions_{ticker}.csv', index=False)

def plot_predictions(ticker, historical_dates, historical_prices, future_dates, predicted):
    plt.figure(figsize=(10, 5))
    plt.plot(historical_dates, historical_prices, marker='o', label="Historical", color='orange', linewidth=2)
    plt.plot(future_dates, predicted, marker='o', label='Predicted', color='blue', linewidth=2)
    plt.title(f'Predicted Closing Prices for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(f'plots/predictions_{ticker}.png')
    plt.close()

def run_automation(ticker):
    print(f"Running automation for {ticker}...")
    data = yf.download(ticker, period='30d', interval='1d')
    if data.empty:
        print(f"No data found for {ticker}.")
        return

    predictions = predict(ticker, data)
    predicted = predictions[-5:]
    last_date = data.index[-1]
    historical_prices = data['Close'].values[-5:]
    historical_dates = data.index[-5:]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 6)]

    plot_predictions(ticker, historical_dates, historical_prices, future_dates, predicted)
    save_predictions(future_dates, predicted, ticker)

    print(f"Automation completed for {ticker}.")
