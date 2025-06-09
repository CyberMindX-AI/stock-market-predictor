import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# RSI function
def compute_RSI(series, window):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Add features
def add_features(df):
    close = df['Close']
    
    df['MA_5'] = close.rolling(5).mean()
    df['MA_10'] = close.rolling(10).mean()
    df['RSI_14'] = compute_RSI(close, 14)
    
    df['Lag_1_Close'] = close.shift(1)
    df['Lag_2_Close'] = close.shift(2)
    
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    
    df['Target'] = close.shift(-1)
    return df.dropna()

# Train and evaluate
def train_and_evaluate(ticker):
    print(f"Processing {ticker}...")
    df = yf.download(ticker, start='2018-01-01', end='2024-01-01')
    df = add_features(df)
    
    feature_cols = ['MA_5', 'MA_10', 'RSI_14', 'Lag_1_Close', 'Lag_2_Close', 'Day_of_Week', 'Month']
    X = df[feature_cols].values
    y = df['Target'].values
    
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
    
    svr = SVR()
    param_distributions = {
        'kernel': ['rbf', 'linear', 'poly'],
        'C': np.logspace(-2, 2, 20),
        'epsilon': np.linspace(0.01, 0.5, 20),
        'gamma': ['scale', 'auto']
    }
    tscv = TimeSeriesSplit(n_splits=5)
    
    search = RandomizedSearchCV(
        svr, param_distributions, n_iter=30, cv=tscv,
        verbose=0, n_jobs=-1, random_state=42
    )
    
    search.fit(X_train_scaled, y_train_scaled)
    
    print(f"Best params for {ticker}: {search.best_params_}")
    best_svr = search.best_estimator_
    
    y_pred_scaled = best_svr.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{ticker} MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
    if r2 < 0.5:
        print(f"Warning: {ticker} model has low R2 score,")
    elif r2 < 0.7:
        print(f"Note: {ticker} model has moderate R2 score.")
    else:
        print(f"{ticker} model has good R2 score.")
    
    return y_test, y_pred, best_svr, scaler_X, scaler_y

# Run for tickers
tickers = ['AAPL', 'MSFT']
results = {}
models = {}

for t in tickers:
    y_test, y_pred, best_svr, scaler_X, scaler_y = train_and_evaluate(t)
    results[t] = (y_test, y_pred)
    models[t] = (best_svr, scaler_X, scaler_y)

# Plot results
plt.figure(figsize=(14,6))
for i, t in enumerate(tickers, 1):
    y_test, y_pred = results[t]
    plt.subplot(1, 2, i)
    plt.plot(y_test, label='Actual Close Price')
    plt.plot(y_pred, label='SVR Predicted Close Price', alpha=0.7)
    plt.title(f'{t} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
plt.tight_layout()
plt.show()

# Save the models
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

for t in models:
    try:
        print(f"Saving model for {t}...")  
        model, scaler_X, scaler_y = models[t]
        joblib.dump(model, os.path.join(output_dir, f'svr_model_{t}.pkl'))
        joblib.dump(scaler_X, os.path.join(output_dir, f'scaler_X_{t}.pkl'))
        joblib.dump(scaler_y, os.path.join(output_dir, f'scaler_y_{t}.pkl'))
        print(f"{t} model and scalers saved successfully.")
    except Exception as e:
        print(f"Error saving {t} model: {e}")

print("All models and scalers processed.")


    