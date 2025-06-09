import pandas as pd
import joblib
import numpy as np

def compute_RSI(series, window):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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

def prepare_features(df):
    feature_cols = ['MA_5', 'MA_10', 'RSI_14', 'Lag_1_Close', 'Lag_2_Close', 'Day_of_Week', 'Month']
    return df[feature_cols].values

def load_models_scalars(ticker):
    model = joblib.load(f'models/svr_model_{ticker}.pkl')
    scaler_X = joblib.load(f'models/scaler_X_{ticker}.pkl')
    scaler_y = joblib.load(f'models/scaler_y_{ticker}.pkl')
    return model, scaler_X, scaler_y

def predict(ticker, recent_data_df):
    df_feat = add_features(recent_data_df)
    X = prepare_features(df_feat)
    model, scaler_X, scaler_y = load_models_scalars(ticker)
    x_scaled = scaler_X.transform(X)
    y_pred_scaled = model.predict(x_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    return y_pred

