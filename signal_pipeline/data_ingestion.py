import yfinance as yf
import pandas as pd
import numpy as np
import time
from functools import lru_cache

__all__ = [
    "get_iv_proxy",
    "get_iv_history_proxy",
    "get_realized_vol",
    "get_volume_ratio",
    "simulate_oi_concentration",
]

@lru_cache(maxsize=32)
def get_iv_proxy(ticker: str, retries: int = 3, delay: float = 2.0) -> float:
    """Return a simple implied volatility proxy using ATM options."""
    for attempt in range(retries):
        try:
            option_chain = yf.Ticker(ticker).option_chain()
            calls = option_chain.calls
            puts = option_chain.puts
            atm_strike = yf.Ticker(ticker).info['regularMarketPrice']
            closest_call = calls.iloc[(calls['strike'] - atm_strike).abs().argsort()[:1]]
            closest_put = puts.iloc[(puts['strike'] - atm_strike).abs().argsort()[:1]]
            iv = (closest_call['impliedVolatility'].values[0] + closest_put['impliedVolatility'].values[0]) / 2
            return float(iv)
        except Exception as e:
            print(f"IV proxy error for {ticker} (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return float('nan')

@lru_cache(maxsize=32)
def get_iv_history_proxy(ticker: str, days: int = 30, retries: int = 3, delay: float = 2.0) -> np.ndarray:
    """Return a history of daily IV proxy values."""
    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker).history(period=f"{days}d")
            daily_range = hist['High'] - hist['Low']
            iv_proxy = daily_range / hist['Close']
            return iv_proxy.values
        except Exception as e:
            print(f"IV history proxy error for {ticker} (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return np.array([])

@lru_cache(maxsize=32)
def get_realized_vol(
    ticker: str,
    window: int = 10,
    retries: int = 3,
    delay: float = 2.0,
    use_garch: bool = False,
    use_lstm: bool = False,
    external_data: pd.DataFrame | None = None,
) -> float:
    """Calculate realized volatility using optional GARCH or LSTM models."""
    try:
        from arch import arch_model
    except ImportError:
        arch_model = None
    try:
        from keras.models import Sequential
        from keras.layers import LSTM, Dense
    except ImportError:
        Sequential = None
    for attempt in range(retries):
        try:
            hist = external_data if external_data is not None else yf.Ticker(ticker).history(period='30d')
            returns = hist['Close'].pct_change().dropna()
            if use_garch and arch_model:
                am = arch_model(returns, vol='Garch', p=1, q=1)
                res = am.fit(disp='off')
                garch_vol = np.mean(res.conditional_volatility[-window:])
                return float(garch_vol)
            elif use_lstm and Sequential:
                X = returns.values.reshape(-1, 1)
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)
                X_seq = np.array([X_scaled[i-window:i] for i in range(window, len(X_scaled))])
                y_seq = X_scaled[window:]
                model = Sequential()
                model.add(LSTM(10, input_shape=(window,1)))
                model.add(Dense(1))
                model.compile(loss='mse', optimizer='adam')
                model.fit(X_seq, y_seq, epochs=5, batch_size=1, verbose=0)
                pred = model.predict(X_seq)
                lstm_vol = np.std(pred[-window:])
                return float(lstm_vol)
            else:
                return float(np.std(returns[-window:]))
        except Exception as e:
            print(f"Realized vol error for {ticker} (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return float('nan')

@lru_cache(maxsize=32)
def get_volume_ratio(ticker: str, retries: int = 3, delay: float = 2.0) -> float:
    """Return the option volume to equity volume ratio."""
    for attempt in range(retries):
        try:
            data = yf.Ticker(ticker).info
            option_vol = data.get('averageDailyVolume10Day', 0)
            equity_vol = data.get('volume', 1)
            return option_vol / equity_vol if equity_vol else 0.0
        except Exception as e:
            print(f"Volume ratio error for {ticker} (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return float('nan')

@lru_cache(maxsize=32)
def simulate_oi_concentration(ticker: str, retries: int = 3, delay: float = 2.0) -> float:
    """Return percent of open interest within ±5% of spot price."""
    for attempt in range(retries):
        try:
            tkr = yf.Ticker(ticker)
            spot = tkr.history(period="1d")['Close'].iloc[-1]
            expirations = tkr.options
            if not expirations:
                return float('nan')
            exp = expirations[0]
            chain = tkr.option_chain(exp)
            df = pd.concat([chain.calls, chain.puts], ignore_index=True)
            df['openInterest'] = df['openInterest'].fillna(0)
            total_oi = df['openInterest'].sum()
            band_mask = (df['strike'] >= spot * 0.95) & (df['strike'] <= spot * 1.05)
            near_oi = df.loc[band_mask, 'openInterest'].sum()
            return near_oi / total_oi if total_oi > 0 else float('nan')
        except Exception as e:
            print(f"OI concentration error for {ticker} (attempt {attempt+1}): {e}")
            time.sleep(delay)
    return float('nan')
