import numpy as np
import pandas as pd
import yfinance as yf

def fetch_financial_data(tickers=["^VIX", "^GSPC"], period="2y", start=None, end=None):
    """
    Fetches real-world financial data using yfinance.
    Can handle multiple tickers for multivariate TDA.
    """
    try:
        # If single ticker string provided, convert to list
        if isinstance(tickers, str):
            tickers = [tickers]
            
        if start and end:
            data = yf.download(tickers, start=start, end=end, progress=False)
        else:
            data = yf.download(tickers, period=period, progress=False)
        if data.empty:
            raise ValueError(f"No data found for tickers {tickers}")
        
        # If multiple tickers, yfinance returns MultiIndex columns.
        # We want the 'Close' prices for all tickers.
        if len(tickers) > 1:
            closes = data['Close']
        else:
            # Single ticker might still have MultiIndex if passed as list
            if 'Close' in data.columns and isinstance(data['Close'], pd.DataFrame):
                closes = data['Close']
            else:
                closes = pd.DataFrame({tickers[0]: data['Close'].values}, index=data.index)
        
        # Clean: Fill missing values and align
        closes = closes.ffill().dropna()
        
        # Reset index to get Date as a column
        data_reset = closes.reset_index()
        
        # Create a time vector
        t = np.arange(len(closes))
        
        # For 'value', if multiple, we'll return the whole dataframe columns or a specific primary
        # For simplicity in the existing app, we'll keep 'value' as the first ticker
        # but the dataframe will contain all.
        df_result = pd.DataFrame({'time': t, 'date': data_reset['Date']})
        for ticker in tickers:
            df_result[ticker] = data_reset[ticker].values
            
        # Add 'value' and 'clean_value' as aliases for the first ticker for backward compatibility
        primary_ticker = tickers[0]
        df_result['value'] = df_result[primary_ticker]
        
        # Reduce smoothing for Bitcoin specifically, as a 20-day mean is too lagging for crypto
        smooth_window = 5 if "BTC" in primary_ticker else 20
        df_result['clean_value'] = pd.Series(df_result[primary_ticker]).rolling(window=smooth_window, min_periods=1).mean().values
        
        return df_result
    except Exception as e:
        print(f"Error fetching data: {e}")
        # Return fallback synthetic data if it fails
        return generate_synthetic_data(n_samples=500, noise_level=0.5)


def generate_synthetic_data(n_samples=1000, noise_level=0.1, complex_pattern=True):
    # Time vector
    t = np.linspace(0, 100, n_samples)
    
    # Base signal (smooth sine wave)
    signal = np.sin(t)
    
    if complex_pattern:
        # Add higher frequency components for complexity
        signal += 0.5 * np.sin(2.5 * t) + 0.2 * np.cos(5 * t)
        
    # Inject Gaussian noise
    noise = np.random.normal(0, noise_level, n_samples)
    y = signal + noise
    
    return pd.DataFrame({'time': t, 'value': y, 'clean_value': signal})

def generate_ecg_data(n_samples=1000, noise_level=0.1, arrhythmic=False):
    """
    Generates a synthetic ECG-like signal for demonstrating medical TDA.
    """
    t = np.linspace(0, 10, n_samples)
    # Basic heartbeat rhythm (QRS complex approximation)
    heartbeat = np.zeros(n_samples)
    for i in range(1, 10):
        pos = i * (n_samples // 10)
        # P wave
        heartbeat += 0.1 * np.exp(-((t - t[pos-20])**2) / 0.01)
        # QRS complex
        heartbeat += 1.0 * np.exp(-((t - t[pos])**2) / 0.001)
        # T wave
        heartbeat += 0.2 * np.exp(-((t - t[pos+30])**2) / 0.02)
    
    if arrhythmic:
        # Add random ectopic beats
        for _ in range(3):
            pos = np.random.randint(100, n_samples-100)
            heartbeat += 0.8 * np.exp(-((t - t[pos])**2) / 0.0005)
            
    # Add noise
    noise = np.random.normal(0, noise_level, n_samples)
    value = heartbeat + noise
    
    # Scale to look like real mV
    value = value * 0.5
    
    return pd.DataFrame({
        'time': np.arange(n_samples),
        'value': value,
        'clean_value': heartbeat * 0.5
    })

def generate_bearing_data(n_samples=1000, noise_level=0.1):
    """
    Simulates Industrial Bearing Vibration data.
    Transitions from Healthy -> Subtle Structural Wear -> Total Seizure.
    """
    t = np.linspace(0, 100, n_samples)
    
    # Base high-frequency vibration (Healthy)
    base_vibe = 0.5 * np.sin(50 * t) 
    
    # 1. Healthy Zone (0-400)
    signal = base_vibe.copy()
    
    # 2. Structural Wear Zone (400-800)
    # Inject subtle, irregular phase-shifts and harmonics (Topology changes)
    wear_start = 400
    for i in range(wear_start, 800):
        # As wear increases, the 'shape' of the vibration becomes more complex/fractured
        severity = (i - wear_start) / 400.0
        signal[i] += severity * 0.4 * np.sin(120 * t[i]) # Adding high-freq micro-jitters
        signal[i] *= (1.0 + 0.1 * np.sin(5 * t[i]))      # Adding subtle amplitude modulation
        
    # 3. Impending Failure / Seizure (800-1000)
    # Huge erratic spikes
    failure_start = 800
    for i in range(failure_start, n_samples):
        severity = (i - failure_start) / 200.0
        signal[i] += severity * 1.5 * np.random.normal()
        
    # Add sensor noise
    noise = np.random.normal(0, noise_level, n_samples)
    value = signal + noise
    
    return pd.DataFrame({
        'time': np.arange(n_samples),
        'value': value,
        'clean_value': signal
    })

def create_sliding_windows(data, clean_data, window_size, forecast_horizon=1):
    X, y, y_clean = [], [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i:(i + window_size)])
        # Always target the first column (the primary ticker) for forecasting
        if data.ndim > 1:
            y_val = data[i + window_size + forecast_horizon - 1, 0]
        else:
            y_val = data[i + window_size + forecast_horizon - 1]
            
        y.append(y_val)
        y_clean.append(clean_data[i + window_size + forecast_horizon - 1])
        
    return np.array(X), np.array(y), np.array(y_clean)
