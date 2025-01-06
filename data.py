import yfinance as yf
import pandas as pd
import numpy as np

def download_data(tickers, start, end):
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end)
        if not df.empty:
            data[ticker] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data

def preprocess_data(data):
    processed_data = {}
    for ticker, df in data.items():
        df['PriceMomentum5'] = ts_mean(df['Close'], 5) - df['Close']
        df['VolumeMomentum5'] = ts_mean(df['Volume'], 5) - df['Volume']
        df['Volatility5'] = ts_std(df['Close'], 5)
        df['PriceVolumeCorr5'] = ts_corr(df['Close'], df['Volume'], 5)
        df['Channel5'] = ts_max(df['High'], 5) - ts_min(df['Low'], 5)        
        df['RSI5'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(5).mean() /
                                         df['Close'].diff().clip(upper=0).abs().rolling(5).mean())))
        df['Bollinger5'] = ts_std(df['Close'], 5) / ts_mean(df['Close'], 5)
        
        df['PriceMomentum10'] = ts_mean(df['Close'], 10) - df['Close']
        df['VolumeMomentum10'] = ts_mean(df['Volume'], 10) - df['Volume']
        df['Volatility10'] = ts_std(df['Close'], 10)
        df['PriceVolumeCorr10'] = ts_corr(df['Close'], df['Volume'], 10)
        df['Channel10'] = ts_max(df['High'], 10) - ts_min(df['Low'], 10)
        df['RSI10'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(10).mean() /
                                         df['Close'].diff().clip(upper=0).abs().rolling(10).mean())))
        df['Bollinger10'] = ts_std(df['Close'], 10) / ts_mean(df['Close'], 10)
        
        df['VolumeImbalance'] = (df['Close'] - df['Open']) / (df['High'] - df['Low']) * df['Volume']
        
        df['Return'] = df['Close'].shift(-1) / df['Open'].shift(-1) - 1
        processed_data[ticker] = df.dropna()
    return processed_data

def get_tickers():
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

    market_caps = []
    for ticker in sp500_tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if "marketCap" in info:
                market_caps.append((ticker, info["marketCap"]))
        except Exception as e:
            continue

    sorted_by_market_cap = sorted(market_caps, key=lambda x: x[1], reverse=True)[:100]
    return [ticker for ticker, _ in sorted_by_market_cap]
