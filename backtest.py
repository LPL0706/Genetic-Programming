import pandas as pd
import numpy as np
from data import *

def define_window_function():
    global ts_mean_5, ts_mean_10, ts_std_5, ts_std_10, ts_rank_5, ts_rank_10, ts_corr_5, ts_corr_10, ts_max_5, ts_max_10, ts_min_5, ts_min_10, add, subtract, multiply, div, neg, sqrt, max, min

    ts_mean_5 = lambda series: ts_mean(series, 5)
    ts_mean_10 = lambda series: ts_mean(series, 10)
    ts_std_5 = lambda series: ts_std(series, 5)
    ts_std_10 = lambda series: ts_std(series, 10)
    ts_rank_5 = lambda series: ts_rank(series, 5)
    ts_rank_10 = lambda series: ts_rank(series, 10)
    ts_corr_5 = lambda series1, series2: ts_corr(series1, series2, 5)
    ts_corr_10 = lambda series1, series2: ts_corr(series1, series2, 10)
    ts_max_5 = lambda series: ts_max(series, 5)
    ts_max_10 = lambda series: ts_max(series, 10)
    ts_min_5 = lambda series: ts_min(series, 5)
    ts_min_10 = lambda series: ts_min(series, 10)
    
    add = np.add
    subtract = np.subtract
    multiply = np.multiply
    div = safe_divide
    neg = np.negative
    sqrt = np.sqrt
    max = np.maximum
    min = np.minimum
    
def calculate_factor(processed_data, alpha_expression):
    for ticker, df in processed_data.items():
        local_vars = {col: df[col] for col in df.columns}
        processed_data[ticker]['Factor'] = eval(alpha_expression, globals(), local_vars)
    return processed_data

def backtest(processed_data):
    combined_df = pd.DataFrame()
    for ticker, df in processed_data.items():
        df = df[['Factor', 'Return']].copy()
        df['Ticker'] = ticker
        combined_df = pd.concat([combined_df, df])

    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'index': 'Date'}, inplace=True)
    grouped = combined_df.groupby('Date')

    net_value = 1
    net_value_series = []
    daily_returns = []
    for date, group in grouped:
        group = group.sort_values(by='Factor', ascending=False)
        long_group = group.head(20)
        short_group = group.tail(20)

        long_return = long_group['Return'].mean()
        short_return = short_group['Return'].mean()
        daily_return = long_return - short_return
        daily_returns.append(daily_return)

        net_value *= (1 + daily_return)
        net_value_series.append({'Date': date, 'NetValue': net_value})

    net_value_df = pd.DataFrame(net_value_series)
    net_value_df.set_index('Date', inplace=True)
    
    trading_days_per_year = 252
    cumulative_return = net_value / net_value_df.iloc[0]['NetValue'] - 1
    annualized_return = np.mean(daily_returns) * trading_days_per_year
    annualized_volatility = np.std(daily_returns) * np.sqrt(trading_days_per_year)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    
    print("Annualized Return:", annualized_return)
    print("Sharpe Ratio:", sharpe_ratio)

    return net_value_df

alphas = {
    "alpha1": 'div(ts_corr_10(multiply(Volatility5, Close), multiply(Low, Volatility5)), min(ts_rank_10(Volatility5), ts_rank_5(Channel10)))',
}

processed_data = preprocess_data(data)
define_window_function()

net_values = {}

for alpha_name, alpha_expression in alphas.items():
    print(alpha_name)
    processed_data = calculate_factor(processed_data, alpha_expression)
    net_value = backtest(processed_data)
    net_values[alpha_name] = net_value

plt.figure(figsize=(12, 8))
for alpha_name, net_value in net_values.items():
    plt.plot(net_value.index, net_value['NetValue'], label=alpha_name)

plt.xlabel('Date')
plt.ylabel('Net Value')
plt.legend()
plt.show()
