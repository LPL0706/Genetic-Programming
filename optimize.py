from scipy.optimize import minimize
from itertools import product
import pandas as pd
import numpy as np

def scale(series):
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return series * 0
    return (series - min_val) / (max_val - min_val)

def zscore(series):
    mean_val = series.mean()
    std_val = series.std()
    if std_val == 0:
        return series * 0
    return (series - mean_val) / std_val

def scale_factor(processed_data, factor_name):
    for date in processed_data[list(processed_data.keys())[0]].index:
        cross_section = []
        for ticker, df in processed_data.items():
            if date in df.index:
                cross_section.append((ticker, df.loc[date, factor_name]))

        if cross_section:
            df_cross_section = pd.DataFrame(cross_section, columns=["Ticker", factor_name])
            df_cross_section[factor_name] = scale(df_cross_section[factor_name])

            for _, row in df_cross_section.iterrows():
                processed_data[row["Ticker"]].loc[date, factor_name] = row[factor_name]
    return processed_data

def zscore_factor(processed_data, factor_name):
    for date in processed_data[list(processed_data.keys())[0]].index:
        cross_section = []
        for ticker, df in processed_data.items():
            if date in df.index:
                cross_section.append((ticker, df.loc[date, factor_name]))

        if cross_section:
            df_cross_section = pd.DataFrame(cross_section, columns=["Ticker", factor_name])
            df_cross_section[factor_name] = zscore(df_cross_section[factor_name])

            for _, row in df_cross_section.iterrows():
                processed_data[row["Ticker"]].loc[date, factor_name] = row[factor_name]
    return processed_data

def combine_factors(processed_data, factor_names, weights):
    combined_name = "Factor"
    for ticker, df in processed_data.items():
        combined_factor = sum(
            weight * df[factor_name] for weight, factor_name in zip(weights, factor_names)
        )
        processed_data[ticker][combined_name] = combined_factor
    return processed_data

def round_weights(weights):
    rounded_weights = np.round(weights, 2)
    correction = 1 - np.sum(rounded_weights)
    if correction != 0:
        max_index = np.argmax(rounded_weights)
        rounded_weights[max_index] += correction 
    return rounded_weights

def greedy_optimize(processed_data, alphas, weight_range=np.arange(0, 1.1, 0.05)):
    alpha_names = list(alphas.keys())
    num_alphas = len(alpha_names)
    best_weights = [1] + [0] * (num_alphas - 1)
    best_sharpe = -np.inf

    for i in range(1, num_alphas):
        current_best_sharpe = -np.inf
        current_best_weight = None
        for weight in weight_range:
            weights = best_weights.copy()
            weights[i] = weight
            weights = np.array(weights) / np.sum(weights)
            
            processed_data = combine_factors(processed_data, alpha_names, weights)

            net_value, annualized_return, sharpe_ratio = backtest(processed_data)
            
            if sharpe_ratio > current_best_sharpe:
                current_best_sharpe = sharpe_ratio
                current_best_weight = weight

        best_weights[i] = current_best_weight
        best_weights = np.array(best_weights) / np.sum(best_weights)

        if current_best_sharpe > best_sharpe:
            best_sharpe = current_best_sharpe
    
    return best_weights, best_sharpe

def objective_function(weights, processed_data, alphas, alpha_names):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    processed_data = combine_factors(processed_data, alpha_names, weights)
    _, _, sharpe_ratio = backtest(processed_data)
    return -sharpe_ratio

def constraint_sum_to_one(weights):
    return np.sum(weights) - 1

def global_optimize(processed_data, alphas):
    alpha_names = list(alphas.keys())
    num_alphas = len(alpha_names)

    initial_weights = np.random.dirichlet(np.ones(num_alphas), size=1)[0]

    constraints = [{'type': 'eq', 'fun': constraint_sum_to_one}]

    bounds = [(0, 1) for _ in range(num_alphas)]

    result = minimize(
        objective_function,
        initial_weights,
        args=(processed_data, alphas, alpha_names),
        bounds=bounds,
        constraints=constraints,
        method='SLSQP'
    )

    if result.success:
        optimal_weights = result.x
        optimal_sharpe = -result.fun
        return optimal_weights, optimal_sharpe
    else:
        raise ValueError("Optimization failed:", result.message)

def refine_global_optimize(processed_data, alphas, iterations=10):
    alpha_names = list(alphas.keys())
    num_alphas = len(alpha_names)
    best_sharpe = -np.inf
    best_weights = None

    for _ in range(iterations):
        initial_weights = np.random.dirichlet(np.ones(num_alphas), size=1)[0]

        result = minimize(
            objective_function,
            initial_weights,
            args=(processed_data, alpha_names),
            bounds=[(0, 1) for _ in range(num_alphas)],
            constraints=[{'type': 'eq', 'fun': constraint_sum_to_one}],
            method='SLSQP',
            options={'ftol': 1e-8, 'disp': False}
        )

        if result.success and -result.fun > best_sharpe:
            best_sharpe = -result.fun
            best_weights = result.x

    if best_weights is not None:
        refined_result = minimize(
            objective_function,
            best_weights,
            args=(processed_data, alpha_names),
            bounds=[(0, 1) for _ in range(num_alphas)],
            constraints=[{'type': 'eq', 'fun': constraint_sum_to_one}],
            method='trust-constr',
            options={'xtol': 1e-10, 'gtol': 1e-10, 'disp': False}
        )

        if refined_result.success and -refined_result.fun > best_sharpe:
            best_sharpe = -refined_result.fun
            best_weights = refined_result.x

    return best_weights, best_sharpe

def grid_search_optimize(processed_data, alphas, step=0.1):
    alpha_names = list(alphas.keys())
    num_alphas = len(alpha_names)
    weight_range = np.arange(0, 1.1, step)

    best_weights = None
    best_sharpe = -np.inf

    for weights in product(weight_range, repeat=num_alphas):
        weights = np.array(weights)
        if not np.isclose(weights.sum(), 1):
            continue

        combined_data = combine_factors(processed_data, alpha_names, weights)
        net_value, _, sharpe_ratio = backtest(combined_data)

        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_weights = weights

    return best_weights, best_sharpe
