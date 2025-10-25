import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from datetime import datetime, timedelta

start_date = "2000-10-01"
end_date = "2025-10-01"


def adf_test(price_series):
    adf_result = sm.tsa.stattools.adfuller(price_series)
    p_value = adf_result[1]
    return p_value < 0.05


def optimal_lag_selection(time_series):
    model = VAR(time_series)
    lag_order_results = model.select_order(maxlags=10)
    optimal_lag = lag_order_results.aic
    return optimal_lag - 1


def johansen_test(asset1, asset2, start_date, end_date):
    asset1_price_series = yf.download(
        asset1, start=start_date, end=end_date, auto_adjust=False
    )["Adj Close"]
    asset2_price_series = yf.download(
        asset2, start=start_date, end=end_date, auto_adjust=False
    )["Adj Close"]
    if adf_test(asset1_price_series) or adf_test(asset2_price_series):
        return [-1, -1]

    merged_price_series = pd.merge(
        asset1_price_series, asset2_price_series, on="Date", how="outer"
    ).dropna()

    johansen_result = coint_johansen(
        merged_price_series,
        det_order=0,
        k_ar_diff=optimal_lag_selection(merged_price_series),
    )

    test_stat_trace_r0 = johansen_result.lr1[0]
    critical_value_trace_r0 = johansen_result.cvt[0, 1]
    return [
        float(test_stat_trace_r0),
        float(critical_value_trace_r0),
    ], merged_price_series


def plot_cointegration(merged_date, asset1, asset2):
    model = sm.OLS(merged_date[asset1], sm.add_constant(merged_date[asset2])).fit()
    hedge_ratio = model.params[asset2]
    spread = merged_date[asset1] - hedge_ratio * merged_date[asset2]

    plt.figure(figsize=(12, 6))
    plt.plot(merged_date.index, spread, label="Spread", color="blue")
    plt.axhline(spread.mean(), color="red", linestyle="--", label="Mean")
    plt.title(f"Cointegration Spread between {asset1} and {asset2}")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.legend()
    plt.show()


asset1 = "GLD"
asset2 = "BTC-USD"
result, merged_data = johansen_test(asset1, asset2, start_date, end_date)
print(result)
plot_cointegration(merged_data, asset1, asset2)
