import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import numpy as np


#asset1 = 'USO'
#asset2 = 'SLV'

start_date = '2023-10-01'
end_date = '2025-10-01'
def adf_test(asset1, asset2, start_date, end_date):
    asset1_price_series = yf.download(asset1, start=start_date, end=end_date, auto_adjust=False, progress = True)['Adj Close']
    asset2_price_series = yf.download(asset2, start=start_date, end=end_date, auto_adjust=False, progress = True)['Adj Close']

    log_asset1 = np.log(asset1_price_series)
    log_asset2 = np.log(asset2_price_series)
    merged_price_series = pd.merge(log_asset1, log_asset2, on="Date", how="outer").dropna()

    coint_result = coint(merged_price_series[asset1], merged_price_series[asset2], autolag= 'AIC', trend = 'ct')

    p_value = coint_result[1]
    return p_value

