import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import numpy as np

asset1 = 'GLD'
asset2 = 'SLV'

start_date = '2020-01-01'
end_date = '2024-01-01'

asset1_price_series = yf.download(asset1, start=start_date, end=end_date, auto_adjust=False, progress = True)['Adj Close']
asset2_price_series = yf.download(asset2, start=start_date, end=end_date, auto_adjust=False, progress = True)['Adj Close']

print(asset1_price_series)
merged_price_series 

'''
log_asset1 = np.log(asset1_price_series)
log_asset2 = np.log(asset2_price_series)

coint_result = coint(log_asset1, log_asset2)

test_statistic = coint_result[0]
p_value = coint_result[1]
critical_values = coint_result[2]

print(f'Cointegration p-value: {p_value}')
'''