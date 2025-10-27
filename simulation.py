import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

tickers = ['GLD', 'SLV']
start_date = '2024-01-01'
end_date = '2024-10-01'

data = yf.download(tickers, start = start_date, end = end_date, auto_adjust=False)['Adj Close']
data["Ratio"] = data.iloc[:, 0] / data.iloc[:, 1]
mean = data['Ratio'].mean()
std = data['Ratio'].std()
data['Z-Score'] = (data['Ratio'] - mean) / std
data['Upper Threshold'] = data['Z-Score'] + std
data['Lower Threshold'] = data['Z-Score'] - std

print(data)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Ratio
axes[0].plot(data.index, data['Ratio'], color='blue', label='Ratio')
axes[0].axhline(mean, color='black', linestyle='--', label='Mean')
axes[0].axhline(mean + std, color='green', linestyle='--', label='+1 Std')
axes[0].axhline(mean - std, color='red', linestyle='--', label='-1 Std')
axes[0].set_title('Ratio and Trade Signal')
axes[0].legend()
axes[0].grid(True)

# Z-Score
axes[1].plot(data.index, data['Z-Score'], color='purple', label='Z-Score')
axes[1].axhline(0, color='black', linestyle='--')
axes[1].axhline(1, color='green', linestyle='--')
axes[1].axhline(-1, color='red', linestyle='--')
axes[1].set_title('Z-Score of Ratio and Trade Signal')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()