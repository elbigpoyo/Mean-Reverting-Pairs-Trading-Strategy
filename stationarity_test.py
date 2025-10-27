import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import plotly.graph_objs as go
from dateutil.relativedelta import relativedelta

start_date = "2017-01-01"
end_date = "2022-01-01"

def aspread_test(price_series):
    aspread_result = adfuller(price_series)
    p_value = aspread_result[1]
    return p_value < 0.05


def optimal_lag_selection(time_series):
    model = VAR(time_series)
    lag_order_results = model.select_order(maxlags=10)
    optimal_lag = lag_order_results.aic
    return optimal_lag - 1


def johansen_test(data, start_date, end_date):
    time_series = yf.download(data, start=start_date, end=end_date, auto_adjust=True)["Close"].dropna()

    if aspread_test(time_series[data[0]]) or aspread_test(time_series[data[1]]):
        return "Not applicable for the Johansen Test", -1 , -1

    johansen_result = coint_johansen(
        time_series,
        det_order=0,
        k_ar_diff=optimal_lag_selection(time_series)
    )

    test_stat_trace_r0 = johansen_result.lr1[0]
    critical_value_trace_r0 = johansen_result.cvt[0, 1]
    coint_vector = johansen_result.evec[:, 0]

    return [
        float(test_stat_trace_r0),
        float(critical_value_trace_r0),
    ], time_series, coint_vector

def plot_cointegration(asset_pairs, data, coint_vector):
    spread = data.dot(coint_vector)
    spread.name = "Stationary Spread"
    plt.figure(figsize=(12, 6))
    spread.plot(title=f"Stationary Cointegrating Spread ({asset_pairs[0]} & {asset_pairs[1]})", legend=True)
    mean = spread.mean()
    std = spread.std()
    plt.axhline(mean, color="red", linestyle="--", label="Mean")
    plt.axhline(mean + 2 * std, color="green", linestyle=":", label="+2 Std Dev")
    plt.axhline(mean - 2 * std, color="green", linestyle=":", label="-2 Std Dev")

    plt.legend(loc="upper left")
    plt.ylabel("Spread Value")
    plt.grid(True, alpha=0.5)
    plt.show()

def compute_time_range(end_date, asset_pairs):
    new_start_date = end_date
    dt_end_date = datetime.strptime(end_date, "%Y-%m-%d")
    new_dt_end_date = dt_end_date + relativedelta(years=3)
    new_end_date_str = new_dt_end_date.strftime("%Y-%m-%d")
    time_series = yf.download(asset_pairs, start=new_start_date, end=new_end_date_str, auto_adjust=True)["Close"].dropna()
    return time_series

def compute_spread(time_series, coint_vector):
    spread = time_series.dot(coint_vector)
    spread_df = pd.DataFrame(spread, columns=["Spread"])
    return spread_df

def plot_ratio(data, asset):
    ratio = data[asset[0]] / data[asset[1]]
    plt.figure(figsize=(12, 6))
    ratio.plot(title=f"Ratio between ({asset_pairs[0]} & {asset_pairs[1]})", legend=True)
    plt.legend()
    plt.ylabel("Ratio Value")
    plt.grid(True, alpha=0.5)
    plt.show()

def plot_bollinger(data):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data["Spread"], mode="lines", name="Price"))

    fig.add_trace(go.Scatter(x=data.index, y=data["Mean"], mode="lines", name="Middle Bollinger Band", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=data.index, y=data["Upper"], mode="lines", name="Upper Bollinger Band", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=data.index, y=data["Lower"], mode="lines", name="Lower Bollinger Band", line=dict(color="green")))

    # --- Identify touch points ---
    upper_touch = data["Spread"] >= data["Upper"]
    lower_touch = data["Spread"] <= data["Lower"]
    #tolerance = 0.08 * data["Spread"].std()
    #middle_touch = (data["Spread"] - data["Mean"]).abs() <= tolerance

    upper_points = data[upper_touch]
    lower_points = data[lower_touch]
    #middle_points = data[middle_touch]

    is_above = data["Spread"] > data["Mean"]
    was_above = is_above.shift(1)
    middle_crossing = is_above != was_above
    middle_crossing.iloc[0] = False
    middle_points = data[middle_crossing]

    # --- Add markers for touches ---
    fig.add_trace(go.Scatter(
        x=middle_points.index,
        y=middle_points["Spread"],
        mode="markers",
        name="Cross Middle Band",
        marker=dict(color="blue", size=10, symbol="star")
    ))

    fig.add_trace(go.Scatter(
        x=upper_points.index,
        y=upper_points["Spread"],
        mode="markers",
        name="Touch Upper Band",
        marker=dict(color="red", size=10, symbol="star")
    ))

    fig.add_trace(go.Scatter(
        x=lower_points.index,
        y=lower_points["Spread"],
        mode="markers",
        name="Touch Lower Band",
        marker=dict(color="green", size=10, symbol="star")
    ))

    # --- Layout ---
    fig.update_layout(
        title="Pair Trading Spread with Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Spread Value",
        showlegend=True,
        template="plotly_white"
    )

    fig.show()


def generate_bollinger_signals(spread, window, num_std):
    spread['Mean'] = spread['Spread'].rolling(window).mean()
    spread['Std'] = spread['Spread'].rolling(window).std()
    spread['Upper'] = spread['Mean'] + num_std * spread['Std']
    spread['Lower'] = spread['Mean'] - num_std * spread['Std']

    position = []
    current_pos = 0  # +1 long, -1 short, 0 flat

    for i in range(len(spread)):
        s = spread['Spread'].iloc[i]
        mean = spread['Mean'].iloc[i]
        ub = spread['Upper'].iloc[i]
        lb = spread['Lower'].iloc[i]

        # Entry/exit logic
        if s > ub and current_pos >= 0:
            current_pos = -1  # short spread
        elif s < lb and current_pos <= 0:
            current_pos = 1   # long spread
        elif (current_pos == 1 and s >= mean) or (current_pos == -1 and s <= mean):
            current_pos = 0   # exit / liquidate

        position.append(current_pos)

    spread['Position'] = position
    return spread

def bollinger_strategy(spread, num_std, window, time_series, asset_pairs, coint_vector):
    '''
    spread["Mean"] = spread["Spread"].rolling(window).mean()
    spread["Std"] = spread["Spread"].rolling(window).std()

    spread["Upper"] = spread["Mean"] + num_std * spread["Std"]
    spread["Lower"] = spread["Mean"] - num_std * spread["Std"]
    '''

    position = ""
    cash = 100000
    asset1 = 0
    asset2 = 0
    multiplier = 50

    spread["Position"] = 0
    spread["Cash"] = cash


    '''
    for t in range(1, len(spread)):
        price = spread["Spread"].iloc[t]
        mean = spread["Mean"].iloc[t]
        upper = spread["Upper"].iloc[t]
        lower = spread["Lower"].iloc[t]
        prev_pos = spread["Position"].iloc[t-1]

        #Entry
        if price >= upper:
            position = "Short" # when above upper bound, create short signal
        elif price <= lower:
            position = "Long" # when below lower bound, create long signal
        elif price <= upper and price <= mean and prev_pos == 'Hold':
            position = 'Liquidate'
        elif price >= lower and price >= mean and prev_pos == 'Hold':
            position = 'Liquidate'
        else:
            position = 'Hold'

        spread.loc[spread.index[t], "Position"] = position
    '''

    for i in range(1, len(spread)):
        asset1_amt = round(coint_vector[0] * multiplier)
        asset2_amt = round(coint_vector[1] * multiplier)

        asset1_price = time_series[asset_pairs[0]].iloc[i] * asset1_amt
        asset2_price = time_series[asset_pairs[1]].iloc[i] * asset2_amt
        pos = spread['Position'].iloc[i]

        if pos == 'Long': #Short gold buy silver
            cash += asset1_price
            cash -= asset2_price
            asset1 -= asset1_amt
            asset2 += asset2_amt
        elif pos == 'Short': #Buy gold short silver
            cash -= asset1_price
            cash += asset2_price
            asset1 += asset1_amt
            asset2 -= asset2_amt
        elif pos == 'Liquidate':
            cash += (asset1 * time_series[asset_pairs[0]].iloc[i]) + (asset2 * time_series[asset_pairs[1]].iloc[i])
            asset1 = 0
            asset2 = 0
        spread.loc[spread.index[i], 'Cash'] = cash

    return_pct = (cash - 100000) / 100000

    return spread.dropna()

def graph_returns(data):
    plt.figure(figsize=(12,6))
    plt.plot(data.index, data['Cash'], label='Cash', color='blue')
    plt.title('Cash Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cash')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


asset_pairs = ["GLD", "SLV"]
result, merged_data, coint_vector = johansen_test(asset_pairs, start_date, end_date)
print(result)
print(coint_vector)

time_series = compute_time_range(end_date, asset_pairs)
test_spread = compute_spread(time_series, coint_vector)

spread1 = generate_bollinger_signals(test_spread, 2, 10)
print(spread1.head(50))
#bollinger_spread = bollinger_strategy(test_spread, time_series, asset_pairs, coint_vector)
#graph_returns(bollinger_spread)
#plot_bollinger(bollinger_spread)
'''
bollinger_spread = bollinger_strategy(test_spread, 2, 10, time_series, asset_pairs, coint_vector)
print(bollinger_spread.head(20))
'''



#plot_cointegration(asset_pairs, merged_data, coint_vector)
#plot_ratio(merged_data, asset_pairs)