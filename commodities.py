import pandas as pd
from stationarity_test import adf_test as adf

commodities_table = pd.read_csv("commodities.csv")
print(commodities_table)

ticker_list = commodities_table["YahooFuturesTicker"].tolist()

column_names = ["commodity 1", "commodity 2", "p-value"]

final = pd.DataFrame(columns = column_names)

start_date = '2025-01-01'
end_date = '2025-10-01'

for i in range(len(ticker_list)):
    for j in range(i + 1, len(ticker_list)):
        final.loc[len(final)]= [ticker_list[i], ticker_list[j], adf(ticker_list[i], ticker_list[j], start_date, end_date)]

final_sorted = final.sort_values(by = "p-value")
print(final_sorted.head(20))