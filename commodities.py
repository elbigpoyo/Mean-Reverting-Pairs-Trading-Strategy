import pandas as pd
from stationarity_test import johansen_test, adf_test

commodities_table = pd.read_csv("commodities.csv")
print(commodities_table)

ticker_list = commodities_table["YahooFuturesTicker"].tolist()

column_names = ["commodity 1", "commodity 2", "comparison"]

final = pd.DataFrame(columns=column_names)

start_date = "2025-01-01"
end_date = "2025-10-01"

for i in range(len(ticker_list)):
    for j in range(i + 1, len(ticker_list)):
        final.loc[len(final)] = [
            ticker_list[i],
            ticker_list[j],
            johansen_test(ticker_list[i], ticker_list[j], start_date, end_date),
        ]

print(final)
