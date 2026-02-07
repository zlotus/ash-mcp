import baostock as bs
import pandas as pd
from data.baostock_client import ensure_baostock_login, query_baostock_table

ensure_baostock_login()
symbol = "sh.600000"
print("--- Raw Data for sh.600000 ---")

profit_df = query_baostock_table(bs.query_profit_data, code=symbol, year=2024, quarter=3)
print("Profit Data:")
print(profit_df.to_string())

balance_df = query_baostock_table(bs.query_balance_data, code=symbol, year=2024, quarter=3)
print("Balance Data:")
print(balance_df.to_string())

growth_df = query_baostock_table(bs.query_growth_data, code=symbol, year=2024, quarter=3)
print("Growth Data:")
print(growth_df.to_string())

bs.logout()