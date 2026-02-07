import baostock as bs
import pandas as pd
from data.baostock_client import ensure_baostock_login, query_baostock_table

ensure_baostock_login()
symbol = "sh.600519"
balance_df = query_baostock_table(bs.query_balance_data, code=symbol, year=2024, quarter=3)
print("Balance Data for sh.600519:")
print(balance_df.to_string())
bs.logout()
