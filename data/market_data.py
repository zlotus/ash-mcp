import akshare as ak
import pandas as pd

from data.baostock_client import query_baostock_history
from models.inputs import AdjustType, KlinePeriod, normalize_symbol


def fetch_daily_close_series(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = AdjustType.HFQ.value,
) -> pd.Series:
    try:
        df = query_baostock_history(
            symbol=symbol,
            fields=["date", "code", "close"],
            start_date=start_date,
            end_date=end_date,
            period=KlinePeriod.DAILY.value,
            adjust=adjust,
        )
        if df.empty or "close" not in df.columns:
            raise RuntimeError("empty close series from baostock")
        series = pd.to_numeric(df["close"], errors="coerce")
        dates = pd.to_datetime(df["date"], errors="coerce")
        result = pd.Series(series.values, index=dates).dropna().sort_index()
        if result.empty:
            raise RuntimeError("invalid close series from baostock")
        return result
    except Exception:
        df = ak.stock_zh_a_hist(
            symbol=normalize_symbol(symbol),
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        if df.empty:
            raise RuntimeError("empty close series from akshare")
        close_col = "收盘" if "收盘" in df.columns else "close"
        date_col = "日期" if "日期" in df.columns else "date"
        series = pd.to_numeric(df[close_col], errors="coerce")
        dates = pd.to_datetime(df[date_col], errors="coerce")
        result = pd.Series(series.values, index=dates).dropna().sort_index()
        if result.empty:
            raise RuntimeError("invalid close series from akshare")
        return result
