from datetime import datetime, timedelta

import akshare as ak
import pandas as pd

from core.utils import format_data, handle_error, trim_dataframe
from data.baostock_client import query_baostock_history
from models.inputs import AdjustType, DateInput, KlinePeriod, SymbolInput


async def get_current_time_impl() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def get_market_indices_impl() -> str:
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=20)).strftime("%Y%m%d")
        index_meta = {
            "sh.000001": "上证指数",
            "sz.399001": "深证成指",
            "sz.399006": "创业板指",
        }
        rows = []
        for code, name in index_meta.items():
            df_idx = query_baostock_history(
                symbol=code,
                fields=["date", "code", "open", "high", "low", "close", "preclose", "pctChg", "volume", "amount"],
                start_date=start_date,
                end_date=end_date,
                period=KlinePeriod.DAILY.value,
                adjust=AdjustType.NONE.value,
            )
            if df_idx.empty:
                continue
            latest = df_idx.iloc[-1]
            rows.append({
                "代码": latest.get("code", code),
                "名称": name,
                "日期": latest.get("date", ""),
                "最新价": latest.get("close", None),
                "涨跌额": (latest.get("close", 0) - latest.get("preclose", 0)) if pd.notna(latest.get("preclose", None)) else None,
                "涨跌幅": latest.get("pctChg", None),
                "昨收": latest.get("preclose", None),
                "今开": latest.get("open", None),
                "最高": latest.get("high", None),
                "最低": latest.get("low", None),
                "成交量": latest.get("volume", None),
                "成交额": latest.get("amount", None),
            })
        df = pd.DataFrame(rows)
        return format_data(df, title="Major Market Indices (baostock)")
    except Exception:
        try:
            df = ak.stock_zh_index_spot_sina()
            major_indices = ["sh000001", "sz399001", "sz399006"]
            df = df[df["代码"].isin(major_indices)]
            df = df[["代码", "名称", "最新价", "涨跌额", "涨跌幅", "昨收", "今开", "最高", "最低", "成交量", "成交额"]]
            return format_data(df, title="Major Market Indices (Sina fallback)")
        except Exception as e:
            return handle_error(e, "get_market_indices")


async def get_sector_fund_flow_impl() -> str:
    try:
        df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
        df = trim_dataframe(df, max_rows=10, columns=["名称", "今日净额", "今日涨跌幅", "今日主力净流入最大股"])
        return format_data(df, title="Industry Sector Fund Flow (Top 10)")
    except Exception as e:
        return handle_error(e, "get_sector_fund_flow")


async def get_north_fund_flow_impl() -> str:
    try:
        df = ak.stock_hsgt_fund_flow_summary_em()
        return format_data(df, title="Northbound Fund Flow Summary (EM)")
    except Exception as e:
        return handle_error(e, "get_north_fund_flow")


async def get_stock_news_impl(params: SymbolInput) -> str:
    try:
        df = ak.stock_news_em(symbol=params.symbol)
        df = trim_dataframe(df, max_rows=5)
        return format_data(df, title=f"Recent News for {params.symbol}")
    except Exception as e:
        return handle_error(e, "get_stock_news")


async def get_dragon_tiger_list_impl(params: DateInput) -> str:
    try:
        date = params.date or datetime.now().strftime("%Y%m%d")
        df = ak.stock_lhb_detail_em(start_date=date)
        df = trim_dataframe(df, max_rows=15)
        return format_data(df, title=f"Dragon Tiger List for {date}")
    except Exception as e:
        return handle_error(e, "get_dragon_tiger_list")
