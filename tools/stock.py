from datetime import datetime, timedelta

import akshare as ak
import baostock as bs
import pandas as pd

from core.utils import format_data, handle_error, normalize_bs_code, trim_dataframe
from data.baostock_client import query_baostock_history, query_baostock_table
from models.inputs import (
    AdjustType,
    BaoStockCodeInput,
    BaoStockDividendInput,
    BaoStockIndexInput,
    BaoStockIndustryInput,
    BaoStockKDataInput,
    BaoStockQuarterInput,
    BaoStockValuationInput,
    KlinePeriod,
    SymbolInput,
)


async def get_stock_basic_impl(params: BaoStockCodeInput) -> str:
    try:
        code = normalize_bs_code(params.symbol)
        basic_df = query_baostock_table(bs.query_stock_basic, code=code)
        if basic_df.empty:
            return f"Stock code {params.symbol} not found."

        row = basic_df.iloc[0].to_dict()
        try:
            industry_df = query_baostock_table(bs.query_stock_industry, code=code)
            if not industry_df.empty:
                industry_row = industry_df.iloc[0].to_dict()
                row.update({
                    "industry": industry_row.get("industry", row.get("industry")),
                    "industryClassification": industry_row.get("industryClassification", row.get("industryClassification")),
                    "industryUpdateDate": industry_row.get("updateDate", ""),
                })
        except Exception:
            pass

        return format_data(pd.DataFrame([row]), title=f"Stock Basic Info for {params.symbol} (baostock)")
    except Exception as e:
        return handle_error(e, "get_stock_basic")


async def get_stock_kdata_impl(params: BaoStockKDataInput) -> str:
    try:
        start_date = params.start_date or (datetime.now() - timedelta(days=365 * 2)).strftime("%Y%m%d")
        end_date = params.end_date or datetime.now().strftime("%Y%m%d")
        fields = [
            "date", "code", "open", "high", "low", "close", "volume", "amount",
            "adjustflag", "turn", "tradestatus", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM",
        ]
        df = query_baostock_history(
            symbol=params.symbol,
            fields=fields,
            start_date=start_date,
            end_date=end_date,
            period=params.frequency,
            adjust=params.adjustflag,
        )
        df = df.tail(params.limit)
        return format_data(df, title=f"Stock KData for {params.symbol} (freq={params.frequency}, adjust={params.adjustflag})")
    except Exception as e:
        return handle_error(e, "get_stock_kdata")


async def get_industry_info_impl(params: BaoStockIndustryInput) -> str:
    try:
        if params.symbol:
            df = query_baostock_table(bs.query_stock_industry, code=normalize_bs_code(params.symbol))
        else:
            df = query_baostock_table(bs.query_stock_industry)
            df = trim_dataframe(df, max_rows=params.limit)
        return format_data(df, title="Industry Info (baostock)")
    except Exception as e:
        return handle_error(e, "get_industry_info")


async def get_dividend_info_impl(params: BaoStockDividendInput) -> str:
    try:
        df = query_baostock_table(
            bs.query_dividend_data,
            code=normalize_bs_code(params.symbol),
            year=params.year or "",
            yearType=params.year_type,
        )
        df = trim_dataframe(df, max_rows=params.limit)
        return format_data(df, title=f"Dividend Info for {params.symbol} (baostock)")
    except Exception as e:
        return handle_error(e, "get_dividend_info")


async def get_profit_info_impl(params: BaoStockQuarterInput) -> str:
    try:
        df = query_baostock_table(
            bs.query_profit_data,
            code=normalize_bs_code(params.symbol),
            year=params.year,
            quarter=params.quarter,
        )
        return format_data(df, title=f"Profit Info for {params.symbol} Y{params.year}Q{params.quarter} (baostock)")
    except Exception as e:
        return handle_error(e, "get_profit_info")


async def get_operation_info_impl(params: BaoStockQuarterInput) -> str:
    try:
        df = query_baostock_table(
            bs.query_operation_data,
            code=normalize_bs_code(params.symbol),
            year=params.year,
            quarter=params.quarter,
        )
        return format_data(df, title=f"Operation Info for {params.symbol} Y{params.year}Q{params.quarter} (baostock)")
    except Exception as e:
        return handle_error(e, "get_operation_info")


async def get_growth_info_impl(params: BaoStockQuarterInput) -> str:
    try:
        df = query_baostock_table(
            bs.query_growth_data,
            code=normalize_bs_code(params.symbol),
            year=params.year,
            quarter=params.quarter,
        )
        return format_data(df, title=f"Growth Info for {params.symbol} Y{params.year}Q{params.quarter} (baostock)")
    except Exception as e:
        return handle_error(e, "get_growth_info")


async def get_index_data_impl(params: BaoStockIndexInput) -> str:
    try:
        start_date = params.start_date or (datetime.now() - timedelta(days=365 * 2)).strftime("%Y%m%d")
        end_date = params.end_date or datetime.now().strftime("%Y%m%d")
        df = query_baostock_history(
            symbol=params.symbol,
            fields=["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"],
            start_date=start_date,
            end_date=end_date,
            period=params.frequency,
            adjust="3",
        )
        df = df.tail(params.limit)
        return format_data(df, title=f"Index Data for {params.symbol} (freq={params.frequency}, baostock)")
    except Exception as e:
        return handle_error(e, "get_index_data")


async def get_valuation_info_impl(params: BaoStockValuationInput) -> str:
    try:
        start_date = params.start_date or (datetime.now() - timedelta(days=365 * 2)).strftime("%Y%m%d")
        end_date = params.end_date or datetime.now().strftime("%Y%m%d")
        df = query_baostock_history(
            symbol=params.symbol,
            fields=["date", "code", "close", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"],
            start_date=start_date,
            end_date=end_date,
            period=params.frequency,
            adjust="3",
        )
        df = df.tail(params.limit)
        return format_data(df, title=f"Valuation Info for {params.symbol} (freq={params.frequency}, baostock)")
    except Exception as e:
        return handle_error(e, "get_valuation_info")


async def get_stock_spot_impl(params: SymbolInput) -> str:
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=20)).strftime("%Y%m%d")
        df = query_baostock_history(
            symbol=params.symbol,
            fields=["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg", "peTTM", "pbMRQ", "isST"],
            start_date=start_date,
            end_date=end_date,
            period=KlinePeriod.DAILY.value,
            adjust=AdjustType.QFQ.value,
        )
        if df.empty:
            return f"Stock code {params.symbol} not found."
        latest = df.iloc[-1]
        stock_data = pd.DataFrame([{
            "代码": latest.get("code", ""),
            "日期": latest.get("date", ""),
            "最新价": latest.get("close", None),
            "涨跌幅": latest.get("pctChg", None),
            "成交量": latest.get("volume", None),
            "成交额": latest.get("amount", None),
            "最高": latest.get("high", None),
            "最低": latest.get("low", None),
            "今开": latest.get("open", None),
            "昨收": latest.get("preclose", None),
            "换手率": latest.get("turn", None),
            "市盈率-动态": latest.get("peTTM", None),
            "市净率": latest.get("pbMRQ", None),
            "是否ST": latest.get("isST", None),
        }])
        return format_data(stock_data, title=f"Spot Info for {params.symbol} (baostock)")
    except Exception:
        try:
            df = ak.stock_zh_a_spot_em()
            stock_data = df[df["代码"] == params.symbol]
            if stock_data.empty:
                return f"Stock code {params.symbol} not found."
            cols = ["代码", "名称", "最新价", "涨跌幅", "成交量", "成交额", "振幅", "最高", "最低", "今开", "昨收", "换手率", "市盈率-动态", "市净率"]
            stock_data = trim_dataframe(stock_data, columns=cols)
            return format_data(stock_data, title=f"Spot Info for {params.symbol} (AKShare fallback)")
        except Exception as e:
            return handle_error(e, "get_stock_spot")
