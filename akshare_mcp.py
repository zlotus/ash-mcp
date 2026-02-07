#!/usr/bin/env python3
"""
MCP Server for AkShare Financial Data.
Enhanced for Long-term Analysis and Grid Trading Strategy.
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Union

import akshare as ak
import baostock as bs
import pandas as pd
from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field, ConfigDict, field_validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("akshare_mcp")

# Initialize FastMCP
mcp = FastMCP("akshare_mcp")

# --- Constants & Enums ---

class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"

class KlinePeriod(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class AdjustType(str, Enum):
    NONE = ""
    QFQ = "qfq"  # Forward adjustment
    HFQ = "hfq"  # Backward adjustment - Recommended for long-term

BAOSTOCK_FREQUENCY_MAP = {
    KlinePeriod.DAILY.value: "d",
    KlinePeriod.WEEKLY.value: "w",
    KlinePeriod.MONTHLY.value: "m",
}
BAOSTOCK_ADJUST_MAP = {
    AdjustType.NONE.value: "3",  # no adjust
    AdjustType.QFQ.value: "2",   # qfq
    AdjustType.HFQ.value: "1",   # hfq
}
_baostock_logged_in = False

# --- Utility Functions ---

def normalize_symbol(symbol: str) -> str:
    """Normalize stock symbol to 6-digit format."""
    digits = "".join(filter(str.isdigit, symbol))
    if len(digits) == 6:
        return digits
    return symbol

def normalize_bs_code(symbol: str) -> str:
    """Convert symbol to baostock format, e.g. sh.600519 / sz.000001."""
    s = symbol.strip().lower()
    if s.startswith("sh.") or s.startswith("sz."):
        return s
    digits = normalize_symbol(s)
    market = "sh" if digits.startswith(("5", "6", "9")) else "sz"
    return f"{market}.{digits}"

def _parse_date_yyyymmdd(date_str: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD for baostock."""
    if not date_str:
        return ""
    if "-" in date_str:
        return date_str
    return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")

def _safe_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Try convert all columns to numeric where possible."""
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            continue
    return df

def ensure_baostock_login() -> None:
    """Initialize baostock session once."""
    global _baostock_logged_in
    if _baostock_logged_in:
        return
    try:
        login_result = bs.login()
    except Exception as e:
        raise RuntimeError(f"baostock login failed: {str(e)}")
    if getattr(login_result, "error_code", "-1") != "0":
        raise RuntimeError(f"baostock login failed: {getattr(login_result, 'error_msg', 'unknown error')}")
    _baostock_logged_in = True

def baostock_result_to_df(result: Any) -> pd.DataFrame:
    """Convert baostock query resultset to DataFrame."""
    if getattr(result, "error_code", "-1") != "0":
        raise RuntimeError(getattr(result, "error_msg", "baostock query failed"))
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)

def query_baostock_history(
    symbol: str,
    fields: List[str],
    start_date: str,
    end_date: str,
    period: str,
    adjust: str
) -> pd.DataFrame:
    """Query historical kline from baostock."""
    ensure_baostock_login()
    rs = bs.query_history_k_data_plus(
        normalize_bs_code(symbol),
        ",".join(fields),
        start_date=_parse_date_yyyymmdd(start_date),
        end_date=_parse_date_yyyymmdd(end_date),
        frequency=BAOSTOCK_FREQUENCY_MAP[period],
        adjustflag=BAOSTOCK_ADJUST_MAP[adjust],
    )
    df = baostock_result_to_df(rs)
    if not df.empty:
        df = _safe_to_numeric(df)
    return df

def format_data(df: pd.DataFrame, format: ResponseFormat = ResponseFormat.MARKDOWN, title: str = "") -> str:
    """Format DataFrame as Markdown or JSON string."""
    if df.empty:
        return "No data found."
    
    if format == ResponseFormat.JSON:
        return df.to_json(orient="records", force_ascii=False)
    else:
        md = ""
        if title:
            md += f"### {title}\n\n"
        md += df.to_markdown(index=False)
        return md

def trim_dataframe(df: pd.DataFrame, max_rows: int = 20, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Trim DataFrame to save tokens."""
    if columns:
        existing_cols = [c for c in columns if c in df.columns]
        df = df[existing_cols]
    
    if len(df) > max_rows:
        return df.head(max_rows)
    return df

def handle_error(e: Exception, tool_name: str) -> str:
    """Consistent error handling."""
    logger.error(f"Error in {tool_name}: {str(e)}", exc_info=True)
    return f"Error: Data source temporarily unavailable or invalid request. Details: {str(e)}"

# --- Input Models ---

class SymbolInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    symbol: str = Field(..., description="Stock symbol (e.g., '600519')")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return normalize_symbol(v)

class KlineInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    symbol: str = Field(..., description="Stock symbol")
    period: KlinePeriod = Field(default=KlinePeriod.DAILY, description="Kline period: daily, weekly, or monthly")
    adjust: AdjustType = Field(default=AdjustType.HFQ, description="Adjustment: hfq (Recommended, for long-term), qfq, , or empty")
    limit: int = Field(default=1000, description="Number of rows to return (e.g. 1000 for ~4 years of daily data)", ge=1, le=5000)
    start_date: Optional[str] = Field(default=None, description="Start date (YYYYMMDD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYYMMDD)")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return normalize_symbol(v)

class DateInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    date: Optional[str] = Field(default=None, description="Date (YYYYMMDD), defaults to today or latest available")

class GridStrategyInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    symbol: str = Field(..., description="Stock symbol")
    current_price: float = Field(..., description="Current market price")
    history_days: int = Field(default=250, description="Days to look back for range calculation", ge=30, le=2500)
    grid_step_pct: float = Field(default=0.05, description="Percentage interval between grids (e.g. 0.05 for 5%)", ge=0.01, le=0.2)
    grid_count: int = Field(default=5, description="Number of grid levels to generate", ge=1, le=10)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return normalize_symbol(v)

# --- Tool Definitions ---

# 1. Market Overview

@mcp.tool(name="get_market_indices")
async def get_market_indices() -> str:
    """
    Get real-time spot prices for major Chinese stock indices (SSE, SZSE, etc.).
    Useful for judging overall market sentiment.
    """
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
            df = df[['代码', '名称', '最新价', '涨跌额', '涨跌幅', '昨收', '今开', '最高', '最低', '成交量', '成交额']]
            return format_data(df, title="Major Market Indices (Sina fallback)")
        except Exception as e:
            return handle_error(e, "get_market_indices")

@mcp.tool(name="get_sector_fund_flow")
async def get_sector_fund_flow() -> str:
    """
    Get industry sector fund flow rankings. 
    Identifies which sectors are currently attracting or losing capital.
    """
    try:
        df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
        df = trim_dataframe(df, max_rows=10, columns=["名称", "今日净额", "今日涨跌幅", "今日主力净流入最大股"])
        return format_data(df, title="Industry Sector Fund Flow (Top 10)")
    except Exception as e:
        return handle_error(e, "get_sector_fund_flow")

@mcp.tool(name="get_north_fund_flow")
async def get_north_fund_flow() -> str:
    """
    Get Northbound (HSGT) fund flow summary.
    Shows foreign capital movement into A-shares.
    """
    try:
        df = ak.stock_hsgt_fund_flow_summary_em()
        return format_data(df, title="Northbound Fund Flow Summary (EM)")
    except Exception as e:
        return handle_error(e, "get_north_fund_flow")

# 2. Stock Deep Dive

@mcp.tool(name="get_stock_spot")
async def get_stock_spot(params: SymbolInput) -> str:
    """Get real-time spot price and basic info for a specific stock."""
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

@mcp.tool(name="get_stock_history_kline")
async def get_stock_history_kline(params: KlineInput) -> str:
    """
    Get historical kline data with adjustment options and long-term support.
    Use adjust='hfq' for accurate long-term trend analysis.
    """
    try:
        start_date = params.start_date or (datetime.now() - timedelta(days=params.limit * 1.5)).strftime("%Y%m%d")
        end_date = params.end_date or datetime.now().strftime("%Y%m%d")

        fields = ["date", "code", "open", "high", "low", "close", "volume", "amount", "pctChg"]
        if params.period == KlinePeriod.DAILY:
            fields.extend(["turn", "peTTM", "pbMRQ", "isST"])
        df = query_baostock_history(
            symbol=params.symbol,
            fields=fields,
            start_date=start_date,
            end_date=end_date,
            period=params.period.value,
            adjust=params.adjust.value,
        )
        df = df.tail(params.limit)
        return format_data(df, title=f"History Kline ({params.period.value}, adjust={params.adjust.value}) for {params.symbol} (baostock)")
    except Exception:
        try:
            df = ak.stock_zh_a_hist(
                symbol=params.symbol,
                period=params.period.value,
                start_date=start_date,
                end_date=end_date,
                adjust=params.adjust.value
            )
            df = df.tail(params.limit)
            return format_data(df, title=f"History Kline ({params.period.value}, adjust={params.adjust.value}) for {params.symbol} (AKShare fallback)")
        except Exception as e:
            return handle_error(e, "get_stock_history_kline")

# 3. Fundamental & Valuation

@mcp.tool(name="get_financial_analysis")
async def get_financial_analysis(params: SymbolInput) -> str:
    """
    Get key financial indicators for the last few years.
    Useful for value investment analysis.
    """
    try:
        ensure_baostock_login()
        code = normalize_bs_code(params.symbol)
        current_year = datetime.now().year
        merged_rows: List[Dict[str, Any]] = []
        for year in range(current_year - 5, current_year + 1):
            for quarter in [1, 2, 3, 4]:
                try:
                    profit_rs = bs.query_profit_data(code=code, year=year, quarter=quarter)
                    profit_df = baostock_result_to_df(profit_rs)
                    if profit_df.empty:
                        continue
                    balance_rs = bs.query_balance_data(code=code, year=year, quarter=quarter)
                    balance_df = baostock_result_to_df(balance_rs)
                    growth_rs = bs.query_growth_data(code=code, year=year, quarter=quarter)
                    growth_df = baostock_result_to_df(growth_rs)
                    row = profit_df.iloc[0].to_dict()
                    if not balance_df.empty:
                        row.update(balance_df.iloc[0].to_dict())
                    if not growth_df.empty:
                        row.update(growth_df.iloc[0].to_dict())
                    merged_rows.append(row)
                except Exception:
                    continue
        if not merged_rows:
            raise RuntimeError("No financial data from baostock.")
        raw_df = pd.DataFrame(merged_rows)
        raw_df = _safe_to_numeric(raw_df)

        def first_existing(row: pd.Series, keys: List[str]) -> Any:
            for key in keys:
                if key in row and pd.notna(row[key]) and row[key] != "":
                    return row[key]
            return None

        cleaned_rows = []
        for _, row in raw_df.iterrows():
            cleaned_rows.append({
                "日期": first_existing(row, ["statDate", "pubDate"]),
                "净资产收益率(%)": first_existing(row, ["roeAvg"]),
                "销售毛利率(%)": first_existing(row, ["gpMargin"]),
                "销售净利率(%)": first_existing(row, ["npMargin"]),
                "营业收入(万元)": first_existing(row, ["MBRevenue"]),
                "净利润(万元)": first_existing(row, ["netProfit"]),
                "资产负债率(%)": first_existing(row, ["liabilityToAsset"]),
                "净利润同比增长率(%)": first_existing(row, ["YOYNI"]),
            })

        df = pd.DataFrame(cleaned_rows).dropna(how="all")
        df = df.sort_values(by="日期", ascending=False)
        df = trim_dataframe(df, max_rows=8)
        return format_data(df, title=f"Financial Indicators for {params.symbol} (baostock)")
    except Exception:
        try:
            df = ak.stock_financial_analysis_indicator(symbol=params.symbol)
            cols = ["日期", "净资产收益率(%)", "销售毛利率(%)", "销售净利率(%)", "总资产净利率(%)", "营业收入(万元)", "净利润(万元)", "资产负债率(%)"]
            df = trim_dataframe(df, max_rows=8, columns=cols)
            return format_data(df, title=f"Financial Indicators for {params.symbol} (AKShare fallback)")
        except Exception as e:
            return handle_error(e, "get_financial_analysis")

@mcp.tool(name="get_dividend_history")
async def get_dividend_history(params: SymbolInput) -> str:
    """
    Get historical dividend payment records for a specific stock.
    Uses JuChao (CnInfo) as the data source.
    """
    try:
        ensure_baostock_login()
        code = normalize_bs_code(params.symbol)
        current_year = datetime.now().year
        frames = []
        for year in range(current_year - 10, current_year + 1):
            rs = bs.query_dividend_data(code=code, year=str(year), yearType="report")
            df_y = baostock_result_to_df(rs)
            if not df_y.empty:
                frames.append(df_y)
        if not frames:
            raise RuntimeError("No dividend data from baostock.")
        df = pd.concat(frames, ignore_index=True)
        df = _safe_to_numeric(df)
        df = df.sort_values(by="dividPlanDate", ascending=False) if "dividPlanDate" in df.columns else df
        return format_data(df, title=f"Dividend History for {params.symbol} (baostock)")
    except Exception:
        try:
            df = ak.stock_dividend_cninfo(symbol=params.symbol)
            return format_data(df, title=f"Dividend History for {params.symbol} (AKShare fallback)")
        except Exception as e:
            return handle_error(e, "get_dividend_history")

@mcp.tool(name="get_valuation_status")
async def get_valuation_status(params: SymbolInput) -> str:
    """
    Check if the stock is currently overvalued or undervalued based on historical PE (TTM) data.
    Provides current value and percentile rank.
    """
    try:
        start_date = (datetime.now() - timedelta(days=365 * 5 + 20)).strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")
        df_pe = query_baostock_history(
            symbol=params.symbol,
            fields=["date", "code", "peTTM"],
            start_date=start_date,
            end_date=end_date,
            period=KlinePeriod.DAILY.value,
            adjust=AdjustType.NONE.value,
        )
        if df_pe.empty or "peTTM" not in df_pe.columns:
            return "Valuation data not available."

        pe_series = pd.to_numeric(df_pe["peTTM"], errors="coerce")
        pe_series = pe_series[(pe_series > 0) & pe_series.notna()]
        if pe_series.empty:
            return "Valuation data not available."

        current_val = pe_series.iloc[-1]
        history_min = pe_series.min()
        history_max = pe_series.max()
        percentile = (pe_series <= current_val).mean() * 100
        
        status = "Fairly Valued"
        if percentile < 20:
            status = "Undervalued (Safe Margin High)"
        elif percentile > 80:
            status = "Overvalued (Risk High)"
        
        summary = pd.DataFrame([{
            "Symbol": params.symbol,
            "Current PE (TTM)": round(current_val, 2),
            "5Y Low": round(history_min, 2),
            "5Y High": round(history_max, 2),
            "Percentile (%)": round(percentile, 2),
            "Status": status
        }])
        
        return format_data(summary, title=f"Valuation Status (5-Year PE) for {params.symbol} (baostock)")
    except Exception:
        try:
            df_pe = ak.stock_zh_valuation_baidu(symbol=params.symbol, indicator="市盈率(TTM)", period="近5年")
            if df_pe.empty:
                return "Valuation data not available."
            current_val = df_pe['value'].iloc[-1]
            history_min = df_pe['value'].min()
            history_max = df_pe['value'].max()
            percentile = (df_pe['value'] <= current_val).mean() * 100
            status = "Fairly Valued"
            if percentile < 20:
                status = "Undervalued (Safe Margin High)"
            elif percentile > 80:
                status = "Overvalued (Risk High)"
            summary = pd.DataFrame([{
                "Symbol": params.symbol,
                "Current PE (TTM)": round(current_val, 2),
                "5Y Low": round(history_min, 2),
                "5Y High": round(history_max, 2),
                "Percentile (%)": round(percentile, 2),
                "Status": status
            }])
            return format_data(summary, title=f"Valuation Status (5-Year PE) for {params.symbol} (AKShare fallback)")
        except Exception as e:
            return handle_error(e, "get_valuation_status")

# 4. News & Alpha

@mcp.tool(name="get_stock_news")
async def get_stock_news(params: SymbolInput) -> str:
    """
    Get recent news for a specific stock from Eastmoney.
    """
    try:
        df = ak.stock_news_em(symbol=params.symbol)
        df = trim_dataframe(df, max_rows=5)
        return format_data(df, title=f"Recent News for {params.symbol}")
    except Exception as e:
        return handle_error(e, "get_stock_news")

@mcp.tool(name="get_dragon_tiger_list")
async def get_dragon_tiger_list(params: DateInput) -> str:
    """
    Get Dragon and Tiger list (LHB) for a specific date from Eastmoney.
    Shows stocks with unusual trading activity and institutional movement.
    """
    try:
        date = params.date or datetime.now().strftime("%Y%m%d")
        # Note: akshare function param name might vary, detail_em uses 'date'
        df = ak.stock_lhb_detail_em(start_date=date)
        df = trim_dataframe(df, max_rows=15)
        return format_data(df, title=f"Dragon Tiger List for {date}")
    except Exception as e:
        return handle_error(e, "get_dragon_tiger_list")

# 5. Strategy Tools

@mcp.tool(name="calculate_grid_strategy")
async def calculate_grid_strategy(params: GridStrategyInput) -> str:
    """
    Generate a grid trading plan based on historical price ranges.
    Designed for low-frequency, 'Buddhist-style' trading.
    """
    try:
        # Use 3-year history for broader perspective if history_days is large
        start_date = (datetime.now() - timedelta(days=params.history_days * 1.5)).strftime("%Y%m%d")
        df = query_baostock_history(
            symbol=params.symbol,
            fields=["date", "code", "high", "low", "close"],
            start_date=start_date,
            end_date=datetime.now().strftime("%Y%m%d"),
            period=KlinePeriod.DAILY.value,
            adjust=AdjustType.HFQ.value,
        )
        df = df.tail(params.history_days)
        
        high_col = "high" if "high" in df.columns else "最高"
        low_col = "low" if "low" in df.columns else "最低"
        year_high = pd.to_numeric(df[high_col], errors="coerce").max()
        year_low = pd.to_numeric(df[low_col], errors="coerce").min()
        
        grids = []
        p = params.current_price
        for i in range(1, params.grid_count + 1):
            buy_price = p * (1 - params.grid_step_pct)
            sell_target = buy_price * (1 + params.grid_step_pct)
            grids.append({
                "Level": f"Buy #{i}",
                "Price": round(buy_price, 2),
                "Sell Target": round(sell_target, 2),
                "Distance from Current": f"-{round(i * params.grid_step_pct * 100, 1)}%"
            })
            p = buy_price
            
        summary_df = pd.DataFrame(grids)
        header = f"Grid Strategy for {params.symbol}\n"
        header += f"- Current Price: {params.current_price}\n"
        header += f"- {params.history_days}-Day Range (hfq): {year_low} - {year_high}\n"
        header += f"- Grid Step: {params.grid_step_pct*100}%\n"
        
        return header + "\n" + format_data(summary_df)
    except Exception:
        try:
            start_date = (datetime.now() - timedelta(days=params.history_days * 1.5)).strftime("%Y%m%d")
            df = ak.stock_zh_a_hist(
                symbol=params.symbol,
                period="daily",
                start_date=start_date,
                adjust="hfq"
            )
            df = df.tail(params.history_days)
            year_high = df['最高'].max()
            year_low = df['最低'].min()
            grids = []
            p = params.current_price
            for i in range(1, params.grid_count + 1):
                buy_price = p * (1 - params.grid_step_pct)
                sell_target = buy_price * (1 + params.grid_step_pct)
                grids.append({
                    "Level": f"Buy #{i}",
                    "Price": round(buy_price, 2),
                    "Sell Target": round(sell_target, 2),
                    "Distance from Current": f"-{round(i * params.grid_step_pct * 100, 1)}%"
                })
                p = buy_price
            summary_df = pd.DataFrame(grids)
            header = f"Grid Strategy for {params.symbol}\n"
            header += f"- Current Price: {params.current_price}\n"
            header += f"- {params.history_days}-Day Range (hfq): {year_low} - {year_high}\n"
            header += f"- Grid Step: {params.grid_step_pct*100}%\n"
            return header + "\n" + format_data(summary_df)
        except Exception as e:
            return handle_error(e, "calculate_grid_strategy")

if __name__ == "__main__":
    import sys
    async def list_tools():
        print("AkShare MCP Server")
        print("Available Tools:")
        tools = await mcp.list_tools()
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

    if "--help" in sys.argv or "-h" in sys.argv:
        asyncio.run(list_tools())
    else:
        mcp.run()
