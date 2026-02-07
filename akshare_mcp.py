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
import pandas as pd
import numpy as np
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

# --- Utility Functions ---

def normalize_symbol(symbol: str) -> str:
    """Normalize stock symbol to 6-digit format."""
    digits = "".join(filter(str.isdigit, symbol))
    if len(digits) == 6:
        return digits
    return symbol

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
        df = ak.stock_zh_index_spot_sina()
        # Filter for major indices
        major_indices = ["sh000001", "sz399001", "sz399006"]
        df = df[df["代码"].isin(major_indices)]
        df = df[['代码', '名称', '最新价', '涨跌额', '涨跌幅', '昨收', '今开', '最高', '最低', '成交量', '成交额']]
        return format_data(df, title="Major Market Indices (Sina)")
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
        df = ak.stock_zh_a_spot_em()
        stock_data = df[df["代码"] == params.symbol]
        if stock_data.empty:
            return f"Stock code {params.symbol} not found."
        
        cols = ["代码", "名称", "最新价", "涨跌幅", "成交量", "成交额", "振幅", "最高", "最低", "今开", "昨收", "换手率", "市盈率-动态", "市净率"]
        stock_data = trim_dataframe(stock_data, columns=cols)
        return format_data(stock_data, title=f"Spot Info for {params.symbol}")
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
        
        df = ak.stock_zh_a_hist(
            symbol=params.symbol,
            period=params.period.value,
            start_date=start_date,
            end_date=end_date,
            adjust=params.adjust.value
        )
        
        df = df.tail(params.limit)
        return format_data(df, title=f"History Kline ({params.period.value}, adjust={params.adjust.value}) for {params.symbol}")
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
        df = ak.stock_financial_analysis_indicator(symbol=params.symbol)
        # Typically rows are reports, columns are indicators
        # Keep recent few years/quarters
        cols = ["日期", "净资产收益率(%)", "销售毛利率(%)", "销售净利率(%)", "总资产净利率(%)", "营业收入(万元)", "净利润(万元)", "资产负债率(%)"]
        df = trim_dataframe(df, max_rows=8, columns=cols)
        return format_data(df, title=f"Financial Indicators for {params.symbol}")
    except Exception as e:
        return handle_error(e, "get_financial_analysis")

@mcp.tool(name="get_dividend_history")
async def get_dividend_history(params: SymbolInput) -> str:
    """
    Get historical dividend payment records for a specific stock.
    Uses JuChao (CnInfo) as the data source.
    """
    try:
        df = ak.stock_dividend_cninfo(symbol=params.symbol)
        return format_data(df, title=f"Dividend History for {params.symbol}")
    except Exception as e:
        return handle_error(e, "get_dividend_history")

@mcp.tool(name="get_valuation_status")
async def get_valuation_status(params: SymbolInput) -> str:
    """
    Check if the stock is currently overvalued or undervalued based on historical PE (TTM) data.
    Provides current value and percentile rank.
    """
    try:
        # Get 5-year PE data from Baidu
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
        
        return format_data(summary, title=f"Valuation Status (5-Year PE) for {params.symbol}")
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