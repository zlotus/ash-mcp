#!/usr/bin/env python3
"""
MCP Server for AkShare Financial Data.
Enhanced for Long-term Analysis and Grid Trading Strategy.
"""

import asyncio

from mcp.server.fastmcp import FastMCP

from models.inputs import (
    BaoStockCodeInput,
    BaoStockDividendInput,
    BaoStockIndexInput,
    BaoStockIndustryInput,
    BaoStockKDataInput,
    BaoStockQuarterInput,
    BaoStockValuationInput,
    DateInput,
    GridStrategyInput,
    LongTermFactorInput,
    LowFreqBacktestInput,
    PortfolioRebalanceInput,
    SymbolInput,
    ValueCandidatesGridInput,
)
from tools.fundamental import (
    get_financial_analysis_impl,
    get_long_term_factor_score_impl,
    get_valuation_status_impl,
)
from tools.market import (
    get_current_time_impl,
    get_dragon_tiger_list_impl,
    get_market_indices_impl,
    get_north_fund_flow_impl,
    get_sector_fund_flow_impl,
    get_stock_news_impl,
)
from tools.stock import (
    get_dividend_info_impl,
    get_growth_info_impl,
    get_index_data_impl,
    get_industry_info_impl,
    get_operation_info_impl,
    get_profit_info_impl,
    get_stock_basic_impl,
    get_stock_kdata_impl,
    get_stock_spot_impl,
    get_valuation_info_impl,
)
from tools.strategy import (
    get_grid_strategy_impl,
    get_low_freq_backtest_impl,
    get_portfolio_rebalance_plan_impl,
    get_value_candidates_and_grid_impl,
)

mcp = FastMCP("akshare_mcp")


@mcp.tool(name="get_current_time")
async def get_current_time() -> str:
    """获取当前服务器时间。命名规范: `get_*`。"""
    return await get_current_time_impl()


@mcp.tool(name="get_stock_basic")
async def get_stock_basic(params: BaoStockCodeInput) -> str:
    """获取证券基础信息（代码、名称、上市状态、行业）。数据源: baostock。"""
    return await get_stock_basic_impl(params)


@mcp.tool(name="get_stock_kdata")
async def get_stock_kdata(params: BaoStockKDataInput) -> str:
    """获取股票历史 K 线（支持日/周/月/分钟级频率与复权参数）。数据源: baostock。"""
    return await get_stock_kdata_impl(params)


@mcp.tool(name="get_industry_info")
async def get_industry_info(params: BaoStockIndustryInput) -> str:
    """获取行业分类信息。可按单只证券查询，也可查询全市场行业映射。数据源: baostock。"""
    return await get_industry_info_impl(params)


@mcp.tool(name="get_dividend_info")
async def get_dividend_info(params: BaoStockDividendInput) -> str:
    """获取分红派息数据（支持按年份与年份类型筛选）。数据源: baostock。"""
    return await get_dividend_info_impl(params)


@mcp.tool(name="get_profit_info")
async def get_profit_info(params: BaoStockQuarterInput) -> str:
    """获取季度盈利能力指标（如 ROE、毛利率、净利润）。数据源: baostock。"""
    return await get_profit_info_impl(params)


@mcp.tool(name="get_operation_info")
async def get_operation_info(params: BaoStockQuarterInput) -> str:
    """获取季度营运能力指标（周转率、周转天数等）。数据源: baostock。"""
    return await get_operation_info_impl(params)


@mcp.tool(name="get_growth_info")
async def get_growth_info(params: BaoStockQuarterInput) -> str:
    """获取季度成长能力指标（营收/利润同比等）。数据源: baostock。"""
    return await get_growth_info_impl(params)


@mcp.tool(name="get_index_data")
async def get_index_data(params: BaoStockIndexInput) -> str:
    """获取指数历史 K 线（任意指数代码、可选日/周/月）。数据源: baostock。"""
    return await get_index_data_impl(params)


@mcp.tool(name="get_valuation_info")
async def get_valuation_info(params: BaoStockValuationInput) -> str:
    """获取估值时序数据（PE/PB/PS/PCF），用于估值回溯分析。数据源: baostock。"""
    return await get_valuation_info_impl(params)


@mcp.tool(name="get_market_indices")
async def get_market_indices() -> str:
    """获取三大指数市场概览。数据源优先级: baostock -> akshare fallback。命名规范: `get_*`。"""
    return await get_market_indices_impl()


@mcp.tool(name="get_sector_fund_flow")
async def get_sector_fund_flow() -> str:
    """获取行业板块资金流排名。数据源: akshare。命名规范: `get_*`。"""
    return await get_sector_fund_flow_impl()


@mcp.tool(name="get_north_fund_flow")
async def get_north_fund_flow() -> str:
    """获取北向资金汇总。数据源: akshare。命名规范: `get_*`。"""
    return await get_north_fund_flow_impl()


@mcp.tool(name="get_stock_spot")
async def get_stock_spot(params: SymbolInput) -> str:
    """获取个股最新行情快照。数据源优先级: baostock -> akshare fallback。命名规范: `get_*`。"""
    return await get_stock_spot_impl(params)


@mcp.tool(name="get_financial_analysis")
async def get_financial_analysis(params: SymbolInput) -> str:
    """获取财务指标摘要（近年多季度聚合）。数据源优先级: baostock -> akshare fallback。命名规范: `get_*`。"""
    return await get_financial_analysis_impl(params)


@mcp.tool(name="get_valuation_status")
async def get_valuation_status(params: SymbolInput) -> str:
    """获取估值分位结论（5 年 PE 分位）。数据源优先级: baostock -> akshare fallback。命名规范: `get_*`。"""
    return await get_valuation_status_impl(params)


@mcp.tool(name="get_stock_news")
async def get_stock_news(params: SymbolInput) -> str:
    """获取个股新闻。数据源: akshare。命名规范: `get_*`。"""
    return await get_stock_news_impl(params)


@mcp.tool(name="get_dragon_tiger_list")
async def get_dragon_tiger_list(params: DateInput) -> str:
    """获取龙虎榜明细。数据源: akshare。命名规范: `get_*`。"""
    return await get_dragon_tiger_list_impl(params)


@mcp.tool(name="get_grid_strategy")
async def get_grid_strategy(params: GridStrategyInput) -> str:
    """获取网格交易计划。数据源优先级: baostock -> akshare fallback。命名规范: `get_*`。"""
    return await get_grid_strategy_impl(params)


@mcp.tool(name="get_long_term_factor_score")
async def get_long_term_factor_score(params: LongTermFactorInput) -> str:
    """获取长期持有多因子评分（估值/质量/成长/分红）。数据源优先级: baostock -> akshare fallback。"""
    return await get_long_term_factor_score_impl(params)


@mcp.tool(name="get_low_freq_backtest")
async def get_low_freq_backtest(params: LowFreqBacktestInput) -> str:
    """执行低频长期持有回测（月度定投 + 手续费），输出收益与回撤指标。数据源优先级: baostock -> akshare fallback。"""
    return await get_low_freq_backtest_impl(params)


@mcp.tool(name="get_portfolio_rebalance_plan")
async def get_portfolio_rebalance_plan(params: PortfolioRebalanceInput) -> str:
    """生成低频组合再平衡计划（等权/逆波动），并给出按当前权重的调仓方向。"""
    return await get_portfolio_rebalance_plan_impl(params)


@mcp.tool(name="get_value_candidates_and_grid")
async def get_value_candidates_and_grid(params: ValueCandidatesGridInput) -> str:
    """围绕锚点股票筛选同类低估标的，并输出每只股票的长期佛系网格计划。支持估值/质量/分红/市值相似度/波动率硬约束，并返回入选原因。"""
    return await get_value_candidates_and_grid_impl(params)


if __name__ == "__main__":
    import sys

    async def list_tools() -> None:
        print("AkShare MCP Server")
        print("Available Tools:")
        tools = await mcp.list_tools()
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

    if "--help" in sys.argv or "-h" in sys.argv:
        asyncio.run(list_tools())
    else:
        mcp.run()
