from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


class KlinePeriod(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class AdjustType(str, Enum):
    NONE = ""
    QFQ = "qfq"
    HFQ = "hfq"


class RebalanceMethod(str, Enum):
    EQUAL = "equal"
    INVERSE_VOL = "inverse_vol"


def normalize_symbol(symbol: str) -> str:
    digits = "".join(filter(str.isdigit, symbol))
    if len(digits) == 6:
        return digits
    return symbol


class SymbolInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    symbol: str = Field(..., description="Stock symbol (e.g., '600519')")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return normalize_symbol(v)


class DateInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    date: Optional[str] = Field(default=None, description="Date (YYYYMMDD), defaults to today or latest available")


class BaoStockCodeInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    symbol: str = Field(..., description="Stock/index code. Supports 600519 or sh.600519")


class BaoStockKDataInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    symbol: str = Field(..., description="Stock code. Supports 600519 or sh.600519")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYYMMDD or YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYYMMDD or YYYY-MM-DD)")
    frequency: str = Field(default="d", description="kline frequency: d/w/m/5/15/30/60")
    adjustflag: str = Field(default="3", description="adjust flag: 1=hfq, 2=qfq, 3=none")
    limit: int = Field(default=1000, ge=1, le=5000, description="Max rows to return")


class BaoStockIndustryInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    symbol: Optional[str] = Field(default=None, description="Optional stock code. Empty means all stocks")
    limit: int = Field(default=200, ge=1, le=2000, description="Max rows for all-stock query")


class BaoStockDividendInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    symbol: str = Field(..., description="Stock code. Supports 600519 or sh.600519")
    year: Optional[str] = Field(default=None, description="Year like 2024. Empty means all years")
    year_type: str = Field(default="report", description="yearType: report/operate/dividend")
    limit: int = Field(default=300, ge=1, le=3000, description="Max rows to return")


class BaoStockQuarterInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    symbol: str = Field(..., description="Stock code. Supports 600519 or sh.600519")
    year: int = Field(..., ge=1990, le=2100, description="Fiscal year")
    quarter: int = Field(..., ge=1, le=4, description="Fiscal quarter: 1/2/3/4")


class BaoStockIndexInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    symbol: str = Field(..., description="Index code. Supports sh.000001 / sz.399001")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYYMMDD or YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYYMMDD or YYYY-MM-DD)")
    frequency: str = Field(default="d", description="kline frequency: d/w/m")
    limit: int = Field(default=1000, ge=1, le=5000, description="Max rows to return")


class BaoStockValuationInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    symbol: str = Field(..., description="Stock code. Supports 600519 or sh.600519")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYYMMDD or YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYYMMDD or YYYY-MM-DD)")
    frequency: str = Field(default="d", description="kline frequency: d/w/m")
    limit: int = Field(default=1000, ge=1, le=5000, description="Max rows to return")


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


class LongTermFactorInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    symbol: str = Field(..., description="Stock symbol (e.g., 600519)")
    valuation_lookback_years: int = Field(default=5, ge=3, le=15, description="Years for valuation percentile lookback")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return normalize_symbol(v)


class LowFreqBacktestInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    symbol: str = Field(..., description="Stock symbol")
    start_date: Optional[str] = Field(default=None, description="Backtest start date (YYYYMMDD or YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="Backtest end date (YYYYMMDD or YYYY-MM-DD)")
    initial_cash: float = Field(default=100000, gt=0, description="Initial capital")
    monthly_contribution: float = Field(default=3000, ge=0, description="Monthly cash contribution")
    fee_rate: float = Field(default=0.0005, ge=0, le=0.01, description="Transaction fee rate")
    adjust: AdjustType = Field(default=AdjustType.HFQ, description="Price adjustment mode")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return normalize_symbol(v)


class PortfolioRebalanceInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    symbols: str = Field(..., description="Comma-separated symbols, e.g. 600519,000858,601318")
    method: RebalanceMethod = Field(default=RebalanceMethod.INVERSE_VOL, description="Target weighting method")
    lookback_days: int = Field(default=252, ge=60, le=1500, description="Lookback days for risk estimation")
    current_weights: Optional[str] = Field(default=None, description="Current weights, e.g. 600519:0.4,000858:0.6")
    max_single_weight: float = Field(default=0.5, gt=0, le=1.0, description="Cap per symbol")


class ValueCandidatesGridInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    anchor_symbol: str = Field(default="601318", description="Anchor stock (e.g. 601318 for Ping An) to define the target scale and quality profile.")
    top_n: int = Field(default=3, ge=1, le=10, description="Number of picks to return")
    lookback_years: int = Field(default=5, ge=3, le=15, description="Years for valuation percentile lookback")
    candidate_limit: int = Field(default=50, ge=20, le=500, description="Max candidate stocks to evaluate from the pool.")
    history_days: int = Field(default=250, ge=60, le=1200, description="Lookback days for grid range")
    grid_step_pct: float = Field(default=0.05, ge=0.01, le=0.2, description="Grid step percentage")
    grid_count: int = Field(default=5, ge=1, le=10, description="Grid levels")
    strict_same_industry: bool = Field(default=False, description="If True, only search within anchor's industry. If False, searches across all blue-chips.")
    max_pe_percentile: float = Field(default=40.0, ge=1, le=100, description="Max allowed PE percentile (ensure undervalued).")
    max_pb_percentile: float = Field(default=50.0, ge=1, le=100, description="Max allowed PB percentile.")
    min_roe: float = Field(default=12.0, ge=-50, le=80, description="Minimum annual ROE(%) - quality floor.")
    max_debt_ratio: float = Field(default=75.0, ge=0, le=100, description="Maximum debt ratio(%). Note: Financials use a higher internal threshold automatically.")
    min_yoy_net_profit: float = Field(default=-10.0, ge=-300, le=500, description="Minimum YoY net profit growth(%) - stability filter.")
    min_dividend_years_5y: int = Field(default=4, ge=0, le=5, description="Minimum dividend-paying years in last 5 years - consistency check.")
    market_cap_ratio_min: float = Field(default=0.2, ge=0.0, le=10.0, description="Min market-cap ratio vs anchor - scale similarity.")
    market_cap_ratio_max: float = Field(default=5.0, ge=0.1, le=50.0, description="Max market-cap ratio vs anchor.")
    max_annual_volatility: float = Field(default=0.35, ge=0.05, le=2.0, description="Max annualized volatility - risk control.")

    @field_validator("anchor_symbol")
    @classmethod
    def validate_anchor_symbol(cls, v: str) -> str:
        return normalize_symbol(v)
