from datetime import datetime, timedelta
from typing import Any, Dict, List

import akshare as ak
import baostock as bs
import pandas as pd

from core.utils import (
    clamp_score,
    format_data,
    handle_error,
    normalize_bs_code,
    percentile_of_last,
    safe_to_numeric,
    trim_dataframe,
)
from data.baostock_client import (
    baostock_result_to_df,
    ensure_baostock_login,
    latest_quarter_row,
    query_baostock_history,
    query_baostock_table,
)
from models.inputs import AdjustType, KlinePeriod, LongTermFactorInput, SymbolInput


async def get_financial_analysis_impl(params: SymbolInput) -> str:
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
        raw_df = safe_to_numeric(raw_df)

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


async def get_valuation_status_impl(params: SymbolInput) -> str:
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
            "Status": status,
        }])

        return format_data(summary, title=f"Valuation Status (5-Year PE) for {params.symbol} (baostock)")
    except Exception:
        try:
            df_pe = ak.stock_zh_valuation_baidu(symbol=params.symbol, indicator="市盈率(TTM)", period="近5年")
            if df_pe.empty:
                return "Valuation data not available."
            current_val = df_pe["value"].iloc[-1]
            history_min = df_pe["value"].min()
            history_max = df_pe["value"].max()
            percentile = (df_pe["value"] <= current_val).mean() * 100
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
                "Status": status,
            }])
            return format_data(summary, title=f"Valuation Status (5-Year PE) for {params.symbol} (AKShare fallback)")
        except Exception as e:
            return handle_error(e, "get_valuation_status")


async def get_long_term_factor_score_impl(params: LongTermFactorInput) -> str:
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=365 * params.valuation_lookback_years + 20)).strftime("%Y%m%d")

        try:
            val_df = query_baostock_history(
                symbol=params.symbol,
                fields=["date", "code", "peTTM", "pbMRQ"],
                start_date=start_date,
                end_date=end_date,
                period=KlinePeriod.DAILY.value,
                adjust=AdjustType.NONE.value,
            )
            pe_pct = percentile_of_last(val_df["peTTM"])
            pb_pct = percentile_of_last(val_df["pbMRQ"])
            current_pe = pd.to_numeric(val_df["peTTM"], errors="coerce").dropna().iloc[-1]
            current_pb = pd.to_numeric(val_df["pbMRQ"], errors="coerce").dropna().iloc[-1]
        except Exception:
            val_df = ak.stock_zh_valuation_baidu(symbol=params.symbol, indicator="市盈率(TTM)", period="近5年")
            pe_pct = percentile_of_last(val_df["value"])
            pb_pct = 50.0
            current_pe = pd.to_numeric(val_df["value"], errors="coerce").dropna().iloc[-1]
            current_pb = float("nan")
        value_score = clamp_score((100 - pe_pct) * 0.6 + (100 - pb_pct) * 0.4)

        profit_row, fin_period = latest_quarter_row(params.symbol, bs.query_profit_data)
        balance_row, _ = latest_quarter_row(params.symbol, bs.query_balance_data)
        growth_row, _ = latest_quarter_row(params.symbol, bs.query_growth_data)

        roe = float(pd.to_numeric(pd.Series([profit_row.get("roeAvg")]), errors="coerce").iloc[0])
        debt = float(pd.to_numeric(pd.Series([balance_row.get("liabilityToAsset")]), errors="coerce").iloc[0])
        yoy_ni = float(pd.to_numeric(pd.Series([growth_row.get("YOYNI")]), errors="coerce").iloc[0])

        quality_score = clamp_score(clamp_score(roe * 4) * 0.6 + clamp_score(100 - debt) * 0.4)
        growth_score = clamp_score((yoy_ni + 20) * 2)

        dividend_cash = []
        current_year = datetime.now().year
        for year in range(current_year - 4, current_year + 1):
            try:
                df_y = query_baostock_table(
                    bs.query_dividend_data,
                    code=normalize_bs_code(params.symbol),
                    year=str(year),
                    yearType="report",
                )
                if df_y.empty:
                    continue
                if "dividCashPsBeforeTax" not in df_y.columns:
                    continue
                c = pd.to_numeric(df_y["dividCashPsBeforeTax"], errors="coerce").dropna()
                if not c.empty:
                    dividend_cash.append(float(c.iloc[0]))
            except Exception:
                continue
        positive_years = sum(1 for v in dividend_cash if v > 0)
        avg_dividend = float(sum(dividend_cash) / len(dividend_cash)) if dividend_cash else 0.0
        dividend_score = clamp_score((positive_years / 5) * 80 + min(avg_dividend, 2.0) * 10)

        total_score = clamp_score(
            value_score * 0.35 + quality_score * 0.35 + growth_score * 0.15 + dividend_score * 0.15
        )

        rating = "Neutral"
        if total_score >= 80:
            rating = "High Conviction"
        elif total_score >= 65:
            rating = "Watchlist Candidate"
        elif total_score < 45:
            rating = "Needs Caution"

        summary = pd.DataFrame([{
            "Symbol": params.symbol,
            "Valuation Window (Y)": params.valuation_lookback_years,
            "Latest Financial Period": fin_period,
            "Current PE(TTM)": round(float(current_pe), 2),
            "Current PB": round(float(current_pb), 2) if pd.notna(current_pb) else None,
            "ROE(%)": round(roe, 2),
            "Debt Ratio(%)": round(debt, 2),
            "YOY Net Profit(%)": round(yoy_ni, 2),
            "Avg Dividend/Share (5Y)": round(avg_dividend, 4),
            "Value Score": round(value_score, 2),
            "Quality Score": round(quality_score, 2),
            "Growth Score": round(growth_score, 2),
            "Dividend Score": round(dividend_score, 2),
            "Total Score": round(total_score, 2),
            "Rating": rating,
        }])
        return format_data(summary, title=f"Long-term Factor Score for {params.symbol}")
    except Exception as e:
        return handle_error(e, "get_long_term_factor_score")
