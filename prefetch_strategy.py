import asyncio
import os
from datetime import datetime, timedelta
from typing import Any

import baostock as bs
import pandas as pd
from loguru import logger

from core.utils import (
    STRATEGY_CACHE_PATH,
    clamp_score,
    load_cache,
    normalize_bs_code,
    percentile_of_last,
    save_cache,
)
from data.baostock_client import latest_quarter_row, query_baostock_history, query_baostock_table, ensure_baostock_login
from models.inputs import KlinePeriod, AdjustType

def _to_float(v: Any) -> float:
    return float(pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0])

def cleanup_cache(keep_days: int = 7):
    """Remove cache entries older than keep_days."""
    cache = load_cache(STRATEGY_CACHE_PATH)
    if not cache:
        return
    
    today = datetime.now()
    cutoff_date = (today - timedelta(days=keep_days)).strftime("%Y%m%d")
    
    original_keys = list(cache.keys())
    new_cache = {k: v for k, v in cache.items() if k >= cutoff_date}
    
    removed = set(original_keys) - set(new_cache.keys())
    if removed:
        logger.info(f"Cleaned up old cache entries: {removed}")
        save_cache(STRATEGY_CACHE_PATH, new_cache)

async def prefetch_index_metrics(index_name="HS300"):
    """
    Prefetch and cache metrics for all index components.
    """
    ensure_baostock_login()
    today = datetime.now().strftime("%Y%m%d")
    cache = load_cache(STRATEGY_CACHE_PATH)
    
    if today not in cache:
        cache[today] = {"cap_map": {}, "metrics": {}}
    
    metrics_cache = cache[today].get("metrics", {})
    
    logger.info(f"Fetching full components for {index_name}...")
    if index_name == "HS300":
        rs = bs.query_hs300_stocks()
    elif index_name == "ZZ500":
        rs = bs.query_zz500_stocks()
    else:
        logger.error(f"Unsupported index: {index_name}")
        return

    stocks = []
    while rs.next():
        stocks.append(rs.get_row_data())
    
    df_stocks = pd.DataFrame(stocks, columns=rs.fields)
    total = len(df_stocks)
    logger.info(f"Processing {total} stocks for {index_name}...")

    lookback_years = 5
    end_date = today
    start_date = (datetime.now() - timedelta(days=365 * lookback_years + 20)).strftime("%Y%m%d")

    count = 0
    added_count = 0
    for _, row in df_stocks.iterrows():
        symbol = row['code'].split('.')[-1]
        count += 1
        
        if symbol in metrics_cache:
            continue
        
        try:
            val_df_raw = query_baostock_history(
                symbol=symbol, fields=["date", "peTTM", "pbMRQ", "close"],
                start_date=start_date, end_date=end_date,
                period=KlinePeriod.DAILY.value, adjust=AdjustType.NONE.value,
            )
            if val_df_raw.empty: continue
            
            pe_series = pd.to_numeric(val_df_raw["peTTM"], errors="coerce").dropna()
            pe_series = pe_series[pe_series > 0]
            if pe_series.empty: continue
            pe_pct = percentile_of_last(pe_series)
            curr_pe = pe_series.iloc[-1]

            pb_series = pd.to_numeric(val_df_raw["pbMRQ"], errors="coerce").dropna()
            pb_series = pb_series[pb_series > 0]
            pb_pct = percentile_of_last(pb_series) if not pb_series.empty else 50.0

            profit_row, p_period = latest_quarter_row(symbol, bs.query_profit_data)
            balance_row, _ = latest_quarter_row(symbol, bs.query_balance_data)
            growth_row, _ = latest_quarter_row(symbol, bs.query_growth_data)
            
            q_num = int(p_period[-1]) if p_period and p_period[-1].isdigit() else 4
            roe_annual = _to_float(profit_row.get("roeAvg")) * (4 / q_num) * 100
            a2e = _to_float(balance_row.get("assetToEquity"))
            debt = (1 - (1 / a2e)) * 100 if a2e > 1 else _to_float(balance_row.get("liabilityToAsset")) * 100
            yoy_ni = _to_float(growth_row.get("YOYNI")) * 100
            
            div_years = 0
            for yr in range(datetime.now().year - 4, datetime.now().year + 1):
                try:
                    ddf = query_baostock_table(bs.query_dividend_data, code=normalize_bs_code(symbol), year=str(yr), yearType="report")
                    if not ddf.empty and _to_float(ddf.iloc[0].get("dividCashPsBeforeTax")) > 0:
                        div_years += 1
                except: continue

            val_df_hfq = query_baostock_history(symbol=symbol, fields=["date", "close"], start_date=start_date, end_date=end_date, period=KlinePeriod.DAILY.value, adjust=AdjustType.HFQ.value)
            close_hfq = pd.to_numeric(val_df_hfq["close"], errors="coerce").dropna()
            ret = close_hfq.pct_change().dropna()
            annual_vol = float(ret.std() * (252 ** 0.5))

            value_score = clamp_score((100 - pe_pct) * 0.6 + (100 - pb_pct) * 0.4)
            quality_score = clamp_score(clamp_score(roe_annual * 4) * 0.7 + clamp_score(100 - debt) * 0.3)
            growth_score = clamp_score(yoy_ni + 50)
            total_score = clamp_score(value_score * 0.4 + quality_score * 0.4 + (div_years * 20) * 0.2)

            metrics_cache[symbol] = {
                "current_price": _to_float(val_df_raw["close"].iloc[-1]),
                "current_pe": curr_pe,
                "pe_pct": pe_pct, "pb_pct": pb_pct,
                "roe": roe_annual, "debt": debt, "yoy_ni": yoy_ni,
                "dividend_years_5y": div_years, "annual_vol": annual_vol,
                "value_score": value_score, "quality_score": quality_score,
                "growth_score": growth_score, "total_score": total_score
            }
            added_count += 1
            
            if added_count % 20 == 0:
                cache[today]["metrics"] = metrics_cache
                save_cache(STRATEGY_CACHE_PATH, cache)
                logger.info(f"[{index_name}] Progress: {count}/{total}. Added {added_count} to cache.")

        except Exception as e:
            continue

    cache[today]["metrics"] = metrics_cache
    save_cache(STRATEGY_CACHE_PATH, cache)
    logger.info(f"Index {index_name} prefetch done. Added {added_count} new entries.")

async def main():
    # 1. Cleanup old data
    cleanup_cache(keep_days=7)
    
    # 2. Prefetch HS300
    await prefetch_index_metrics("HS300")
    
    # 3. Prefetch ZZ500
    await prefetch_index_metrics("ZZ500")
    
    bs.logout()
    logger.info("All prefetch tasks completed.")

if __name__ == "__main__":
    asyncio.run(main())