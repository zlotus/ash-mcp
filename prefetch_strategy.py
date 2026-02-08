import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import akshare as ak
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


def _enable_debug_logging_if_requested() -> None:
    flag = str(os.getenv("PREFETCH_DEBUG", "")).strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG",
    )
    logger.add("mcp.log", rotation="10 MB", retention="10 days", level="DEBUG")
    logger.info("PREFETCH_DEBUG enabled: debug logs are on.")


def _to_float(v: Any) -> float:
    return float(pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0])


def _get_total_share_ak(symbol: str) -> float:
    try:
        logger.debug(f"[AK] request stock_individual_info_em symbol={symbol}")
        info_df = ak.stock_individual_info_em(symbol=symbol)
        if info_df.empty or "item" not in info_df.columns or "value" not in info_df.columns:
            return float("nan")
        row = info_df[info_df["item"].astype(str).str.contains("总股本", na=False)]
        if row.empty:
            return float("nan")
        return _to_float(row.iloc[0].get("value"))
    except Exception:
        return float("nan")


def _get_total_share_profit(symbol: str) -> tuple[float, str]:
    # totalShare is exposed in profit_data for this environment; balance_data does not provide it.
    try:
        logger.debug(f"[BS] request latest_quarter_row query_profit_data symbol={symbol}")
        profit_row, p_period = latest_quarter_row(symbol, bs.query_profit_data)
        total_share = _to_float(profit_row.get("totalShare"))
        if pd.notna(total_share) and total_share > 0:
            return float(total_share), p_period
    except Exception:
        pass
    return float("nan"), ""

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
        cache[today] = {"cap_map": {}, "metrics": {}, "industry_map": {}, "financial_cache": {}}

    metrics_cache = cache[today].get("metrics", {})
    cap_map = cache[today].get("cap_map", {})
    financial_cache = cache[today].get("financial_cache", {})
    logger.info(
        f"[{index_name}] cache bootstrap: metrics={len(metrics_cache)}, cap_map={len(cap_map)}, financial_cache={len(financial_cache)}"
    )
    
    logger.info(f"Fetching full components for {index_name}...")
    if index_name == "HS300":
        logger.debug("[BS] request query_hs300_stocks")
        rs = bs.query_hs300_stocks()
    elif index_name == "ZZ500":
        logger.debug("[BS] request query_zz500_stocks")
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
    cache_hit_count = 0
    fetch_count = 0
    skip_count = 0
    fail_count = 0
    share_unavailable_count = 0
    checkpoint_every = 20

    def checkpoint(force: bool = False):
        if not force and (count % checkpoint_every != 0):
            return
        cache[today]["metrics"] = metrics_cache
        cache[today]["cap_map"] = cap_map
        cache[today]["financial_cache"] = financial_cache
        save_cache(STRATEGY_CACHE_PATH, cache)
        logger.info(
            f"[{index_name}] Progress: {count}/{total} | added={added_count} | "
            f"cache_hit={cache_hit_count} | fetched={fetch_count} | skipped={skip_count} | "
            f"share_unavailable={share_unavailable_count} | failed={fail_count}"
        )

    for _, row in df_stocks.iterrows():
        symbol = row['code'].split('.')[-1]
        count += 1
        m_cached = metrics_cache.get(symbol)
        if m_cached:
            cache_hit_count += 1
            # If metrics already exist, try to backfill market cap from cached shares only.
            fin_cached = financial_cache.get(symbol, {})
            if ("total_share" in fin_cached) and (not bool(fin_cached.get("unavailable"))):
                ts_cached = _to_float(fin_cached.get("total_share"))
                if pd.isna(ts_cached) or ts_cached <= 0:
                    financial_cache[symbol] = {
                        "financial_period": str(fin_cached.get("financial_period", "")),
                        "total_share": None,
                        "unavailable": True,
                    }
            if bool(fin_cached.get("unavailable")):
                checkpoint()
                continue
            total_share_cached = _to_float(fin_cached.get("total_share"))
            current_price_cached = _to_float(m_cached.get("current_price"))
            if symbol not in cap_map and pd.notna(total_share_cached) and total_share_cached > 0 and pd.notna(current_price_cached) and current_price_cached > 0:
                cap_map[symbol] = float(current_price_cached * total_share_cached)
            elif symbol not in cap_map:
                # Re-try totalShare in nightly prefetch only: profit_data first, AKShare fallback.
                share_profit, period_profit = _get_total_share_profit(symbol)
                share_val = share_profit
                share_source = "baostock.profit_data"
                share_period = period_profit or str(fin_cached.get("financial_period", ""))
                if not (pd.notna(share_val) and share_val > 0):
                    share_ak = _get_total_share_ak(symbol)
                    if pd.notna(share_ak) and share_ak > 0:
                        share_val = float(share_ak)
                        share_source = "akshare"
                if pd.notna(share_val) and share_val > 0 and pd.notna(current_price_cached) and current_price_cached > 0:
                    financial_cache[symbol] = {
                        "financial_period": share_period,
                        "total_share": float(share_val),
                        "unavailable": False,
                        "source": share_source,
                    }
                    cap_map[symbol] = float(current_price_cached * share_val)
                else:
                    financial_cache[symbol] = {
                        "financial_period": share_period,
                        "total_share": None,
                        "unavailable": True,
                        "source": "baostock.profit_data->akshare",
                    }
                    share_unavailable_count += 1
            checkpoint()
            continue
        
        try:
            logger.debug(f"[BS] request query_history_k_data_plus(valuation) symbol={symbol} range={start_date}~{end_date}")
            val_df_raw = query_baostock_history(
                symbol=symbol, fields=["date", "peTTM", "pbMRQ", "close"],
                start_date=start_date, end_date=end_date,
                period=KlinePeriod.DAILY.value, adjust=AdjustType.NONE.value,
            )
            if val_df_raw.empty:
                skip_count += 1
                checkpoint()
                continue
            
            pe_series = pd.to_numeric(val_df_raw["peTTM"], errors="coerce").dropna()
            pe_series = pe_series[pe_series > 0]
            if pe_series.empty:
                skip_count += 1
                checkpoint()
                continue
            pe_pct = percentile_of_last(pe_series)
            curr_pe = pe_series.iloc[-1]

            pb_series = pd.to_numeric(val_df_raw["pbMRQ"], errors="coerce").dropna()
            pb_series = pb_series[pb_series > 0]
            pb_pct = percentile_of_last(pb_series) if not pb_series.empty else 50.0

            logger.debug(f"[BS] request latest_quarter_row query_profit_data symbol={symbol}")
            profit_row, p_period = latest_quarter_row(symbol, bs.query_profit_data)
            logger.debug(f"[BS] request latest_quarter_row query_balance_data symbol={symbol}")
            balance_row, _ = latest_quarter_row(symbol, bs.query_balance_data)
            logger.debug(f"[BS] request latest_quarter_row query_growth_data symbol={symbol}")
            growth_row, _ = latest_quarter_row(symbol, bs.query_growth_data)
            
            q_num = int(p_period[-1]) if p_period and p_period[-1].isdigit() else 4
            roe_annual = _to_float(profit_row.get("roeAvg")) * (4 / q_num) * 100
            a2e = _to_float(balance_row.get("assetToEquity"))
            debt = (1 - (1 / a2e)) * 100 if a2e > 1 else _to_float(balance_row.get("liabilityToAsset")) * 100
            yoy_ni = _to_float(growth_row.get("YOYNI")) * 100
            # totalShare source: profit_data (primary), AKShare fallback only in nightly prefetch.
            total_share = _to_float(profit_row.get("totalShare"))
            share_source = "baostock.profit_data"
            if not (pd.notna(total_share) and total_share > 0):
                share_ak = _get_total_share_ak(symbol)
                if pd.notna(share_ak) and share_ak > 0:
                    total_share = float(share_ak)
                    share_source = "akshare"
            
            div_years = 0
            for yr in range(datetime.now().year - 4, datetime.now().year + 1):
                try:
                    logger.debug(f"[BS] request query_dividend_data symbol={symbol} year={yr}")
                    ddf = query_baostock_table(bs.query_dividend_data, code=normalize_bs_code(symbol), year=str(yr), yearType="report")
                    if not ddf.empty and _to_float(ddf.iloc[0].get("dividCashPsBeforeTax")) > 0:
                        div_years += 1
                except: continue

            logger.debug(f"[BS] request query_history_k_data_plus(hfq) symbol={symbol} range={start_date}~{end_date}")
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
            if pd.notna(total_share) and total_share > 0:
                financial_cache[symbol] = {
                    "financial_period": p_period,
                    "total_share": float(total_share),
                    "unavailable": False,
                    "source": share_source,
                }
                cap_map[symbol] = float(_to_float(val_df_raw["close"].iloc[-1]) * total_share)
            else:
                financial_cache[symbol] = {
                    "financial_period": p_period,
                    "total_share": None,
                    "unavailable": True,
                    "source": "baostock.profit_data->akshare",
                }
                share_unavailable_count += 1
            added_count += 1
            fetch_count += 1
            checkpoint()

        except Exception as e:
            fail_count += 1
            logger.debug(f"[{index_name}] failed for {symbol}: {e}")
            checkpoint()
            continue

    checkpoint(force=True)
    logger.info(
        f"Index {index_name} prefetch done. total={total}, added={added_count}, "
        f"cache_hit={cache_hit_count}, fetched={fetch_count}, skipped={skip_count}, "
        f"share_unavailable={share_unavailable_count}, failed={fail_count}."
    )


async def prefetch_industry_map():
    """Prefetch and cache full stock-industry mapping (non-realtime static metadata)."""
    ensure_baostock_login()
    today = datetime.now().strftime("%Y%m%d")
    cache = load_cache(STRATEGY_CACHE_PATH)
    if today not in cache:
        cache[today] = {"cap_map": {}, "metrics": {}, "industry_map": {}, "financial_cache": {}}

    try:
        logger.debug("[BS] request query_stock_industry(all)")
        ind_df = query_baostock_table(bs.query_stock_industry)
        industry_map = {}
        if not ind_df.empty:
            for _, r in ind_df.iterrows():
                code = str(r.get("code", "")).strip().lower()
                industry = str(r.get("industry", "")).strip()
                if code and industry:
                    industry_map[code] = industry
        cache[today]["industry_map"] = industry_map
        save_cache(STRATEGY_CACHE_PATH, cache)
        logger.info(f"Industry map prefetched. Entries: {len(industry_map)}")
    except Exception as e:
        logger.warning(f"Failed to prefetch industry map: {e}")

async def main():
    _enable_debug_logging_if_requested()
    # 1. Cleanup old data
    cleanup_cache(keep_days=7)
    
    # 2. Prefetch static industry mapping
    await prefetch_industry_map()
    
    # 3. Prefetch HS300
    await prefetch_index_metrics("HS300")
    
    # 4. Prefetch ZZ500
    await prefetch_index_metrics("ZZ500")
    
    bs.logout()
    logger.info("All prefetch tasks completed.")

if __name__ == "__main__":
    asyncio.run(main())
