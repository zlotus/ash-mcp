from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import akshare as ak
import baostock as bs
import pandas as pd

from loguru import logger

from core.utils import (
    STRATEGY_CACHE_PATH,
    apply_weight_cap,
    clamp_score,
    compute_equity_metrics,
    format_data,
    handle_error,
    load_cache,
    normalize_bs_code,
    parse_symbol_list,
    parse_weight_map,
    percentile_of_last,
    save_cache,
    to_yyyymmdd,
    trim_dataframe,
)
from data.baostock_client import latest_quarter_row, query_baostock_history, query_baostock_table
from data.market_data import fetch_daily_close_series
from models.inputs import (
    AdjustType,
    GridStrategyInput,
    KlinePeriod,
    LowFreqBacktestInput,
    PortfolioRebalanceInput,
    RebalanceMethod,
    ValueCandidatesGridInput,
)


async def get_grid_strategy_impl(params: GridStrategyInput) -> str:
    try:
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
                "Distance from Current": f"-{round(i * params.grid_step_pct * 100, 1)}%",
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
                adjust="hfq",
            )
            df = df.tail(params.history_days)
            year_high = df["最高"].max()
            year_low = df["最低"].min()
            grids = []
            p = params.current_price
            for i in range(1, params.grid_count + 1):
                buy_price = p * (1 - params.grid_step_pct)
                sell_target = buy_price * (1 + params.grid_step_pct)
                grids.append({
                    "Level": f"Buy #{i}",
                    "Price": round(buy_price, 2),
                    "Sell Target": round(sell_target, 2),
                    "Distance from Current": f"-{round(i * params.grid_step_pct * 100, 1)}%",
                })
                p = buy_price
            summary_df = pd.DataFrame(grids)
            header = f"Grid Strategy for {params.symbol}\n"
            header += f"- Current Price: {params.current_price}\n"
            header += f"- {params.history_days}-Day Range (hfq): {year_low} - {year_high}\n"
            header += f"- Grid Step: {params.grid_step_pct*100}%\n"
            return header + "\n" + format_data(summary_df)
        except Exception as e:
            return handle_error(e, "get_grid_strategy")


async def get_low_freq_backtest_impl(params: LowFreqBacktestInput) -> str:
    try:
        default_start = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y%m%d")
        default_end = datetime.now().strftime("%Y%m%d")
        start_date = to_yyyymmdd(params.start_date, default_start)
        end_date = to_yyyymmdd(params.end_date, default_end)

        close_series = fetch_daily_close_series(params.symbol, start_date, end_date, adjust=params.adjust.value)
        if not isinstance(close_series.index, pd.DatetimeIndex):
            close_series.index = pd.to_datetime(close_series.index, errors="coerce")
            close_series = close_series[close_series.index.notna()]
        close_series = close_series.sort_index()
        if close_series.empty:
            raise RuntimeError("No valid price data in backtest window.")

        dt_index = pd.DatetimeIndex(close_series.index)
        month_index = dt_index.to_period("M")
        first_trade_day_by_month = close_series.groupby(month_index).head(1).index
        first_day_set = {pd.Timestamp(x) for x in first_trade_day_by_month}
        first_month = month_index[0]

        cash = float(params.initial_cash)
        shares = 0.0
        invested = float(params.initial_cash)
        equity_points: List[Tuple[pd.Timestamp, float]] = []
        trade_rows: List[Dict[str, Any]] = []

        for dt, px in zip(dt_index, close_series.to_numpy()):
            if dt in first_day_set:
                month = dt.to_period("M")
                if month != first_month and params.monthly_contribution > 0:
                    cash += float(params.monthly_contribution)
                    invested += float(params.monthly_contribution)
                if cash > 0 and px > 0:
                    buy_cash = cash * (1 - params.fee_rate)
                    buy_share = buy_cash / px
                    shares += buy_share
                    trade_rows.append({
                        "日期": dt.strftime("%Y-%m-%d"),
                        "买入价": round(float(px), 4),
                        "投入资金": round(float(cash), 2),
                        "买入股数": round(float(buy_share), 4),
                        "累计持仓股数": round(float(shares), 4),
                    })
                    cash = 0.0
            equity = shares * float(px) + cash
            equity_points.append((dt, equity))

        equity_curve = pd.Series({dt: eq for dt, eq in equity_points}).sort_index()
        metrics = compute_equity_metrics(equity_curve)
        final_value = float(equity_curve.iloc[-1])

        summary = pd.DataFrame([{
            "Symbol": params.symbol,
            "Start Date": pd.Timestamp(close_series.index[0]).strftime("%Y-%m-%d"),
            "End Date": pd.Timestamp(close_series.index[-1]).strftime("%Y-%m-%d"),
            "Initial Cash": round(params.initial_cash, 2),
            "Monthly Contribution": round(params.monthly_contribution, 2),
            "Total Invested": round(invested, 2),
            "Final Value": round(final_value, 2),
            "PnL": round(final_value - invested, 2),
            "Total Return(%)": round(metrics["total_return"] * 100, 2),
            "Annual Return(%)": round(metrics["annual_return"] * 100, 2),
            "Annual Vol(%)": round(metrics["annual_vol"] * 100, 2),
            "Sharpe": round(metrics["sharpe"], 3),
            "Max Drawdown(%)": round(metrics["max_drawdown"] * 100, 2),
            "Trade Count": len(trade_rows),
        }])
        trades = trim_dataframe(pd.DataFrame(trade_rows), max_rows=12) if trade_rows else pd.DataFrame()
        result = format_data(summary, title=f"Low-frequency Backtest Summary for {params.symbol}")
        if not trades.empty:
            result += "\n\n" + format_data(trades, title="Recent Monthly Buy Records")
        return result
    except Exception as e:
        return handle_error(e, "get_low_freq_backtest")


async def get_portfolio_rebalance_plan_impl(params: PortfolioRebalanceInput) -> str:
    try:
        symbols = parse_symbol_list(params.symbols)
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=int(params.lookback_days * 1.8))).strftime("%Y%m%d")

        close_data: Dict[str, pd.Series] = {}
        for symbol in symbols:
            close_data[symbol] = fetch_daily_close_series(symbol, start_date, end_date, adjust=AdjustType.HFQ.value)

        price_df = pd.DataFrame(close_data).dropna(how="any")
        if len(price_df) < 40:
            raise RuntimeError("Not enough overlapping price data for rebalance.")
        returns = price_df.pct_change().dropna(how="any")
        if returns.empty:
            raise RuntimeError("Empty return matrix.")

        if params.method == RebalanceMethod.EQUAL:
            raw_weights = pd.Series(1 / len(symbols), index=symbols, dtype=float)
        else:
            vol = returns.std()
            inv_vol = 1 / vol.replace(0, pd.NA)
            inv_vol = inv_vol.fillna(0.0)
            if inv_vol.sum() <= 0:
                raise RuntimeError("Unable to compute inverse-vol weights.")
            raw_weights = inv_vol / inv_vol.sum()

        target_weights = apply_weight_cap(raw_weights, params.max_single_weight)
        current_weight_map = parse_weight_map(params.current_weights)
        current_weights = pd.Series({s: current_weight_map.get(s, 0.0) for s in symbols}, dtype=float)
        if current_weights.sum() > 0:
            current_weights = current_weights / current_weights.sum()

        delta = target_weights - current_weights
        action = delta.apply(lambda x: "BUY" if x > 0.01 else ("SELL" if x < -0.01 else "HOLD"))
        plan_df = pd.DataFrame({
            "Symbol": symbols,
            "Current Weight": current_weights.reindex(symbols).values,
            "Target Weight": target_weights.reindex(symbols).values,
            "Delta": delta.reindex(symbols).values,
            "Action": action.reindex(symbols).values,
        })
        plan_df["Current Weight"] = (plan_df["Current Weight"] * 100).round(2)
        plan_df["Target Weight"] = (plan_df["Target Weight"] * 100).round(2)
        plan_df["Delta"] = (plan_df["Delta"] * 100).round(2)
        plan_df = plan_df.sort_values(by="Delta", ascending=False)

        cov = returns.cov() * 252
        w = target_weights.reindex(symbols).values
        port_vol = float((w @ cov.values @ w) ** 0.5)

        summary_df = pd.DataFrame([{
            "Method": params.method.value,
            "Lookback Days": params.lookback_days,
            "Symbols": len(symbols),
            "Max Single Weight(%)": round(params.max_single_weight * 100, 2),
            "Estimated Portfolio Vol(%)": round(port_vol * 100, 2),
            "Data Start": price_df.index[0].strftime("%Y-%m-%d"),
            "Data End": price_df.index[-1].strftime("%Y-%m-%d"),
        }])

        result = format_data(summary_df, title="Portfolio Rebalance Summary")
        result += "\n\n" + format_data(plan_df, title="Rebalance Actions")
        return result
    except Exception as e:
        return handle_error(e, "get_portfolio_rebalance_plan")


async def get_value_candidates_and_grid_impl(params: ValueCandidatesGridInput) -> str:
    try:
        today = datetime.now().strftime("%Y%m%d")
        cache = load_cache(STRATEGY_CACHE_PATH)
        
        # Initialize daily cache if not present
        if today not in cache:
            cache[today] = {"cap_map": {}, "metrics": {}, "industry_map": {}, "financial_cache": {}}

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=365 * params.lookback_years + 20)).strftime("%Y%m%d")
        anchor_market_cap = None

        def _to_float(v: Any) -> float:
            return float(pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0])

        # 1) Build candidate universe.
        anchor_code = normalize_bs_code(params.anchor_symbol)
        anchor_ind_df = query_baostock_table(bs.query_stock_industry, code=anchor_code)
        anchor_industry = ""
        if not anchor_ind_df.empty and "industry" in anchor_ind_df.columns:
            anchor_industry = str(anchor_ind_df.iloc[0].get("industry", "") or "")

        if params.strict_same_industry and anchor_industry:
            all_ind = query_baostock_table(bs.query_stock_industry)
            candidates_df = all_ind[all_ind["industry"] == anchor_industry].copy()
        else:
            try:
                hs300 = query_baostock_table(bs.query_hs300_stocks)
                zz500 = query_baostock_table(bs.query_zz500_stocks)
                candidates_df = pd.concat([hs300, zz500]).drop_duplicates(subset=["code"])
            except Exception:
                candidates_df = query_baostock_table(bs.query_stock_industry)
            
            candidates_df = candidates_df.sample(frac=1).reset_index(drop=True)

        code_col = "code" if "code" in candidates_df.columns else None
        name_col = "code_name" if "code_name" in candidates_df.columns else None
        if not code_col:
            raise RuntimeError("Industry/Index data missing code column.")

        ind_map = {}
        if "industry" in candidates_df.columns:
            for _, r in candidates_df.iterrows():
                ind_map[str(r[code_col])] = str(r.get("industry", ""))
        industry_map_cache = cache[today].get("industry_map", {})

        # Market-cap estimation is cache-first: cap_map -> financial_cache(total_share) + latest close.
        cap_map = cache[today].get("cap_map", {})
        financial_cache = cache[today].get("financial_cache", {})
        if cap_map:
            logger.info("Market cap cache hit.")
        else:
            logger.info("Market cap cache miss. Runtime will rely on financial_cache and skip heavy downloads.")

        def _estimate_market_cap(symbol_like: str) -> float:
            sym = str(symbol_like).split(".")[-1].strip()
            if not sym:
                return 0.0
            cached = _to_float(cap_map.get(sym))
            if pd.notna(cached) and cached > 0:
                logger.debug(f"[cap_map] hit for {sym}")
                return cached
            logger.debug(f"[cap_map] miss for {sym}")

            fin_cached = financial_cache.get(sym, {})
            fin_period_cached = str(fin_cached.get("financial_period", ""))
            if ("total_share" in fin_cached) and (not bool(fin_cached.get("unavailable"))):
                ts_cached = _to_float(fin_cached.get("total_share"))
                if pd.isna(ts_cached) or ts_cached <= 0:
                    financial_cache[sym] = {
                        "financial_period": fin_period_cached,
                        "total_share": None,
                        "unavailable": True,
                    }
                    logger.debug(f"[financial_cache] normalized invalid total_share -> unavailable for {sym}")
                    return 0.0
            if bool(fin_cached.get("unavailable")):
                logger.debug(f"[financial_cache] unavailable flag hit for {sym}, skip re-download")
                return 0.0
            total_share_cached = _to_float(fin_cached.get("total_share"))
            if pd.notna(total_share_cached) and total_share_cached > 0:
                logger.debug(f"[financial_cache] hit total_share for {sym}")
                try:
                    px_df = query_baostock_history(
                        symbol=sym,
                        fields=["date", "close"],
                        start_date=(datetime.now() - timedelta(days=20)).strftime("%Y%m%d"),
                        end_date=end_date,
                        period=KlinePeriod.DAILY.value,
                        adjust=AdjustType.NONE.value,
                    )
                    if not px_df.empty and "close" in px_df.columns:
                        close_series = pd.to_numeric(px_df["close"], errors="coerce").dropna()
                        if not close_series.empty:
                            latest_close = float(close_series.iloc[-1])
                            mcap = latest_close * total_share_cached
                            if pd.notna(mcap) and mcap > 0:
                                cap_map[sym] = float(mcap)
                                logger.info(f"[cap_map] updated from financial_cache+price for {sym}")
                                return float(mcap)
                except Exception as e:
                    logger.debug(f"Market cap estimate (financial_cache path) failed for {sym}: {e}")
            else:
                logger.debug(f"[financial_cache] miss total_share for {sym}")

            logger.debug(f"[financial_cache] no usable total_share for {sym}, skip runtime market-cap download")
            return 0.0

        anchor_market_cap = _estimate_market_cap(anchor_code)
        if anchor_market_cap > 0:
            logger.info(f"Anchor market cap estimated for {anchor_code}: {anchor_market_cap:.2f}")
        else:
            logger.warning(f"Anchor market cap unavailable for {anchor_code}; cap-ratio filter will be skipped.")

        potential_codes = []
        for _, row in candidates_df.iterrows():
            code = str(row.get(code_col, "")).strip().lower()
            if not code or code == anchor_code:
                continue
            symbol = code.split(".")[-1]
            
            mcap = _estimate_market_cap(symbol)
            ratio = mcap / anchor_market_cap if anchor_market_cap else 1.0
            
            if anchor_market_cap:
                if not (params.market_cap_ratio_min <= ratio <= params.market_cap_ratio_max):
                    continue
            
            industry = row.get("industry", ind_map.get(code, ""))
            if not industry:
                industry = industry_map_cache.get(code, "")
            if not industry:
                try:
                    ind_res = query_baostock_table(bs.query_stock_industry, code=code)
                    if not ind_res.empty:
                        industry = str(ind_res.iloc[0].get("industry", ""))
                        if industry:
                            industry_map_cache[code] = industry
                except Exception:
                    industry = "Unknown"

            potential_codes.append({
                "symbol": symbol,
                "name": str(row.get(name_col, "")) if name_col else "",
                "industry": industry,
                "market_cap": mcap,
                "cap_ratio": ratio,
            })
            if len(potential_codes) >= params.candidate_limit:
                break

        cache[today]["cap_map"] = cap_map
        cache[today]["industry_map"] = industry_map_cache
        cache[today]["financial_cache"] = financial_cache
        save_cache(STRATEGY_CACHE_PATH, cache)

        if not potential_codes:
            return f"No candidates found matching market cap scale ({params.market_cap_ratio_min}-{params.market_cap_ratio_max}x of anchor)."

        # 2) Score candidates for quality and value.
        scored: List[Dict[str, Any]] = []
        metrics_cache = cache[today].get("metrics", {})
        cache_updated = False

        for c in potential_codes:
            symbol = c["symbol"]
            
            # Check Metrics Cache
            if symbol in metrics_cache:
                m = metrics_cache[symbol]
                logger.debug(f"Cache hit for {symbol}")
                # Check if it passes filters (filters are dynamic based on params)
                is_financial = any(k in c["industry"] for k in ["银行", "保险", "证券", "金融"])
                effective_max_debt = 92.0 if is_financial else params.max_debt_ratio
                
                pass_filters = (
                    m["pe_pct"] <= params.max_pe_percentile
                    and m["pb_pct"] <= params.max_pb_percentile
                    and m["roe"] >= params.min_roe
                    and m["debt"] <= effective_max_debt
                    and m["yoy_ni"] >= params.min_yoy_net_profit
                    and m["dividend_years_5y"] >= params.min_dividend_years_5y
                    and m["annual_vol"] <= params.max_annual_volatility
                )
                if pass_filters:
                    scored.append({**c, **m, "is_financial": is_financial, "hard_pass": True})
                continue
            
            logger.info(f"Cache miss for {symbol}. Fetching live metrics...")
            try:
                # Value indicators (Baostock)
                val_df_raw = query_baostock_history(
                    symbol=symbol,
                    fields=["date", "peTTM", "pbMRQ", "close"],
                    start_date=start_date,
                    end_date=end_date,
                    period=KlinePeriod.DAILY.value,
                    adjust=AdjustType.NONE.value,
                )
                if val_df_raw.empty or "peTTM" not in val_df_raw.columns:
                    continue
                
                pe_series = pd.to_numeric(val_df_raw["peTTM"], errors="coerce").dropna()
                pe_series = pe_series[pe_series > 0]
                if pe_series.empty: continue
                pe_pct = percentile_of_last(pe_series)
                curr_pe = pe_series.iloc[-1]

                pb_series = pd.to_numeric(val_df_raw["pbMRQ"], errors="coerce").dropna()
                pb_series = pb_series[pb_series > 0]
                pb_pct = percentile_of_last(pb_series) if not pb_series.empty else 50.0

                # Quality indicators (Baostock)
                profit_row, p_period = latest_quarter_row(symbol, bs.query_profit_data)
                balance_row, _ = latest_quarter_row(symbol, bs.query_balance_data)
                growth_row, _ = latest_quarter_row(symbol, bs.query_growth_data)
                
                q_num = int(p_period[-1]) if p_period and p_period[-1].isdigit() else 4
                roe_raw = _to_float(profit_row.get("roeAvg"))
                roe_annual = roe_raw * (4 / q_num) * 100
                
                a2e = _to_float(balance_row.get("assetToEquity"))
                debt = (1 - (1 / a2e)) * 100 if a2e > 1 else _to_float(balance_row.get("liabilityToAsset")) * 100
                
                yoy_ni = _to_float(growth_row.get("YOYNI")) * 100
                
                is_financial = any(k in c["industry"] for k in ["银行", "保险", "证券", "金融"])
                effective_max_debt = 92.0 if is_financial else params.max_debt_ratio

                # Dividend consistency (Baostock)
                div_years = 0
                cy = datetime.now().year
                for yr in range(cy - 4, cy + 1):
                    try:
                        ddf = query_baostock_table(bs.query_dividend_data, code=normalize_bs_code(symbol), year=str(yr), yearType="report")
                        if not ddf.empty and _to_float(ddf.iloc[0].get("dividCashPsBeforeTax")) > 0:
                            div_years += 1
                    except Exception:
                        continue

                # Risk/Volatility (Baostock HFQ)
                val_df_hfq = query_baostock_history(symbol=symbol, fields=["date", "close"], start_date=start_date, end_date=end_date, period=KlinePeriod.DAILY.value, adjust=AdjustType.HFQ.value)
                close_hfq = pd.to_numeric(val_df_hfq["close"], errors="coerce").dropna()
                ret = close_hfq.pct_change().dropna()
                annual_vol = float(ret.std() * (252 ** 0.5))

                # Scoring
                value_score = clamp_score((100 - pe_pct) * 0.6 + (100 - pb_pct) * 0.4)
                quality_score = clamp_score(clamp_score(roe_annual * 4) * 0.7 + clamp_score(100 - debt) * 0.3)
                growth_score = clamp_score(yoy_ni + 50)
                total_score = clamp_score(value_score * 0.4 + quality_score * 0.4 + (div_years * 20) * 0.2)

                m_data = {
                    "current_price": _to_float(val_df_raw["close"].iloc[-1] if "close" in val_df_raw.columns else val_df_hfq["close"].iloc[-1]),
                    "current_pe": curr_pe,
                    "pe_pct": pe_pct, "pb_pct": pb_pct,
                    "roe": roe_annual, "debt": debt, "yoy_ni": yoy_ni,
                    "dividend_years_5y": div_years, "annual_vol": annual_vol,
                    "value_score": value_score, "quality_score": quality_score,
                    "growth_score": growth_score, "total_score": total_score
                }
                
                # Update Cache
                metrics_cache[symbol] = m_data
                cache_updated = True

                pass_filters = (
                    pe_pct <= params.max_pe_percentile
                    and pb_pct <= params.max_pb_percentile
                    and roe_annual >= params.min_roe
                    and debt <= effective_max_debt
                    and yoy_ni >= params.min_yoy_net_profit
                    and div_years >= params.min_dividend_years_5y
                    and annual_vol <= params.max_annual_volatility
                )
                if pass_filters:
                    scored.append({**c, **m_data, "is_financial": is_financial, "hard_pass": True})
            except Exception:
                continue

        if cache_updated:
            cache[today]["metrics"] = metrics_cache
            save_cache(STRATEGY_CACHE_PATH, cache)

        if not scored:
            return "No candidates found matching the high-quality blue-chip filters. Consider relaxing ROE or PE percentile requirements."

        scored_df = pd.DataFrame(scored).sort_values(by="total_score", ascending=False).head(params.top_n)
        picks = scored_df.to_dict(orient="records")
        
        summary = scored_df.rename(columns={
            "symbol": "Symbol", "name": "Name", "industry": "Industry",
            "current_price": "Current Price", "current_pe": "Current PE",
            "pe_pct": "PE Pct(%)", "pb_pct": "PB Pct(%)",
            "roe": "ROE(%)", "debt": "Debt(%)", "yoy_ni": "YoY Profit(%)",
            "dividend_years_5y": "DivYears(5Y)", "market_cap": "MCap",
            "cap_ratio": "MCap/Anchor", "annual_vol": "Vol",
            "value_score": "ValScore", "quality_score": "QualScore",
            "growth_score": "GrowScore", "total_score": "Score",
            "hard_pass": "Pass"
        })[[
            "Symbol", "Name", "Industry", "Current Price", "Current PE",
            "PE Pct(%)", "PB Pct(%)", "ROE(%)", "Debt(%)", "YoY Profit(%)", 
            "DivYears(5Y)", "MCap", "MCap/Anchor", "Vol",
            "ValScore", "QualScore", "GrowScore", "Score", "Pass"
        ]]
        
        for col in [
            "Current Price", "Current PE", "PE Pct(%)", "PB Pct(%)",
            "ROE(%)", "Debt(%)", "YoY Profit(%)", "MCap", "MCap/Anchor", "Vol",
            "ValScore", "QualScore", "GrowScore", "Score"
        ]:
            summary[col] = pd.to_numeric(summary[col], errors="coerce").round(2)

        cfg_df = pd.DataFrame([{
            "Anchor": params.anchor_symbol,
            "Anchor Industry": anchor_industry or "N/A",
            "Strict": params.strict_same_industry,
            "Top N": params.top_n,
            "Max PE%": params.max_pe_percentile,
            "Min ROE": params.min_roe,
            "MCap Min": params.market_cap_ratio_min,
            "MCap Max": params.market_cap_ratio_max,
        }])

        result = format_data(cfg_df, title="Selection Constraints")
        result += "\n\n" + format_data(summary, title=f"Value Candidates Similar to {params.anchor_symbol}")

        reasons: List[Dict[str, Any]] = []
        for item in picks:
            tags: List[str] = []
            if float(item.get("pe_pct", 100)) <= 30: tags.append("PE分位低")
            if float(item.get("pb_pct", 100)) <= 30: tags.append("PB分位低")
            if float(item.get("roe", 0)) >= 12: tags.append("ROE较高")
            if float(item.get("dividend_years_5y", 0)) >= 4: tags.append("分红稳定")
            if float(item.get("annual_vol", 9)) <= 0.35: tags.append("波动较低")
            if not tags: tags.append("综合评分达标")
            reasons.append({"Symbol": item.get("symbol"), "Why Selected": "、".join(tags)})
        result += "\n\n" + format_data(pd.DataFrame(reasons), title="Selection Reasons")

        # 3) Build grid strategy for each selected stock (Baostock only)
        for item in picks:
            symbol = str(item["symbol"])
            current_price = float(item["current_price"])
            start_grid = (datetime.now() - timedelta(days=params.history_days * 2)).strftime("%Y%m%d")
            try:
                hdf = query_baostock_history(
                    symbol=symbol,
                    fields=["date", "code", "high", "low", "close"],
                    start_date=start_grid,
                    end_date=end_date,
                    period=KlinePeriod.DAILY.value,
                    adjust=AdjustType.HFQ.value,
                ).tail(params.history_days)
                if hdf.empty: continue
                year_high = float(pd.to_numeric(hdf["high"], errors="coerce").dropna().max())
                year_low = float(pd.to_numeric(hdf["low"], errors="coerce").dropna().min())
            except Exception:
                continue

            grids = []
            p = current_price
            for i in range(1, params.grid_count + 1):
                buy_price = p * (1 - params.grid_step_pct)
                sell_target = buy_price * (1 + params.grid_step_pct)
                grids.append({
                    "Level": f"Buy #{i}",
                    "Buy Price": round(buy_price, 2),
                    "Sell Target": round(sell_target, 2),
                    "Dist.": f"-{round(i * params.grid_step_pct * 100, 1)}%",
                })
                p = buy_price

            ctx_df = pd.DataFrame([{
                "Symbol": symbol,
                "Price": round(current_price, 2),
                "Range": f"{round(year_low, 2)} - {round(year_high, 2)}",
                "Step(%)": round(params.grid_step_pct * 100, 2),
            }])
            result += "\n\n" + format_data(ctx_df, title=f"Grid Context for {symbol}")
            result += "\n\n" + format_data(pd.DataFrame(grids), title=f"Grid Plan for {symbol}")

        return result
    except Exception as e:
        return handle_error(e, "get_value_candidates_and_grid")
