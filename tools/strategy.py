from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import akshare as ak
import baostock as bs
import pandas as pd

from core.utils import (
    apply_weight_cap,
    clamp_score,
    compute_equity_metrics,
    format_data,
    handle_error,
    normalize_bs_code,
    parse_symbol_list,
    parse_weight_map,
    percentile_of_last,
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
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=365 * params.lookback_years + 20)).strftime("%Y%m%d")
        anchor_market_cap = None

        def _to_float(v: Any) -> float:
            return float(pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0])

        # 1) Build candidate universe around anchor industry (Ping An-style financials by default).
        anchor_code = normalize_bs_code(params.anchor_symbol)
        anchor_ind_df = query_baostock_table(bs.query_stock_industry, code=anchor_code)
        anchor_industry = ""
        if not anchor_ind_df.empty and "industry" in anchor_ind_df.columns:
            anchor_industry = str(anchor_ind_df.iloc[0].get("industry", "") or "")

        all_ind = query_baostock_table(bs.query_stock_industry)
        if all_ind.empty:
            raise RuntimeError("Industry mapping is empty.")

        if anchor_industry:
            same_ind = all_ind[all_ind["industry"] == anchor_industry] if "industry" in all_ind.columns else pd.DataFrame()
        else:
            same_ind = pd.DataFrame()

        candidates_df = same_ind
        if params.strict_same_industry and not candidates_df.empty:
            pass
        elif candidates_df.empty or len(candidates_df) < params.top_n:
            keywords = ["保险", "银行", "证券", "多元金融", "金融"]
            if "industry" in all_ind.columns:
                mask = all_ind["industry"].astype(str).str.contains("|".join(keywords), na=False)
                candidates_df = all_ind[mask]

        if candidates_df.empty:
            candidates_df = all_ind

        code_col = "code" if "code" in candidates_df.columns else None
        name_col = "code_name" if "code_name" in candidates_df.columns else None
        if not code_col:
            raise RuntimeError("Industry data missing code column.")

        candidate_rows: List[Dict[str, Any]] = []
        seen = set()
        for _, row in candidates_df.iterrows():
            code = str(row.get(code_col, "")).strip().lower()
            if not code or code == anchor_code:
                continue
            symbol = code.split(".")[-1]
            if symbol in seen:
                continue
            seen.add(symbol)
            candidate_rows.append({
                "symbol": symbol,
                "name": str(row.get(name_col, "")) if name_col else "",
                "industry": str(row.get("industry", "")),
            })
            if len(candidate_rows) >= params.candidate_limit:
                break

        if not candidate_rows:
            raise RuntimeError("No candidates found from industry mapping.")

        cap_map: Dict[str, float] = {}
        try:
            spot_df = ak.stock_zh_a_spot_em()
            if not spot_df.empty and "代码" in spot_df.columns and "总市值" in spot_df.columns:
                for _, r in spot_df.iterrows():
                    sym = str(r.get("代码", "")).strip()
                    if not sym:
                        continue
                    cap_val = _to_float(r.get("总市值"))
                    if pd.notna(cap_val) and cap_val > 0:
                        cap_map[sym] = cap_val
                anchor_market_cap = cap_map.get(params.anchor_symbol)
        except Exception:
            cap_map = {}

        # 2) Score candidates for undervaluation + quality.
        scored: List[Dict[str, Any]] = []
        for c in candidate_rows:
            symbol = c["symbol"]
            try:
                val_df = query_baostock_history(
                    symbol=symbol,
                    fields=["date", "code", "close", "peTTM", "pbMRQ"],
                    start_date=start_date,
                    end_date=end_date,
                    period=KlinePeriod.DAILY.value,
                    adjust=AdjustType.NONE.value,
                )
                if val_df.empty or "peTTM" not in val_df.columns:
                    continue
                pe_series = pd.to_numeric(val_df["peTTM"], errors="coerce")
                pe_series = pe_series[(pe_series > 0) & pe_series.notna()]
                if pe_series.empty:
                    continue

                pb_series = pd.to_numeric(val_df["pbMRQ"], errors="coerce")
                pb_series = pb_series[(pb_series > 0) & pb_series.notna()]
                pe_pct = percentile_of_last(pe_series)
                pb_pct = percentile_of_last(pb_series) if not pb_series.empty else 50.0
                value_score = clamp_score((100 - pe_pct) * 0.6 + (100 - pb_pct) * 0.4)

                profit_row, _ = latest_quarter_row(symbol, bs.query_profit_data)
                balance_row, _ = latest_quarter_row(symbol, bs.query_balance_data)
                growth_row, _ = latest_quarter_row(symbol, bs.query_growth_data)
                roe = float(pd.to_numeric(pd.Series([profit_row.get("roeAvg")]), errors="coerce").iloc[0])
                debt = float(pd.to_numeric(pd.Series([balance_row.get("liabilityToAsset")]), errors="coerce").iloc[0])
                yoy_ni = float(pd.to_numeric(pd.Series([growth_row.get("YOYNI")]), errors="coerce").iloc[0])
                quality_score = clamp_score(clamp_score(roe * 4) * 0.6 + clamp_score(100 - debt) * 0.4)
                growth_score = clamp_score((yoy_ni + 20) * 2)
                total_score = clamp_score(value_score * 0.5 + quality_score * 0.35 + growth_score * 0.15)

                dividend_years = 0
                current_year = datetime.now().year
                for year in range(current_year - 4, current_year + 1):
                    try:
                        ddf = query_baostock_table(
                            bs.query_dividend_data,
                            code=normalize_bs_code(symbol),
                            year=str(year),
                            yearType="report",
                        )
                        if ddf.empty or "dividCashPsBeforeTax" not in ddf.columns:
                            continue
                        dv = pd.to_numeric(ddf["dividCashPsBeforeTax"], errors="coerce").dropna()
                        if not dv.empty and float(dv.iloc[0]) > 0:
                            dividend_years += 1
                    except Exception:
                        continue

                hard_pass = (
                    pe_pct <= params.max_pe_percentile
                    and pb_pct <= params.max_pb_percentile
                    and roe >= params.min_roe
                    and debt <= params.max_debt_ratio
                    and yoy_ni >= params.min_yoy_net_profit
                    and dividend_years >= params.min_dividend_years_5y
                )
                if not hard_pass:
                    continue

                close_series = pd.to_numeric(val_df["close"], errors="coerce").dropna()
                if close_series.empty:
                    continue
                current_price = float(close_series.iloc[-1])
                ret = close_series.pct_change().dropna()
                annual_vol = float(ret.std() * (252 ** 0.5)) if not ret.empty else float("nan")
                market_cap = cap_map.get(symbol)
                cap_ratio = (
                    float(market_cap / anchor_market_cap)
                    if (market_cap and anchor_market_cap and anchor_market_cap > 0)
                    else float("nan")
                )

                vol_ok = pd.notna(annual_vol) and annual_vol <= params.max_annual_volatility
                cap_ok = True
                if anchor_market_cap and anchor_market_cap > 0 and market_cap and market_cap > 0:
                    cap_ok = params.market_cap_ratio_min <= cap_ratio <= params.market_cap_ratio_max
                if not (vol_ok and cap_ok):
                    continue
                scored.append({
                    **c,
                    "current_price": current_price,
                    "pe_pct": pe_pct,
                    "pb_pct": pb_pct,
                    "value_score": value_score,
                    "quality_score": quality_score,
                    "growth_score": growth_score,
                    "total_score": total_score,
                    "current_pe": float(pe_series.iloc[-1]),
                    "roe": roe,
                    "debt": debt,
                    "yoy_ni": yoy_ni,
                    "dividend_years_5y": dividend_years,
                    "market_cap": market_cap,
                    "cap_ratio": cap_ratio,
                    "annual_vol": annual_vol,
                    "hard_pass": "YES" if hard_pass else "NO",
                })
            except Exception:
                continue

        if not scored:
            return "No undervalued candidates found with current filters."

        scored_df = pd.DataFrame(scored).sort_values(
            by=["value_score", "total_score"], ascending=[False, False]
        ).head(params.top_n)

        picks = scored_df.to_dict(orient="records")
        summary = scored_df.rename(columns={
            "symbol": "Symbol",
            "name": "Name",
            "industry": "Industry",
            "current_price": "Current Price",
            "current_pe": "Current PE",
            "pe_pct": "PE Percentile(%)",
            "pb_pct": "PB Percentile(%)",
            "roe": "ROE(%)",
            "debt": "Debt Ratio(%)",
            "yoy_ni": "YoY Net Profit(%)",
            "dividend_years_5y": "Dividend Years (5Y)",
            "market_cap": "Market Cap",
            "cap_ratio": "MCap/Anchor",
            "annual_vol": "Annual Vol",
            "value_score": "Value Score",
            "quality_score": "Quality Score",
            "growth_score": "Growth Score",
            "total_score": "Total Score",
            "hard_pass": "Hard Filter Pass",
        })[[
            "Symbol", "Name", "Industry", "Current Price", "Current PE",
            "PE Percentile(%)", "PB Percentile(%)",
            "ROE(%)", "Debt Ratio(%)", "YoY Net Profit(%)", "Dividend Years (5Y)",
            "Market Cap", "MCap/Anchor", "Annual Vol",
            "Value Score", "Quality Score", "Growth Score", "Total Score",
            "Hard Filter Pass",
        ]]
        for col in [
            "Current Price", "Current PE", "PE Percentile(%)", "PB Percentile(%)",
            "ROE(%)", "Debt Ratio(%)", "YoY Net Profit(%)",
            "Market Cap", "MCap/Anchor", "Annual Vol",
            "Value Score", "Quality Score", "Growth Score", "Total Score",
        ]:
            summary[col] = pd.to_numeric(summary[col], errors="coerce").round(2)

        cfg_df = pd.DataFrame([{
            "Anchor": params.anchor_symbol,
            "Anchor Industry": anchor_industry or "N/A",
            "Strict Same Industry": params.strict_same_industry,
            "Top N": params.top_n,
            "Max PE Percentile": params.max_pe_percentile,
            "Max PB Percentile": params.max_pb_percentile,
            "Min ROE(%)": params.min_roe,
            "Max Debt Ratio(%)": params.max_debt_ratio,
            "Min YoY Net Profit(%)": params.min_yoy_net_profit,
            "Min Dividend Years (5Y)": params.min_dividend_years_5y,
            "MCap Ratio Min": params.market_cap_ratio_min,
            "MCap Ratio Max": params.market_cap_ratio_max,
            "Max Annual Vol": params.max_annual_volatility,
        }])

        result = format_data(cfg_df, title="Selection Constraints")
        result += "\n\n" + format_data(
            summary,
            title=f"Value Candidates Similar to {params.anchor_symbol} (Top {len(picks)})",
        )

        reasons: List[Dict[str, Any]] = []
        for item in picks:
            tags: List[str] = []
            if float(item.get("pe_pct", 100)) <= 30:
                tags.append("PE分位低")
            if float(item.get("pb_pct", 100)) <= 30:
                tags.append("PB分位低")
            if float(item.get("roe", 0)) >= 12:
                tags.append("ROE较高")
            if float(item.get("dividend_years_5y", 0)) >= 4:
                tags.append("分红稳定")
            if float(item.get("annual_vol", 9)) <= 0.35:
                tags.append("波动较低")
            if not tags:
                tags.append("综合评分达标")
            reasons.append({
                "Symbol": item.get("symbol"),
                "Why Selected": "、".join(tags),
            })
        result += "\n\n" + format_data(pd.DataFrame(reasons), title="Selection Reasons")

        # 3) Build grid strategy for each selected stock.
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
                )
                hdf = hdf.tail(params.history_days)
                if hdf.empty:
                    raise RuntimeError("empty history")
                year_high = float(pd.to_numeric(hdf["high"], errors="coerce").dropna().max())
                year_low = float(pd.to_numeric(hdf["low"], errors="coerce").dropna().min())
            except Exception:
                hdf = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_grid,
                    adjust="hfq",
                ).tail(params.history_days)
                if hdf.empty:
                    continue
                year_high = float(pd.to_numeric(hdf["最高"], errors="coerce").dropna().max())
                year_low = float(pd.to_numeric(hdf["最低"], errors="coerce").dropna().min())

            grids = []
            p = current_price
            for i in range(1, params.grid_count + 1):
                buy_price = p * (1 - params.grid_step_pct)
                sell_target = buy_price * (1 + params.grid_step_pct)
                grids.append({
                    "Level": f"Buy #{i}",
                    "Buy Price": round(buy_price, 2),
                    "Sell Target": round(sell_target, 2),
                    "Distance from Current": f"-{round(i * params.grid_step_pct * 100, 1)}%",
                })
                p = buy_price

            header_df = pd.DataFrame([{
                "Symbol": symbol,
                "Current Price": round(current_price, 2),
                "Range Low": round(year_low, 2),
                "Range High": round(year_high, 2),
                "Grid Step(%)": round(params.grid_step_pct * 100, 2),
            }])
            result += "\n\n" + format_data(header_df, title=f"Grid Context for {symbol}")
            result += "\n\n" + format_data(pd.DataFrame(grids), title=f"Grid Plan for {symbol}")

        return result
    except Exception as e:
        return handle_error(e, "get_value_candidates_and_grid")
