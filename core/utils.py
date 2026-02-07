import json
import os
import sys
from datetime import datetime
from math import sqrt
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from models.inputs import AdjustType, KlinePeriod, ResponseFormat, normalize_symbol

# Configure loguru
logger.remove()  # Remove default handler
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("mcp.log", rotation="10 MB", retention="10 days", level="INFO")

STRATEGY_CACHE_PATH = "prefetch-file-server/data/strategy_cache.json"


BAOSTOCK_FREQUENCY_MAP = {
    KlinePeriod.DAILY.value: "d",
    KlinePeriod.WEEKLY.value: "w",
    KlinePeriod.MONTHLY.value: "m",
}

BAOSTOCK_ADJUST_MAP = {
    AdjustType.NONE.value: "3",
    AdjustType.QFQ.value: "2",
    AdjustType.HFQ.value: "1",
}


def normalize_bs_code(symbol: str) -> str:
    s = symbol.strip().lower()
    if s.startswith("sh.") or s.startswith("sz."):
        return s
    digits = normalize_symbol(s)
    market = "sh" if digits.startswith(("5", "6", "9")) else "sz"
    return f"{market}.{digits}"


def parse_date_yyyymmdd(date_str: str) -> str:
    if not date_str:
        return ""
    if "-" in date_str:
        return date_str
    return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")


def normalize_bs_frequency(period: str) -> str:
    if not period:
        return "d"
    p = period.lower().strip()
    return BAOSTOCK_FREQUENCY_MAP.get(p, p)


def normalize_bs_adjust(adjust: str) -> str:
    if adjust is None:
        return "3"
    a = str(adjust).lower().strip()
    if a in {"1", "2", "3"}:
        return a
    return BAOSTOCK_ADJUST_MAP.get(a, "3")


def safe_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            continue
    return df


def format_data(df: pd.DataFrame, format: ResponseFormat = ResponseFormat.MARKDOWN, title: str = "") -> str:
    if df.empty:
        return "No data found."

    if format == ResponseFormat.JSON:
        return df.to_json(orient="records", force_ascii=False)

    md = ""
    if title:
        md += f"### {title}\n\n"
    md += df.to_markdown(index=False)
    return md


def trim_dataframe(df: pd.DataFrame, max_rows: int = 20, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if columns:
        existing_cols = [c for c in columns if c in df.columns]
        df = df[existing_cols]

    if len(df) > max_rows:
        return df.head(max_rows)
    return df


def handle_error(e: Exception, tool_name: str) -> str:
    logger.exception(f"Error in {tool_name}: {str(e)}")
    return f"Error: Data source temporarily unavailable or invalid request. Details: {str(e)}"


def to_yyyymmdd(value: Optional[str], default_value: str) -> str:
    if not value:
        return default_value
    s = value.strip()
    if "-" in s:
        return datetime.strptime(s, "%Y-%m-%d").strftime("%Y%m%d")
    return s


def compute_equity_metrics(equity_curve: pd.Series) -> Dict[str, float]:
    if equity_curve.empty:
        raise RuntimeError("equity curve is empty")
    equity = equity_curve.dropna()
    if equity.empty:
        raise RuntimeError("equity curve is invalid")
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    daily_ret = equity.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
    periods = max(len(daily_ret), 1)
    annual_return = (1 + total_return) ** (252 / periods) - 1
    annual_vol = float(daily_ret.std()) * sqrt(252) if not daily_ret.empty else 0.0
    sharpe = annual_return / annual_vol if annual_vol > 1e-12 else 0.0
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
    }


def percentile_of_last(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    v = v[v.notna()]
    if v.empty:
        raise RuntimeError("empty series for percentile")
    last = v.iloc[-1]
    return float((v <= last).mean() * 100)


def clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def parse_symbol_list(symbols: str) -> List[str]:
    items = [normalize_symbol(s.strip()) for s in symbols.split(",") if s.strip()]
    ordered: List[str] = []
    seen = set()
    for s in items:
        if s not in seen:
            ordered.append(s)
            seen.add(s)
    if not ordered:
        raise RuntimeError("symbols is empty")
    return ordered


def parse_weight_map(weights: Optional[str]) -> Dict[str, float]:
    if not weights:
        return {}
    result: Dict[str, float] = {}
    for item in weights.split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            raise RuntimeError(f"Invalid weight token: {token}")
        symbol, weight = token.split(":", 1)
        result[normalize_symbol(symbol.strip())] = float(weight.strip())
    total = sum(max(v, 0.0) for v in result.values())
    if total > 0:
        result = {k: max(v, 0.0) / total for k, v in result.items()}
    return result


def apply_weight_cap(raw_weights: pd.Series, max_single_weight: float) -> pd.Series:
    if raw_weights.empty:
        return raw_weights
    weights = raw_weights.copy()
    if max_single_weight <= 0 or max_single_weight >= 1:
        return weights / weights.sum()
    over = weights > max_single_weight
    if not over.any():
        return weights / weights.sum()
    weights[over] = max_single_weight
    remain = 1.0 - weights[over].sum()
    under = ~over
    under_sum = weights[under].sum()
    if remain <= 0 or under_sum <= 0:
        return weights / weights.sum()
    weights[under] = weights[under] / under_sum * remain
    return weights / weights.sum()


def load_cache(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(file_path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
