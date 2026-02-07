from datetime import datetime
from typing import Any, Dict, List, Tuple

import baostock as bs
import pandas as pd
from loguru import logger

from core.utils import (
    normalize_bs_adjust,
    normalize_bs_code,
    normalize_bs_frequency,
    parse_date_yyyymmdd,
    safe_to_numeric,
)

_baostock_logged_in = False


def ensure_baostock_login() -> None:
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
    logger.info("Baostock login successful.")


def baostock_result_to_df(result: Any) -> pd.DataFrame:
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
    adjust: str,
) -> pd.DataFrame:
    ensure_baostock_login()
    rs = bs.query_history_k_data_plus(
        normalize_bs_code(symbol),
        ",".join(fields),
        start_date=parse_date_yyyymmdd(start_date),
        end_date=parse_date_yyyymmdd(end_date),
        frequency=normalize_bs_frequency(period),
        adjustflag=normalize_bs_adjust(adjust),
    )
    df = baostock_result_to_df(rs)
    if not df.empty:
        df = safe_to_numeric(df)
    return df


def query_baostock_table(query_func: Any, **kwargs: Any) -> pd.DataFrame:
    ensure_baostock_login()
    rs = query_func(**kwargs)
    df = baostock_result_to_df(rs)
    if not df.empty:
        df = safe_to_numeric(df)
    return df


def latest_quarter_row(symbol: str, query_func: Any) -> Tuple[Dict[str, Any], str]:
    code = normalize_bs_code(symbol)
    current_year = datetime.now().year
    for year in range(current_year, current_year - 6, -1):
        for quarter in [4, 3, 2, 1]:
            try:
                df = query_baostock_table(query_func, code=code, year=year, quarter=quarter)
                if not df.empty:
                    row = safe_to_numeric(df).iloc[0].to_dict()
                    return row, f"{year}Q{quarter}"
            except Exception:
                continue
    raise RuntimeError("No quarterly data found.")
