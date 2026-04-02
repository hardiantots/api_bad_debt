from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import IO, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("bad_debt_api")


@dataclass(frozen=True)
class RawInputFrames:
    invoice: pd.DataFrame
    receipt: pd.DataFrame
    customer: Optional[pd.DataFrame] = None
    target_trx_ids: Optional[list] = None


def _to_dt_naive(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)
    failed = int((dt.isna() & s.notna()).sum())
    if failed > 0:
        logger.warning("Datetime parsing coerced %d rows to NaT", failed)
    return dt


def fix_year_month(s: pd.Series, snapshot_date: pd.Timestamp) -> pd.Series:
    s = _to_dt_naive(s)

    y = s.dt.year
    m = s.dt.month
    d = s.dt.day

    y = np.where((y >= 2200) & (y < 2300), y - 200, y)
    y = np.where((y > snapshot_date.year) & (y <= snapshot_date.year + 20), y - 10, y)

    out = pd.to_datetime({"year": y, "month": m, "day": d}, errors="coerce")
    dropped = int((out.isna() & s.notna()).sum())
    if dropped > 0:
        logger.warning("fix_year_month could not repair %d datetime rows", dropped)

    mask = (
        out.notna()
        & (out > snapshot_date)
        & (out <= snapshot_date + pd.Timedelta(days=45))
    )
    out.loc[mask] = out.loc[mask] - pd.DateOffset(months=1)
    return out


def _infer_format(source_name: str | None) -> str | None:
    if not source_name:
        return None
    name = source_name.lower()
    if name.endswith(".parquet") or name.endswith(".pq"):
        return "parquet"
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".json"):
        return "json"
    if name.endswith(".jsonl"):
        return "jsonl"
    return None


def _read_customer(
    customer_input: IO[bytes] | str,
    *,
    customer_name: str | None = None,
    customer_format: str | None = None,
) -> pd.DataFrame:
    fmt = customer_format or _infer_format(customer_name)

    if fmt == "csv":
        return pd.read_csv(customer_input, low_memory=False)
    if fmt == "parquet":
        try:
            return pd.read_parquet(customer_input)
        except Exception as exc:
            raise ValueError(
                "Failed to read parquet customer file. Install pyarrow or use CSV/JSON instead."
            ) from exc
    if fmt == "jsonl":
        return pd.read_json(customer_input, lines=True)
    return pd.read_json(customer_input)


def load_raw_inputs(
    invoice_csv: IO[bytes] | str,
    receipt_csv: IO[bytes] | str,
    customer_json: IO[bytes] | str | None = None,
    *,
    customer_name: str | None = None,
    customer_format: str | None = None,
) -> RawInputFrames:
    df_invoice = pd.read_csv(invoice_csv, low_memory=False)
    df_receipt = pd.read_csv(receipt_csv, low_memory=False)

    df_customer = None
    if customer_json is not None:
        df_customer = _read_customer(
            customer_json,
            customer_name=customer_name,
            customer_format=customer_format,
        )

    return RawInputFrames(invoice=df_invoice, receipt=df_receipt, customer=df_customer)
