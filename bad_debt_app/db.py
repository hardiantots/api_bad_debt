"""Database connectivity for Bad Debt Early-Warning system.

Reads connection credentials from .env and provides helper functions
to fetch invoice, receipt, and customer data from the MySQL database
with optional time-range filters.

NOTE: Invoice TRX_DATE is stored as 'DD-Mon-YYYY' strings (e.g. '30-Sep-2017').
      Receipt APPLY_DATE/RECEIPT_DATE are ISO strings (e.g. '2018-09-04T00:00:00.000+00:00').
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine

logger = logging.getLogger("bad_debt_api")

# Load .env from project root
_BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BASE_DIR / ".env")

# ── Time-range filter presets ─────────────────────────────────────────
TIME_RANGE_OPTIONS = {
    "1w": "1 minggu terakhir",
    "2w": "2 minggu terakhir",
    "1m": "1 bulan terakhir",
    "3m": "3 bulan terakhir",
    "6m": "6 bulan terakhir",
    "1y": "1 tahun terakhir",
    "all": "Semua data (terlama s/d terbaru)",
    "custom": "Rentang Kustom (Pilih Tanggal)",
}

_INTERVAL_MAP = {
    "1w": 7,
    "2w": 14,
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "1y": 365,
}


# ── Engine ────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create (and cache) the SQLAlchemy engine from env vars."""
    user = os.getenv("DB_USER", "").strip().strip("'\"")
    password = os.getenv("DB_PASSWORD", "").strip()
    host = os.getenv("DB_HOST", "").strip().strip("'\"")
    port = int(os.getenv("DB_PORT", "3306").strip().strip("'\""))
    db_name = os.getenv("DB_NAME", "").strip().strip("'\"")

    connection_url = URL.create(
        drivername="mysql+pymysql",
        username=user,
        password=password,
        host=host,
        port=port,
        database=db_name,
    )
    return create_engine(
        connection_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
    )


# ── Query helpers ─────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_data_date_range() -> dict[str, str]:
    """Fetch the absolute minimum and maximum invoice dates in the database (YYYY-MM-DD)."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            res = conn.execute(
                text(
                    "SELECT MIN(STR_TO_DATE(TRX_DATE, '%d-%b-%Y')), "
                    "MAX(STR_TO_DATE(TRX_DATE, '%d-%b-%Y')) FROM ar_invoice_list"
                )
            ).fetchone()
            if res and res[0] and res[1]:
                return {"min_date": str(res[0])[:10], "max_date": str(res[1])[:10]}
    except Exception as e:
        logger.warning(f"Failed to fetch data date range: {e}")
    # Fallbacks just in case
    return {"min_date": "2020-01-01", "max_date": "2026-02-10"}


def _build_invoice_query(
    time_range: str, *, year: int = 2026, start_date: str = None, end_date: str = None
) -> tuple[str, dict]:
    """Build SELECT + WHERE for ar_invoice_list with parameterized values."""
    if time_range == "all":
        return "SELECT * FROM ar_invoice_list", {}
    if time_range == "custom" and start_date and end_date:
        return (
            "SELECT * FROM ar_invoice_list WHERE STR_TO_DATE(TRX_DATE, :fmt) BETWEEN :start_date AND :end_date",
            {"fmt": "%d-%b-%Y", "start_date": start_date, "end_date": end_date},
        )
    if time_range in _INTERVAL_MAP:
        days = _INTERVAL_MAP[time_range]
        latest_date = get_data_date_range().get("max_date")
        return (
            "SELECT * FROM ar_invoice_list WHERE STR_TO_DATE(TRX_DATE, :fmt) >= DATE_SUB(:latest_date, INTERVAL :days DAY)",
            {"fmt": "%d-%b-%Y", "latest_date": latest_date, "days": days},
        )
    return (
        "SELECT * FROM ar_invoice_list WHERE RIGHT(TRX_DATE, 4) = :year",
        {"year": str(year)},
    )


def _build_receipt_query(
    time_range: str, *, year: int = 2026, start_date: str = None, end_date: str = None
) -> tuple[str, dict]:
    """Build SELECT + WHERE for ar_receipt_list with parameterized values."""
    if time_range == "all":
        return "SELECT * FROM ar_receipt_list", {}
    if time_range == "custom" and start_date and end_date:
        return (
            "SELECT * FROM ar_receipt_list WHERE "
            "(STR_TO_DATE(LEFT(APPLY_DATE, 10), :fmt) BETWEEN :start_date AND :end_date "
            "OR STR_TO_DATE(LEFT(RECEIPT_DATE, 10), :fmt) BETWEEN :start_date AND :end_date)",
            {"fmt": "%Y-%m-%d", "start_date": start_date, "end_date": end_date},
        )
    if time_range in _INTERVAL_MAP:
        days = _INTERVAL_MAP[time_range]
        latest_date = get_data_date_range().get("max_date")
        return (
            "SELECT * FROM ar_receipt_list WHERE "
            "(STR_TO_DATE(LEFT(APPLY_DATE, 10), :fmt) >= DATE_SUB(:latest_date, INTERVAL :days DAY) "
            "OR STR_TO_DATE(LEFT(RECEIPT_DATE, 10), :fmt) >= DATE_SUB(:latest_date, INTERVAL :days DAY))",
            {"fmt": "%Y-%m-%d", "latest_date": latest_date, "days": days},
        )
    return (
        "SELECT * FROM ar_receipt_list WHERE (LEFT(APPLY_DATE, 4) = :year OR LEFT(RECEIPT_DATE, 4) = :year)",
        {"year": str(year)},
    )


def fetch_invoices(
    time_range: str = "1w",
    year: int = 2026,
    engine: Optional[Engine] = None,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """Fetch invoice data from ar_invoice_list with time-range filter."""
    eng = engine or get_engine()
    query, params = _build_invoice_query(
        time_range, year=year, start_date=start_date, end_date=end_date
    )
    return pd.read_sql(text(query), eng, params=params)


def fetch_receipts(
    time_range: str = "1w",
    year: int = 2026,
    engine: Optional[Engine] = None,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """Fetch receipt data from ar_receipt_list with time-range filter."""
    eng = engine or get_engine()
    query, params = _build_receipt_query(
        time_range, year=year, start_date=start_date, end_date=end_date
    )
    return pd.read_sql(text(query), eng, params=params)


def fetch_all_invoices(engine: Optional[Engine] = None) -> pd.DataFrame:
    """Fetch ALL invoices (no time filter) for historical feature engineering."""
    eng = engine or get_engine()
    return pd.read_sql(text("SELECT * FROM ar_invoice_list"), eng)


def fetch_all_receipts(engine: Optional[Engine] = None) -> pd.DataFrame:
    """Fetch ALL receipts (no time filter) for historical feature engineering."""
    eng = engine or get_engine()
    return pd.read_sql(text("SELECT * FROM ar_receipt_list"), eng)


def fetch_customers(engine: Optional[Engine] = None) -> pd.DataFrame:
    """Fetch customer data from OracleCustomer table (slim columns)."""
    eng = engine or get_engine()
    query = text(
        """
        SELECT PARTY_ID, ACCOUNT_NUMBER, TRX_NUMBER, CUSTOMER_NAME
        FROM OracleCustomer
    """
    )
    return pd.read_sql(query, eng)


def fetch_raw_inputs(
    time_range: str = "1w",
    year: int = 2026,
    start_date: str = None,
    end_date: str = None,
) -> "RawInputFrames":
    """Fetch invoice + receipt + customer from DB and return RawInputFrames.

    This is the main entry point for DB-backed inference.
    """
    from bad_debt_app.features import RawInputFrames

    engine = get_engine()
    df_invoice = fetch_invoices(
        time_range=time_range,
        year=year,
        engine=engine,
        start_date=start_date,
        end_date=end_date,
    )
    df_receipt = fetch_receipts(
        time_range=time_range,
        year=year,
        engine=engine,
        start_date=start_date,
        end_date=end_date,
    )
    df_customer = fetch_customers(engine=engine)

    return RawInputFrames(
        invoice=df_invoice,
        receipt=df_receipt,
        customer=df_customer,
    )
