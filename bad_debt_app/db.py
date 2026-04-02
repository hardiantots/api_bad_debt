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
                    "SELECT MIN(STR_TO_DATE(TRX_DATE, '%d-%b-%Y')), MAX(STR_TO_DATE(TRX_DATE, '%d-%b-%Y')) FROM ar_invoice_list_2"
                )
            ).fetchone()
            if res and res[0] and res[1]:
                return {"min_date": str(res[0])[:10], "max_date": str(res[1])[:10]}
    except Exception as e:
        logger.warning(f"Failed to fetch data date range: {e}")
    # Fallbacks just in case
    return {"min_date": "2020-01-01", "max_date": "2026-03-15"}


def _build_invoice_query(
    time_range: str,
    *,
    year: int = 2026,
    start_date: str = None,
    end_date: str = None,
    snapshot_date: str = None,
) -> tuple[str, dict]:
    """Build SELECT + WHERE for ar_invoice_list_2 with parameterized values."""
    if time_range == "all":
        if snapshot_date:
            return (
                "SELECT * FROM ar_invoice_list_2 WHERE STR_TO_DATE(TRX_DATE, :fmt) <= :snapshot_date",
                {"fmt": "%d-%b-%Y", "snapshot_date": snapshot_date},
            )
        return "SELECT * FROM ar_invoice_list_2", {}
    if time_range == "custom" and start_date and end_date:
        return (
            "SELECT * FROM ar_invoice_list_2 WHERE STR_TO_DATE(TRX_DATE, :fmt) BETWEEN :start_date AND :end_date",
            {"fmt": "%d-%b-%Y", "start_date": start_date, "end_date": end_date},
        )
    if time_range in _INTERVAL_MAP:
        days = _INTERVAL_MAP[time_range]
        latest_date = (
            snapshot_date if snapshot_date else get_data_date_range().get("max_date")
        )
        return (
            "SELECT * FROM ar_invoice_list_2 WHERE STR_TO_DATE(TRX_DATE, :fmt) BETWEEN DATE_SUB(:latest_date, INTERVAL :days DAY) AND :latest_date",
            {"fmt": "%d-%b-%Y", "latest_date": latest_date, "days": days},
        )
    return (
        "SELECT * FROM ar_invoice_list_2 WHERE RIGHT(TRX_DATE, 4) = :year",
        {"year": str(year)},
    )


def _build_receipt_query(
    time_range: str,
    *,
    year: int = 2026,
    start_date: str = None,
    end_date: str = None,
    snapshot_date: str = None,
) -> tuple[str, dict]:
    """Build SELECT + WHERE for ar_receipt_list with parameterized values."""
    if time_range == "all":
        if snapshot_date:
            return (
                "SELECT * FROM ar_receipt_list WHERE "
                "STR_TO_DATE(LEFT(APPLY_DATE, 10), :fmt) <= :snapshot_date "
                "OR STR_TO_DATE(LEFT(RECEIPT_DATE, 10), :fmt) <= :snapshot_date",
                {"fmt": "%Y-%m-%d", "snapshot_date": snapshot_date},
            )
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
        latest_date = (
            snapshot_date if snapshot_date else get_data_date_range().get("max_date")
        )
        return (
            "SELECT * FROM ar_receipt_list WHERE "
            "(STR_TO_DATE(LEFT(APPLY_DATE, 10), :fmt) BETWEEN DATE_SUB(:latest_date, INTERVAL :days DAY) AND :latest_date "
            "OR STR_TO_DATE(LEFT(RECEIPT_DATE, 10), :fmt) BETWEEN DATE_SUB(:latest_date, INTERVAL :days DAY) AND :latest_date)",
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
    snapshot_date: str = None,
) -> pd.DataFrame:
    """Fetch invoice data from ar_invoice_list_2 with time-range filter."""
    eng = engine or get_engine()
    query, params = _build_invoice_query(
        time_range,
        year=year,
        start_date=start_date,
        end_date=end_date,
        snapshot_date=snapshot_date,
    )
    return pd.read_sql(text(query), eng, params=params)


def fetch_receipts(
    time_range: str = "1w",
    year: int = 2026,
    engine: Optional[Engine] = None,
    start_date: str = None,
    end_date: str = None,
    snapshot_date: str = None,
) -> pd.DataFrame:
    """Fetch receipt data from ar_receipt_list with time-range filter."""
    eng = engine or get_engine()
    query, params = _build_receipt_query(
        time_range,
        year=year,
        start_date=start_date,
        end_date=end_date,
        snapshot_date=snapshot_date,
    )
    return pd.read_sql(text(query), eng, params=params)


def fetch_all_invoices(engine: Optional[Engine] = None) -> pd.DataFrame:
    """Fetch ALL invoices (no time filter) for historical feature engineering."""
    eng = engine or get_engine()
    return pd.read_sql(text("SELECT * FROM ar_invoice_list_2"), eng)


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
    snapshot_date: str = None,
) -> "RawInputFrames":
    """Fetch invoice + receipt + customer from DB using a Two-Pass query.

    Phase 1: Finds target invoices based on the selected time-range.
    Phase 2: Fetches full historical data for ONLY the customers present in Phase 1.
    """
    from bad_debt_app.features import RawInputFrames
    import numpy as np

    engine = get_engine()
    df_customer = fetch_customers(engine=engine)

    # 1) Get Target Invoices (the requested time slice)
    df_target_inv = fetch_invoices(
        time_range=time_range,
        year=year,
        engine=engine,
        start_date=start_date,
        end_date=end_date,
        snapshot_date=snapshot_date,
    )

    if df_target_inv.empty:
        return RawInputFrames(
            invoice=pd.DataFrame(), receipt=pd.DataFrame(), customer=df_customer
        )

    # Capture the specific invoices that the user wants to score
    target_trx_ids = df_target_inv["CUSTOMER_TRX_ID"].dropna().unique().tolist()

    if time_range == "all":
        df_receipt = fetch_receipts(
            time_range=time_range,
            year=year,
            engine=engine,
            start_date=start_date,
            end_date=end_date,
            snapshot_date=snapshot_date,
        )
        return RawInputFrames(
            invoice=df_target_inv,
            receipt=df_receipt,
            customer=df_customer,
            target_trx_ids=target_trx_ids,
        )

    # 2) Discover which Customers are involved, dealing with missing PARTY_ID
    df_t = df_target_inv.copy()
    if "PARTY_ID" not in df_t.columns:
        df_t["PARTY_ID"] = np.nan

    missing_party_mask = df_t["PARTY_ID"].isna() | (df_t["PARTY_ID"] == "")
    if (
        missing_party_mask.any()
        and "TRX_NUMBER" in df_t.columns
        and not df_customer.empty
    ):
        mapping = df_customer.set_index("TRX_NUMBER")["PARTY_ID"].to_dict()
        df_t.loc[missing_party_mask, "PARTY_ID"] = df_t.loc[
            missing_party_mask, "TRX_NUMBER"
        ].map(mapping)

    target_parties = df_t["PARTY_ID"].dropna().unique().tolist()

    # 3) Phase 2: Fetch full historical invoices for these customers up to snapshot_date
    if not target_parties:
        df_all_inv = df_target_inv
    else:
        party_list_str = ",".join([f"'{str(p)}'" for p in target_parties])
        snap_cond = ""
        params = {}
        if snapshot_date:
            snap_cond = "AND STR_TO_DATE(TRX_DATE, :fmt) <= :snapshot_date"
            params = {"fmt": "%d-%b-%Y", "snapshot_date": snapshot_date}

        history_query = f"""
            SELECT * FROM ar_invoice_list_2 
            WHERE PARTY_ID IN ({party_list_str}) 
            {snap_cond}
        """
        df_hist_inv = pd.read_sql(text(history_query), engine, params=params)
        df_all_inv = pd.concat([df_target_inv, df_hist_inv]).drop_duplicates(
            subset=["CUSTOMER_TRX_ID"]
        )

    # 4) Fetch full historical receipts for those aggregated invoices
    all_historic_trx_ids = df_all_inv["CUSTOMER_TRX_ID"].dropna().unique().tolist()
    if not all_historic_trx_ids:
        df_all_receipt = pd.DataFrame()
    else:
        # DB limits: chunk to prevent "IN clause too large" errors
        all_receipts = []
        chunk_size = 5000
        for i in range(0, len(all_historic_trx_ids), chunk_size):
            chunk = all_historic_trx_ids[i : i + chunk_size]
            trx_list_str = ",".join([f"'{str(t)}'" for t in chunk])
            snap_cond_rcpt = ""
            params_rcpt = {}
            if snapshot_date:
                snap_cond_rcpt = "AND (STR_TO_DATE(LEFT(APPLY_DATE, 10), :fmt) <= :snapshot_date OR STR_TO_DATE(LEFT(RECEIPT_DATE, 10), :fmt) <= :snapshot_date)"
                params_rcpt = {"fmt": "%Y-%m-%d", "snapshot_date": snapshot_date}

            rcpt_query = f"""
                SELECT * FROM ar_receipt_list 
                WHERE APPLIED_CUSTOMER_TRX_ID IN ({trx_list_str})
                {snap_cond_rcpt}
            """
            chunk_df = pd.read_sql(text(rcpt_query), engine, params=params_rcpt)
            all_receipts.append(chunk_df)

        df_all_receipt = (
            pd.concat(all_receipts).drop_duplicates()
            if all_receipts
            else pd.DataFrame()
        )

    return RawInputFrames(
        invoice=df_all_inv,
        receipt=df_all_receipt,
        customer=df_customer,
        target_trx_ids=target_trx_ids,
    )
