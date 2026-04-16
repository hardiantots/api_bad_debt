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
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.engine import URL, Engine

logger = logging.getLogger("bad_debt_api")

if TYPE_CHECKING:
    from bad_debt_app.feature_engineering.pipeline import RawInputFrames

# Load .env from project root (Model API/)
_BASE_DIR = Path(__file__).resolve().parent.parent.parent
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
    port_raw = os.getenv("DB_PORT", "3306").strip().strip("'\"")
    db_name = os.getenv("DB_NAME", "").strip().strip("'\"")

    missing = [
        key
        for key, value in {
            "DB_USER": user,
            "DB_PASSWORD": password,
            "DB_HOST": host,
            "DB_NAME": db_name,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing required DB env vars: {', '.join(missing)}. Check .env file."
        )

    if not re.fullmatch(r"\d+", port_raw):
        raise RuntimeError("DB_PORT must be numeric.")
    port = int(port_raw)

    connection_url = URL.create(
        drivername="mysql+pymysql",
        username=user,
        password=password,
        host=host,
        port=port,
        database=db_name,
    )
    try:
        return create_engine(
            connection_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
        )
    except Exception as exc:
        logger.exception("Failed to create SQLAlchemy engine")
        raise RuntimeError("Database engine creation failed.") from exc


# ── Query helpers ─────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_data_date_range() -> dict[str, str]:
    """Fetch the absolute minimum and maximum invoice dates in the database (YYYY-MM-DD)."""
    try:
        engine = get_engine()
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
    try:
        return pd.read_sql(text(query), eng, params=params)
    except Exception as exc:
        logger.exception("Failed to fetch invoices")
        raise RuntimeError("Invoice query failed.") from exc


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
    try:
        return pd.read_sql(text(query), eng, params=params)
    except Exception as exc:
        logger.exception("Failed to fetch receipts")
        raise RuntimeError("Receipt query failed.") from exc


def fetch_all_invoices(engine: Optional[Engine] = None) -> pd.DataFrame:
    """Fetch ALL invoices (no time filter) for historical feature engineering."""
    eng = engine or get_engine()
    try:
        return pd.read_sql(text("SELECT * FROM ar_invoice_list_2"), eng)
    except Exception as exc:
        logger.exception("Failed to fetch all invoices")
        raise RuntimeError("Full invoice query failed.") from exc


def fetch_all_receipts(engine: Optional[Engine] = None) -> pd.DataFrame:
    """Fetch ALL receipts (no time filter) for historical feature engineering."""
    eng = engine or get_engine()
    try:
        return pd.read_sql(text("SELECT * FROM ar_receipt_list"), eng)
    except Exception as exc:
        logger.exception("Failed to fetch all receipts")
        raise RuntimeError("Full receipt query failed.") from exc


def fetch_customers(engine: Optional[Engine] = None) -> pd.DataFrame:
    """Fetch customer data from OracleCustomer table (slim columns)."""
    eng = engine or get_engine()
    query = text(
        """
        SELECT PARTY_ID, ACCOUNT_NUMBER, TRX_NUMBER, CUSTOMER_NAME
        FROM OracleCustomer
    """
    )
    try:
        return pd.read_sql(query, eng)
    except Exception as exc:
        logger.exception("Failed to fetch customers")
        raise RuntimeError("Customer query failed.") from exc


def fetch_raw_inputs(
    time_range: str = "1w",
    year: int = 2026,
    start_date: str = None,
    end_date: str = None,
    snapshot_date: str = None,
) -> RawInputFrames:
    """Fetch invoice + receipt + customer from DB via modular two-pass strategy."""
    from bad_debt_app.feature_engineering.pipeline import RawInputFrames
    from bad_debt_app.data.db_two_pass import fetch_raw_inputs_two_pass

    return fetch_raw_inputs_two_pass(
        time_range=time_range,
        year=year,
        start_date=start_date,
        end_date=end_date,
        snapshot_date=snapshot_date,
        get_engine=get_engine,
        fetch_customers=fetch_customers,
        fetch_invoices=fetch_invoices,
        fetch_receipts=fetch_receipts,
        raw_frames_cls=RawInputFrames,
    )


def _table_columns(engine: Engine, table_name: str) -> set[str]:
    inspector = inspect(engine)
    names = {c.get("name") for c in inspector.get_columns(table_name)}
    return {n for n in names if n}


def publish_score_to_hasil_baddebt(
    *,
    df_score: pd.DataFrame,
    model_key: str,
    snapshot_date: str,
    time_range: str,
    source_job_id: str,
    target_table: str = "hasil_baddebt",
    replace_partition: bool = True,
) -> dict:
    """Publish invoice-level compute output to MySQL table hasil_baddebt.

    Uses the same DB credentials/engine as invoice retrieval.
    """
    if df_score is None or df_score.empty:
        return {
            "table": target_table,
            "rows_inserted": 0,
            "rows_deleted": 0,
            "message": "No score rows to publish.",
        }

    if not re.fullmatch(r"[A-Za-z0-9_]+", target_table):
        raise RuntimeError("Invalid target table name.")

    engine = get_engine()
    inspector = inspect(engine)
    if not inspector.has_table(target_table):
        raise RuntimeError(
            f"Target table '{target_table}' not found in database '{engine.url.database}'."
        )

    rec = df_score.copy()
    # Keep backward-compatible metadata columns that may exist in target schema.
    rec["job_id"] = source_job_id
    rec["model_key"] = model_key
    rec["snapshot_date"] = snapshot_date
    rec["time_range"] = time_range

    # Preferred source_* metadata columns for publish tracking.
    rec["source_job_id"] = source_job_id
    rec["source_model_key"] = model_key
    rec["source_snapshot_date"] = snapshot_date
    rec["source_time_range"] = time_range

    # Add created timestamp if target schema supports it.
    rec["published_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    for c in rec.columns:
        lower_c = c.lower()
        if lower_c in {"id", "job_id", "source_job_id"}:
            continue
        if lower_c.endswith("_id") or lower_c in {
            "days_to_due",
            "trx_amount",
            "trx_amount_gross",
            "credit_memo_reduction",
            "prob_bad_debt",
            "expected_financial_loss",
        }:
            rec[c] = pd.to_numeric(rec[c], errors="coerce")

    rec = rec.replace([float("inf"), float("-inf")], None)
    rec = rec.where(pd.notnull(rec), None)

    target_cols = _table_columns(engine, target_table)
    publish_cols = [c for c in rec.columns if c in target_cols and c.lower() != "id"]
    if not publish_cols:
        raise RuntimeError(
            f"No matching columns between compute output and target table '{target_table}'."
        )

    rows_deleted = 0
    replace_skipped_reason = None
    with engine.begin() as conn:
        if replace_partition:
            required_partition_cols = {
                "source_model_key",
                "source_snapshot_date",
                "source_time_range",
            }
            if required_partition_cols.issubset(target_cols):
                params = {
                    "mk": model_key,
                    "sd": snapshot_date,
                    "tr": time_range,
                }
                try:
                    result = conn.execute(
                        text(
                            f"DELETE FROM {target_table} "
                            "WHERE source_model_key=:mk "
                            "AND source_snapshot_date=:sd "
                            "AND source_time_range=:tr"
                        ),
                        params,
                    )
                    rows_deleted = int(result.rowcount or 0)
                except OperationalError as exc:
                    err_text = str(exc).lower()
                    if "delete command denied" in err_text:
                        replace_skipped_reason = "replace_partition skipped because DB user has no DELETE privilege"
                        logger.warning(
                            "Publish append-only fallback for table %s: DELETE denied",
                            target_table,
                        )
                    else:
                        raise
            else:
                replace_skipped_reason = (
                    "replace_partition skipped because target table lacks "
                    "source_model_key/source_snapshot_date/source_time_range columns"
                )

        rec[publish_cols].to_sql(
            target_table,
            conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )

    summary = {
        "table": target_table,
        "rows_inserted": int(rec.shape[0]),
        "rows_deleted": rows_deleted,
        "replace_partition": replace_partition,
        "columns_used": publish_cols,
    }
    if replace_skipped_reason:
        summary["replace_partition_note"] = replace_skipped_reason
    return summary


_MYSQL_ALLOWED_SCORE_SORT = {
    "prob_bad_debt",
    "expected_financial_loss",
    "TRX_AMOUNT",
    "CUSTOMER_NAME",
    "TRX_DATE",
    "DUE_DATE",
    "days_to_due",
    "risk_level",
    "CUSTOMER_TRX_ID",
}


def _validate_table_name(table_name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_]+", table_name):
        raise RuntimeError("Invalid target table name.")


def _build_sql_date_filter(
    time_range: str,
    snapshot_date: str,
    custom_start: str | None = None,
    custom_end: str | None = None,
) -> tuple[str, dict]:
    sql = ""
    params = {}
    if time_range == "all":
        return sql, params

    if time_range == "custom" and custom_start and custom_end:
        sql = " AND STR_TO_DATE(TRX_DATE, '%Y-%m-%d') BETWEEN STR_TO_DATE(:dt_st, '%Y-%m-%d') AND STR_TO_DATE(:dt_en, '%Y-%m-%d') "
        params["dt_st"] = custom_start
        params["dt_en"] = custom_end
        return sql, params

    days = _INTERVAL_MAP.get(time_range, 0)
    if days > 0:
        sql = " AND STR_TO_DATE(TRX_DATE, '%Y-%m-%d') >= DATE_SUB(STR_TO_DATE(:dt_sd, '%Y-%m-%d'), INTERVAL :dt_days DAY) "
        params["dt_sd"] = snapshot_date
        params["dt_days"] = days
    return sql, params


def query_mysql_score_results(
    *,
    snapshot_date: str,
    job_id: str,
    time_range: str,
    custom_start: str | None = None,
    custom_end: str | None = None,
    page: int,
    page_size: int,
    sort_by: str,
    sort_order: str,
    risk_level: str | None,
    search: str | None,
    target_table: str = "hasil_baddebt",
) -> tuple[list[dict], int]:
    _validate_table_name(target_table)
    if sort_by not in _MYSQL_ALLOWED_SCORE_SORT:
        sort_by = "prob_bad_debt"
    sort_order = (
        "desc" if sort_order.lower() not in ("asc", "desc") else sort_order.lower()
    )

    date_filter, dt_params = _build_sql_date_filter(
        time_range, snapshot_date, custom_start, custom_end
    )
    where = ["COALESCE(source_job_id, job_id)=:jid " + date_filter]
    params: dict[str, object] = {"jid": job_id, **dt_params}
    if risk_level in ("HIGH", "MEDIUM", "LOW"):
        where.append("risk_level=:rl")
        params["rl"] = risk_level
    if search:
        where.append("CUSTOMER_NAME LIKE :s")
        params["s"] = f"%{search}%"
    wc = " AND ".join(where)

    cols = (
        "CUSTOMER_TRX_ID,ACCOUNT_NUMBER,CUSTOMER_NAME,"
        "TRX_DATE,DUE_DATE,days_to_due,"
        "TRX_AMOUNT,TRX_AMOUNT_GROSS,credit_memo_reduction,"
        "prob_bad_debt,risk_level,recommended_action,expected_financial_loss"
    )

    engine = get_engine()
    with engine.connect() as conn:
        total = (
            conn.execute(
                text(
                    f"WITH ranked AS (SELECT CUSTOMER_TRX_ID, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_TRX_ID ORDER BY id DESC) as _rn FROM {target_table} WHERE {wc}) SELECT COUNT(*) FROM ranked WHERE _rn = 1"
                ),
                params,
            ).scalar()
            or 0
        )
        params["lim"] = page_size
        params["off"] = (page - 1) * page_size
        rows = (
            conn.execute(
                text(
                    f"WITH ranked AS (SELECT {cols}, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_TRX_ID ORDER BY id DESC) as _rn FROM {target_table} WHERE {wc}) SELECT {cols} FROM ranked WHERE _rn = 1 ORDER BY {sort_by} {sort_order} LIMIT :lim OFFSET :off"
                ),
                params,
            )
            .mappings()
            .fetchall()
        )
    return [dict(r) for r in rows], int(total)


def query_mysql_top_efl(
    *,
    snapshot_date: str,
    job_id: str,
    time_range: str,
    custom_start: str | None = None,
    custom_end: str | None = None,
    top_n: int = 50,
    target_table: str = "hasil_baddebt",
) -> list[dict]:
    _validate_table_name(target_table)
    cols = (
        "CUSTOMER_TRX_ID,ACCOUNT_NUMBER,CUSTOMER_NAME,"
        "TRX_DATE,DUE_DATE,days_to_due,"
        "TRX_AMOUNT,TRX_AMOUNT_GROSS,credit_memo_reduction,"
        "prob_bad_debt,risk_level,recommended_action,expected_financial_loss"
    )
    engine = get_engine()
    date_filter, dt_params = _build_sql_date_filter(
        time_range, snapshot_date, custom_start, custom_end
    )
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    f"WITH ranked AS (SELECT {cols}, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_TRX_ID ORDER BY id DESC) as _rn "
                    f"FROM {target_table} "
                    f"WHERE COALESCE(source_job_id, job_id)=:jid {date_filter} "
                    "AND expected_financial_loss IS NOT NULL) "
                    f"SELECT {cols} FROM ranked WHERE _rn = 1 "
                    "ORDER BY expected_financial_loss DESC LIMIT :n"
                ),
                {"jid": job_id, "n": top_n, **dt_params},
            )
            .mappings()
            .fetchall()
        )
    return [dict(r) for r in rows]


def query_mysql_alerts(
    *,
    snapshot_date: str,
    job_id: str,
    time_range: str,
    custom_start: str | None = None,
    custom_end: str | None = None,
    threshold: float,
    page: int,
    page_size: int,
    sort_by: str,
    sort_order: str,
    search: str | None,
    target_table: str = "hasil_baddebt",
) -> tuple[list[dict], int]:
    _validate_table_name(target_table)
    if sort_by not in {
        "prob_bad_debt",
        "expected_financial_loss",
        "TRX_AMOUNT",
        "CUSTOMER_NAME",
    }:
        sort_by = "prob_bad_debt"
    sort_order = (
        "desc" if sort_order.lower() not in ("asc", "desc") else sort_order.lower()
    )

    date_filter, dt_params = _build_sql_date_filter(
        time_range, snapshot_date, custom_start, custom_end
    )
    where = ["COALESCE(source_job_id, job_id)=:jid " + date_filter]
    params: dict[str, object] = {"jid": job_id, **dt_params}
    if search:
        where.append("CUSTOMER_NAME LIKE :s")
        params["s"] = f"%{search}%"
    wc = " AND ".join(where)

    cols = (
        "CUSTOMER_TRX_ID,ACCOUNT_NUMBER,CUSTOMER_NAME,"
        "TRX_DATE,DUE_DATE,days_to_due,"
        "TRX_AMOUNT,TRX_AMOUNT_GROSS,credit_memo_reduction,"
        "prob_bad_debt,risk_level,recommended_action,expected_financial_loss"
    )

    engine = get_engine()
    with engine.connect() as conn:
        total = (
            conn.execute(
                text(
                    f"WITH ranked AS (SELECT CUSTOMER_TRX_ID, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_TRX_ID ORDER BY id DESC) as _rn FROM {target_table} WHERE {wc}) SELECT COUNT(*) FROM ranked WHERE _rn = 1"
                ),
                params,
            ).scalar()
            or 0
        )
        params["lim"] = page_size
        params["off"] = (page - 1) * page_size
        rows = (
            conn.execute(
                text(
                    f"WITH ranked AS (SELECT {cols}, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_TRX_ID ORDER BY id DESC) as _rn FROM {target_table} WHERE {wc}) SELECT {cols} FROM ranked WHERE _rn = 1 ORDER BY {sort_by} {sort_order} LIMIT :lim OFFSET :off"
                ),
                params,
            )
            .mappings()
            .fetchall()
        )
    return [dict(r) for r in rows], int(total)


def fetch_mysql_scores_df(
    *,
    snapshot_date: str,
    job_id: str,
    time_range: str,
    custom_start: str | None = None,
    custom_end: str | None = None,
    target_table: str = "hasil_baddebt",
) -> pd.DataFrame:
    _validate_table_name(target_table)
    engine = get_engine()
    date_filter, dt_params = _build_sql_date_filter(
        time_range, snapshot_date, custom_start, custom_end
    )
    return pd.read_sql(
        text(
            f"WITH ranked AS (SELECT *, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_TRX_ID ORDER BY id DESC) as _rn FROM {target_table} "
            f"WHERE COALESCE(source_job_id, job_id)=:jid {date_filter}) SELECT * FROM ranked WHERE _rn = 1"
        ),
        engine,
        params={"jid": job_id, **dt_params},
    )


# ── TTL cache for get_latest_mysql_source_job ─────────────────────────
# Avoids a heavy GROUP-BY full-table-scan on every filter change.
_mysql_job_cache: dict[tuple, tuple] = {}
_mysql_job_cache_lock = threading.Lock()
_MYSQL_JOB_CACHE_TTL = 30  # seconds


def _lookup_mysql_source_job(
    *,
    engine,
    model_key: str,
    snapshot_date: str,
    time_range: str,
    target_table: str,
) -> dict | None:
    """Internal uncached implementation of the MySQL source-job lookup."""
    req_days = _INTERVAL_MAP.get(time_range, 0) if time_range != "all" else 9999

    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    f"SELECT "
                    f"COALESCE(source_job_id, job_id) AS source_job_id, "
                    f"COALESCE(source_time_range, time_range) AS source_time_range, "
                    f"COALESCE(source_snapshot_date, snapshot_date) AS source_snapshot_date, "
                    f"published_at AS last_published_at, "
                    f"COUNT(*) AS total_invoices "
                    f"FROM {target_table} "
                    f"WHERE COALESCE(source_model_key, model_key)=:mk "
                    f"AND COALESCE(source_snapshot_date, snapshot_date)=:sd "
                    f"GROUP BY COALESCE(source_job_id, job_id), "
                    f"COALESCE(source_time_range, time_range), "
                    f"COALESCE(source_snapshot_date, snapshot_date), published_at "
                    f"ORDER BY last_published_at DESC"
                ),
                {"mk": model_key, "sd": snapshot_date},
            )
            .mappings()
            .fetchall()
        )

    # Prefer job whose stored range covers the requested range
    for row in rows:
        job_tr = row["source_time_range"]
        if time_range == "custom":
            if job_tr == "custom":
                return dict(row)
        elif job_tr == "all":
            return dict(row)
        elif job_tr in _INTERVAL_MAP and time_range in _INTERVAL_MAP:
            if _INTERVAL_MAP[job_tr] >= req_days:
                return dict(row)

    # Exact-match fallback (covers cases like time_range=="custom" with no match above)
    with engine.connect() as conn:
        row = (
            conn.execute(
                text(
                    f"SELECT "
                    f"COALESCE(source_job_id, job_id) AS source_job_id, "
                    f"COALESCE(source_snapshot_date, snapshot_date) AS source_snapshot_date, "
                    f"MAX(published_at) AS last_published_at "
                    f"FROM {target_table} "
                    f"WHERE COALESCE(source_model_key, model_key)=:mk "
                    f"AND COALESCE(source_snapshot_date, snapshot_date)=:sd "
                    f"AND COALESCE(source_time_range, time_range)=:tr "
                    f"GROUP BY COALESCE(source_job_id, job_id), "
                    f"COALESCE(source_snapshot_date, snapshot_date) "
                    f"ORDER BY last_published_at DESC LIMIT 1"
                ),
                {"mk": model_key, "sd": snapshot_date, "tr": time_range},
            )
            .mappings()
            .fetchone()
        )
    return dict(row) if row else None


def get_latest_mysql_source_job(
    *,
    model_key: str,
    snapshot_date: str,
    time_range: str,
    target_table: str = "hasil_baddebt",
) -> dict | None:
    """Return the most recent published MySQL job matching the given parameters.

    Results are cached for _MYSQL_JOB_CACHE_TTL seconds to avoid repeated
    heavy GROUP-BY full-table-scans on every filter change from the frontend.
    """
    _validate_table_name(target_table)
    cache_key = (model_key, snapshot_date, time_range, target_table)
    now = time.monotonic()

    with _mysql_job_cache_lock:
        if cache_key in _mysql_job_cache:
            cached_result, cached_at = _mysql_job_cache[cache_key]
            if now - cached_at < _MYSQL_JOB_CACHE_TTL:
                return cached_result

    result = _lookup_mysql_source_job(
        engine=get_engine(),
        model_key=model_key,
        snapshot_date=snapshot_date,
        time_range=time_range,
        target_table=target_table,
    )

    with _mysql_job_cache_lock:
        _mysql_job_cache[cache_key] = (result, now)

    return result


def query_mysql_risk_summary(
    *,
    snapshot_date: str,
    job_id: str,
    time_range: str,
    custom_start: str | None = None,
    custom_end: str | None = None,
    target_table: str = "hasil_baddebt",
) -> dict[str, int]:
    _validate_table_name(target_table)
    engine = get_engine()
    date_filter, dt_params = _build_sql_date_filter(
        time_range, snapshot_date, custom_start, custom_end
    )
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    f"SELECT risk_level, COUNT(*) AS cnt FROM {target_table} "
                    f"WHERE COALESCE(source_job_id, job_id)=:jid {date_filter} "
                    "GROUP BY risk_level"
                ),
                {"jid": job_id, **dt_params},
            )
            .mappings()
            .fetchall()
        )
    return {str(r["risk_level"]): int(r["cnt"] or 0) for r in rows if r["risk_level"]}


def query_mysql_chart_data(
    *,
    job_id: str,
    snapshot_date: str,
    time_range: str,
    custom_start: str | None = None,
    custom_end: str | None = None,
    target_table: str = "hasil_baddebt",
) -> list[dict]:
    _validate_table_name(target_table)
    engine = get_engine()

    date_filter, dt_params = _build_sql_date_filter(
        time_range, snapshot_date, custom_start, custom_end
    )
    params = {"jid": job_id, **dt_params}

    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    "WITH ranked AS (SELECT CUSTOMER_TRX_ID, ACCOUNT_NUMBER, CUSTOMER_NAME, "
                    "prob_bad_debt, expected_financial_loss, risk_level, recommended_action, ROW_NUMBER() OVER(PARTITION BY CUSTOMER_TRX_ID ORDER BY id DESC) as _rn "
                    f"FROM {target_table} "
                    "WHERE COALESCE(source_job_id, job_id)=:jid " + date_filter + ") "
                    "SELECT CUSTOMER_TRX_ID, ACCOUNT_NUMBER, CUSTOMER_NAME, prob_bad_debt, "
                    "expected_financial_loss, risk_level, recommended_action "
                    "FROM ranked WHERE _rn = 1"
                ),
                params,
            )
            .mappings()
            .fetchall()
        )
    return [dict(r) for r in rows]


# ── MySQL cleanup ─────────────────────────────────────────────────────


def cleanup_mysql_old_scoring_results(
    *,
    target_table: str = "hasil_baddebt",
    keep_days: int = 1,
) -> int:
    """Delete rows from MySQL target table where published_at is older than keep_days.

    Called after a new scoring job completes to keep the DB lean.
    Returns the number of rows deleted (or -1 if DELETE privilege is not available).
    """
    _validate_table_name(target_table)
    if keep_days < 0:
        return 0

    try:
        engine = get_engine()
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    f"DELETE FROM {target_table} "
                    "WHERE published_at < DATE_SUB(NOW(), INTERVAL :days DAY)"
                ),
                {"days": keep_days},
            )
            deleted = int(result.rowcount or 0)
            if deleted > 0:
                logger.info(
                    "MySQL cleanup: removed %d old row(s) from %s (keep_days=%d)",
                    deleted,
                    target_table,
                    keep_days,
                )
            return deleted
    except Exception as exc:
        err_text = str(exc).lower()
        if "delete command denied" in err_text:
            logger.warning(
                "MySQL cleanup skipped for %s: DELETE privilege not granted to DB user.",
                target_table,
            )
            return -1
        logger.exception("MySQL cleanup failed for table %s", target_table)
        return -1
