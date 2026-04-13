"""Local SQLite storage for pre-computed scoring results.

Uses SQLite so results are visible as a .db file in local_data/.
Can be migrated to MySQL later when CREATE TABLE access is available.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

logger = logging.getLogger("bad_debt_api")

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOCAL_DB_DIR = _BASE_DIR / "local_data"
LOCAL_DB_PATH = LOCAL_DB_DIR / "scoring.db"


@lru_cache(maxsize=1)
def get_local_engine() -> Engine:
    """Create and cache the SQLite engine."""
    LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{LOCAL_DB_PATH}",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_conn, connection_record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    return engine


# ── Table creation ────────────────────────────────────────────────────


def ensure_tables():
    """Create tables if they don't exist."""
    engine = get_local_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS bad_debt_compute_jobs (
                job_id          TEXT PRIMARY KEY,
                status          TEXT NOT NULL DEFAULT 'running',
                model_key       TEXT NOT NULL,
                model_label     TEXT,
                model_flow      TEXT,
                label_strategy  TEXT,
                snapshot_date   TEXT NOT NULL,
                time_range      TEXT NOT NULL,
                start_date      TEXT,
                end_date        TEXT,
                total_invoices      INTEGER,
                total_customers     INTEGER,
                risk_summary_json   TEXT,
                customer_risk_summary_json TEXT,
                error_message   TEXT,
                started_at      TEXT DEFAULT (datetime('now','localtime')),
                completed_at    TEXT,
                duration_sec    REAL
            )
        """
            )
        )
        conn.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS bad_debt_score_results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id          TEXT NOT NULL,
                snapshot_date   TEXT NOT NULL,
                time_range      TEXT NOT NULL,
                model_key       TEXT NOT NULL,
                CUSTOMER_TRX_ID INTEGER,
                ACCOUNT_NUMBER  TEXT,
                CUSTOMER_NAME   TEXT,
                TRX_DATE        TEXT,
                DUE_DATE        TEXT,
                days_to_due     REAL,
                TRX_AMOUNT      REAL,
                TRX_AMOUNT_GROSS REAL,
                credit_memo_reduction REAL,
                prob_bad_debt           REAL NOT NULL,
                risk_level              TEXT NOT NULL,
                recommended_action      TEXT,
                expected_financial_loss  REAL,
                created_at      TEXT DEFAULT (datetime('now','localtime'))
            )
        """
            )
        )
        conn.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS bad_debt_customer_risk (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id          TEXT NOT NULL,
                snapshot_date   TEXT NOT NULL,
                time_range      TEXT NOT NULL,
                model_key       TEXT NOT NULL,
                PARTY_ID        TEXT,
                ACCOUNT_NUMBER  TEXT,
                CUSTOMER_NAME   TEXT,
                cust_score_max              REAL,
                cust_score_mean             REAL,
                cust_score_wavg_amount      REAL,
                invoice_cnt                 INTEGER,
                total_amount                REAL,
                total_amt_paid_pre_due      REAL,
                paid_ratio_pre_due_total    REAL,
                paid_ratio_pre_due_mean     REAL,
                pct_invoices_gap_gt_90_pre_due REAL,
                party_prior_invoice_cnt     REAL,
                party_prior_gap90_cnt       REAL,
                party_prior_gap90_rate      REAL,
                risk_cust                   TEXT,
                created_at      TEXT DEFAULT (datetime('now','localtime'))
            )
        """
            )
        )

        for idx in [
            "CREATE INDEX IF NOT EXISTS idx_sr_job ON bad_debt_score_results(job_id)",
            "CREATE INDEX IF NOT EXISTS idx_sr_snap ON bad_debt_score_results(snapshot_date,time_range,model_key)",
            "CREATE INDEX IF NOT EXISTS idx_sr_risk ON bad_debt_score_results(risk_level)",
            "CREATE INDEX IF NOT EXISTS idx_sr_prob ON bad_debt_score_results(prob_bad_debt)",
            "CREATE INDEX IF NOT EXISTS idx_sr_efl ON bad_debt_score_results(expected_financial_loss)",
            "CREATE INDEX IF NOT EXISTS idx_cr_job ON bad_debt_customer_risk(job_id)",
            "CREATE INDEX IF NOT EXISTS idx_cr_snap ON bad_debt_customer_risk(snapshot_date,time_range,model_key)",
            "CREATE INDEX IF NOT EXISTS idx_cr_risk ON bad_debt_customer_risk(risk_cust)",
            "CREATE INDEX IF NOT EXISTS idx_cj_stat ON bad_debt_compute_jobs(status)",
        ]:
            conn.execute(text(idx))
    logger.info("Local SQLite tables ensured at %s", LOCAL_DB_PATH)


# ── Job helpers ───────────────────────────────────────────────────────


def generate_job_id() -> str:
    return str(uuid.uuid4())


def has_running_job() -> bool:
    engine = get_local_engine()
    with engine.connect() as conn:
        n = conn.execute(
            text("SELECT COUNT(*) FROM bad_debt_compute_jobs WHERE status='running'")
        ).scalar()
        return (n or 0) > 0


def recover_stale_running_jobs(
    max_age_minutes: int,
    reason: str = "Auto-recovered stale running job",
) -> int:
    """Mark long-running stale jobs as failed to prevent permanent compute lock."""
    if max_age_minutes <= 0:
        return 0

    engine = get_local_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text(
                "UPDATE bad_debt_compute_jobs SET "
                "status='failed',"
                "error_message=COALESCE(error_message, :reason),"
                "completed_at=datetime('now','localtime'),"
                "duration_sec=COALESCE(duration_sec, (julianday(datetime('now','localtime')) - julianday(started_at)) * 86400.0) "
                "WHERE status='running' AND started_at < datetime('now','localtime', :off)"
            ),
            {"reason": reason, "off": f"-{max_age_minutes} minutes"},
        )
        recovered = result.rowcount or 0
        if recovered > 0:
            logger.warning(
                "Recovered %d stale running compute job(s) older than %d minute(s)",
                recovered,
                max_age_minutes,
            )
        return recovered


def create_job(
    job_id: str,
    *,
    model_key: str,
    model_label: str,
    model_flow: str | None,
    label_strategy: str | None,
    snapshot_date: str,
    time_range: str,
    start_date: str | None = None,
    end_date: str | None = None,
):
    engine = get_local_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO bad_debt_compute_jobs "
                "(job_id,status,model_key,model_label,model_flow,label_strategy,"
                "snapshot_date,time_range,start_date,end_date) "
                "VALUES (:jid,'running',:mk,:ml,:mf,:ls,:sd,:tr,:s,:e)"
            ),
            {
                "jid": job_id,
                "mk": model_key,
                "ml": model_label,
                "mf": model_flow,
                "ls": label_strategy,
                "sd": snapshot_date,
                "tr": time_range,
                "s": start_date,
                "e": end_date,
            },
        )


def complete_job(
    job_id: str,
    *,
    total_invoices: int,
    total_customers: int,
    risk_summary: dict,
    customer_risk_summary: dict,
    duration_sec: float,
):
    engine = get_local_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE bad_debt_compute_jobs SET status='completed',"
                "total_invoices=:ti,total_customers=:tc,"
                "risk_summary_json=:rs,customer_risk_summary_json=:crs,"
                "completed_at=datetime('now','localtime'),duration_sec=:dur "
                "WHERE job_id=:jid"
            ),
            {
                "jid": job_id,
                "ti": total_invoices,
                "tc": total_customers,
                "rs": json.dumps(risk_summary),
                "crs": json.dumps(customer_risk_summary),
                "dur": duration_sec,
            },
        )


def fail_job(job_id: str, error_message: str, duration_sec: float):
    engine = get_local_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE bad_debt_compute_jobs SET status='failed',"
                "error_message=:err,completed_at=datetime('now','localtime'),"
                "duration_sec=:dur WHERE job_id=:jid"
            ),
            {"jid": job_id, "err": error_message, "dur": duration_sec},
        )


def get_job(job_id: str) -> dict | None:
    engine = get_local_engine()
    with engine.connect() as conn:
        row = (
            conn.execute(
                text("SELECT * FROM bad_debt_compute_jobs WHERE job_id=:jid"),
                {"jid": job_id},
            )
            .mappings()
            .fetchone()
        )
        if not row:
            return None
        d = dict(row)
        for k in ("risk_summary_json", "customer_risk_summary_json"):
            if d.get(k):
                d[k.replace("_json", "")] = json.loads(d[k])
        return d


def get_latest_job(
    model_key: str,
    snapshot_date: str,
    time_range: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict | None:
    engine = get_local_engine()
    with engine.connect() as conn:
        where_parts = [
            "model_key=:mk",
            "snapshot_date=:sd",
            "time_range=:tr",
            "status='completed'",
        ]
        params: dict[str, str] = {
            "mk": model_key,
            "sd": snapshot_date,
            "tr": time_range,
        }

        # Prevent accidental cross-selection between different custom date windows.
        if time_range == "custom":
            if start_date is not None:
                where_parts.append("start_date=:start_date")
                params["start_date"] = start_date
            if end_date is not None:
                where_parts.append("end_date=:end_date")
                params["end_date"] = end_date

        where_sql = " AND ".join(where_parts)
        row = (
            conn.execute(
                text(
                    "SELECT * FROM bad_debt_compute_jobs "
                    f"WHERE {where_sql} ORDER BY completed_at DESC LIMIT 1"
                ),
                params,
            )
            .mappings()
            .fetchone()
        )
        if not row:
            return None
        d = dict(row)
        for k in ("risk_summary_json", "customer_risk_summary_json"):
            if d.get(k):
                d[k.replace("_json", "")] = json.loads(d[k])
        return d


def get_all_jobs(limit: int = 20) -> list[dict]:
    engine = get_local_engine()
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    "SELECT job_id,status,model_key,snapshot_date,time_range,"
                    "total_invoices,total_customers,duration_sec,"
                    "started_at,completed_at,error_message "
                    "FROM bad_debt_compute_jobs ORDER BY started_at DESC LIMIT :lim"
                ),
                {"lim": limit},
            )
            .mappings()
            .fetchall()
        )
        return [dict(r) for r in rows]


def _delete_result_rows_by_job_ids(conn, table_name: str, job_ids: list[str]) -> int:
    """Delete rows in a result table for a list of job IDs."""
    if not job_ids:
        return 0
    params = {f"jid_{i}": jid for i, jid in enumerate(job_ids)}
    placeholders = ",".join([f":jid_{i}" for i in range(len(job_ids))])
    result = conn.execute(
        text(f"DELETE FROM {table_name} WHERE job_id IN ({placeholders})"), params
    )
    return int(result.rowcount or 0)


def replace_partition_results_for_job(job_id: str) -> dict[str, int]:
    """Remove old partition rows so latest compute acts as replace-partition.

    Partition key:
    - model_key + snapshot_date + time_range
    - plus start_date + end_date when time_range=custom
    """
    engine = get_local_engine()
    with engine.begin() as conn:
        job = (
            conn.execute(
                text(
                    "SELECT model_key,snapshot_date,time_range,start_date,end_date "
                    "FROM bad_debt_compute_jobs WHERE job_id=:jid"
                ),
                {"jid": job_id},
            )
            .mappings()
            .fetchone()
        )
        if not job:
            return {"score_deleted": 0, "customer_deleted": 0, "jobs_affected": 0}

        where_parts = [
            "model_key=:mk",
            "snapshot_date=:sd",
            "time_range=:tr",
            "job_id<>:jid",
        ]
        params: dict[str, str | None] = {
            "mk": job["model_key"],
            "sd": job["snapshot_date"],
            "tr": job["time_range"],
            "jid": job_id,
        }

        # Keep custom windows isolated; only replace rows from the same exact window.
        if job["time_range"] == "custom":
            where_parts.append(
                "((start_date IS NULL AND :start_date IS NULL) OR start_date=:start_date)"
            )
            where_parts.append(
                "((end_date IS NULL AND :end_date IS NULL) OR end_date=:end_date)"
            )
            params["start_date"] = job["start_date"]
            params["end_date"] = job["end_date"]

        match_sql = " AND ".join(where_parts)
        old_job_rows = conn.execute(
            text(f"SELECT job_id FROM bad_debt_compute_jobs WHERE {match_sql}"), params
        ).fetchall()
        old_job_ids = [r[0] for r in old_job_rows]
        if not old_job_ids:
            return {"score_deleted": 0, "customer_deleted": 0, "jobs_affected": 0}

        score_deleted = _delete_result_rows_by_job_ids(
            conn, "bad_debt_score_results", old_job_ids
        )
        customer_deleted = _delete_result_rows_by_job_ids(
            conn, "bad_debt_customer_risk", old_job_ids
        )

        logger.info(
            "Replace-partition cleanup for job %s removed %d score rows and %d customer rows across %d previous jobs",
            job_id,
            score_deleted,
            customer_deleted,
            len(old_job_ids),
        )
        return {
            "score_deleted": score_deleted,
            "customer_deleted": customer_deleted,
            "jobs_affected": len(old_job_ids),
        }


# ── Insert results ────────────────────────────────────────────────────


def _prep_df(
    df: pd.DataFrame,
    job_id: str,
    snapshot_date: str,
    time_range: str,
    model_key: str,
    cols: list[str],
) -> pd.DataFrame:
    rec = df.copy()
    rec["job_id"] = job_id
    rec["snapshot_date"] = snapshot_date
    rec["time_range"] = time_range
    rec["model_key"] = model_key
    available = [c for c in cols if c in rec.columns]
    rec = rec[available]
    rec = rec.replace([np.inf, -np.inf], None)
    rec = rec.where(pd.notnull(rec), None)
    return rec


_SCORE_COLS = [
    "job_id",
    "snapshot_date",
    "time_range",
    "model_key",
    "CUSTOMER_TRX_ID",
    "ACCOUNT_NUMBER",
    "CUSTOMER_NAME",
    "TRX_DATE",
    "DUE_DATE",
    "days_to_due",
    "TRX_AMOUNT",
    "TRX_AMOUNT_GROSS",
    "credit_memo_reduction",
    "prob_bad_debt",
    "risk_level",
    "recommended_action",
    "expected_financial_loss",
]

_CUST_RISK_COLS = [
    "job_id",
    "snapshot_date",
    "time_range",
    "model_key",
    "PARTY_ID",
    "ACCOUNT_NUMBER",
    "CUSTOMER_NAME",
    "cust_score_max",
    "cust_score_mean",
    "cust_score_wavg_amount",
    "invoice_cnt",
    "total_amount",
    "total_amt_paid_pre_due",
    "paid_ratio_pre_due_total",
    "paid_ratio_pre_due_mean",
    "pct_invoices_gap_gt_90_pre_due",
    "party_prior_invoice_cnt",
    "party_prior_gap90_cnt",
    "party_prior_gap90_rate",
    "risk_cust",
]


def insert_score_results(
    job_id: str, snapshot_date: str, time_range: str, model_key: str, df: pd.DataFrame
):
    engine = get_local_engine()
    rec = _prep_df(df, job_id, snapshot_date, time_range, model_key, _SCORE_COLS)
    rec.to_sql(
        "bad_debt_score_results",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )
    logger.info("Inserted %d score results for job %s", len(rec), job_id)


def insert_customer_risk(
    job_id: str, snapshot_date: str, time_range: str, model_key: str, df: pd.DataFrame
):
    engine = get_local_engine()
    rec = _prep_df(df, job_id, snapshot_date, time_range, model_key, _CUST_RISK_COLS)
    rec.to_sql(
        "bad_debt_customer_risk",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )
    logger.info("Inserted %d customer risk records for job %s", len(rec), job_id)


# ── Query helpers ─────────────────────────────────────────────────────

_ALLOWED_SCORE_SORT = {
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

_SCORE_SELECT = (
    "CUSTOMER_TRX_ID,ACCOUNT_NUMBER,CUSTOMER_NAME,"
    "TRX_DATE,DUE_DATE,days_to_due,"
    "TRX_AMOUNT,TRX_AMOUNT_GROSS,credit_memo_reduction,"
    "prob_bad_debt,risk_level,recommended_action,expected_financial_loss"
)


def query_score_results(
    job_id: str,
    page: int = 1,
    page_size: int = 50,
    sort_by: str = "prob_bad_debt",
    sort_order: str = "desc",
    risk_level: str | None = None,
    search: str | None = None,
) -> tuple[list[dict], int]:
    """Return (records, total_count)."""
    if sort_by not in _ALLOWED_SCORE_SORT:
        sort_by = "prob_bad_debt"
    sort_order = (
        "desc" if sort_order.lower() not in ("asc", "desc") else sort_order.lower()
    )

    where, params = ["job_id=:jid"], {"jid": job_id}
    if risk_level in ("HIGH", "MEDIUM", "LOW"):
        where.append("risk_level=:rl")
        params["rl"] = risk_level
    if search:
        where.append("CUSTOMER_NAME LIKE :s")
        params["s"] = f"%{search}%"
    wc = " AND ".join(where)

    engine = get_local_engine()
    with engine.connect() as conn:
        total = (
            conn.execute(
                text(f"SELECT COUNT(*) FROM bad_debt_score_results WHERE {wc}"), params
            ).scalar()
            or 0
        )
        params["lim"] = page_size
        params["off"] = (page - 1) * page_size
        rows = (
            conn.execute(
                text(
                    f"SELECT {_SCORE_SELECT} FROM bad_debt_score_results WHERE {wc} ORDER BY {sort_by} {sort_order} LIMIT :lim OFFSET :off"
                ),
                params,
            )
            .mappings()
            .fetchall()
        )
    return [dict(r) for r in rows], int(total)


def query_top_efl(job_id: str, top_n: int = 50) -> list[dict]:
    engine = get_local_engine()
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    f"SELECT {_SCORE_SELECT} FROM bad_debt_score_results WHERE job_id=:jid AND expected_financial_loss IS NOT NULL ORDER BY expected_financial_loss DESC LIMIT :n"
                ),
                {"jid": job_id, "n": top_n},
            )
            .mappings()
            .fetchall()
        )
    return [dict(r) for r in rows]


def fetch_score_results_by_job(job_id: str) -> pd.DataFrame:
    """Load all invoice-level score rows for one compute job."""
    engine = get_local_engine()
    query = text(
        "SELECT * FROM bad_debt_score_results WHERE job_id=:jid ORDER BY id ASC"
    )
    return pd.read_sql(query, engine, params={"jid": job_id})


_ALLOWED_CR_SORT = {
    "cust_score_max",
    "cust_score_mean",
    "cust_score_wavg_amount",
    "invoice_cnt",
    "total_amount",
    "risk_cust",
    "CUSTOMER_NAME",
}

_CR_SELECT = (
    "PARTY_ID,ACCOUNT_NUMBER,CUSTOMER_NAME,"
    "cust_score_max,cust_score_mean,cust_score_wavg_amount,"
    "invoice_cnt,total_amount,"
    "total_amt_paid_pre_due,paid_ratio_pre_due_total,"
    "paid_ratio_pre_due_mean,pct_invoices_gap_gt_90_pre_due,"
    "party_prior_invoice_cnt,party_prior_gap90_cnt,"
    "party_prior_gap90_rate,risk_cust"
)


def query_customer_risk(
    job_id: str,
    page: int = 1,
    page_size: int = 50,
    sort_by: str = "cust_score_max",
    sort_order: str = "desc",
    risk_cust: str | None = None,
    search: str | None = None,
) -> tuple[list[dict], int]:
    if sort_by not in _ALLOWED_CR_SORT:
        sort_by = "cust_score_max"
    sort_order = (
        "desc" if sort_order.lower() not in ("asc", "desc") else sort_order.lower()
    )

    where, params = ["job_id=:jid"], {"jid": job_id}
    if risk_cust in ("HIGH", "MEDIUM", "LOW"):
        where.append("risk_cust=:rc")
        params["rc"] = risk_cust
    if search:
        where.append("CUSTOMER_NAME LIKE :s")
        params["s"] = f"%{search}%"
    wc = " AND ".join(where)

    engine = get_local_engine()
    with engine.connect() as conn:
        total = (
            conn.execute(
                text(f"SELECT COUNT(*) FROM bad_debt_customer_risk WHERE {wc}"), params
            ).scalar()
            or 0
        )
        params["lim"] = page_size
        params["off"] = (page - 1) * page_size
        rows = (
            conn.execute(
                text(
                    f"SELECT {_CR_SELECT} FROM bad_debt_customer_risk WHERE {wc} ORDER BY {sort_by} {sort_order} LIMIT :lim OFFSET :off"
                ),
                params,
            )
            .mappings()
            .fetchall()
        )
    return [dict(r) for r in rows], int(total)


# ── Maintenance ───────────────────────────────────────────────────────


def cleanup_old_jobs(keep_days: int = 30) -> int:
    engine = get_local_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT job_id FROM bad_debt_compute_jobs WHERE started_at < datetime('now','localtime',:off)"
            ),
            {"off": f"-{keep_days} days"},
        ).fetchall()
        if not rows:
            return 0
        old_ids = [r[0] for r in rows]
        params = {f"jid_{i}": jid for i, jid in enumerate(old_ids)}
        placeholders = ",".join([f":jid_{i}" for i in range(len(old_ids))])
        conn.execute(
            text(
                f"DELETE FROM bad_debt_score_results WHERE job_id IN ({placeholders})"
            ),
            params,
        )
        conn.execute(
            text(
                f"DELETE FROM bad_debt_customer_risk WHERE job_id IN ({placeholders})"
            ),
            params,
        )
        conn.execute(
            text(f"DELETE FROM bad_debt_compute_jobs WHERE job_id IN ({placeholders})"),
            params,
        )
        logger.info("Cleaned up %d old compute jobs", len(old_ids))
        return len(old_ids)
