"""DB-backed scoring endpoints — read pre-computed results from local SQLite.

GET endpoints are now read-only and paginated.
Scoring pipeline is triggered via POST /db/compute (routes_compute.py).
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Query
from fastapi.responses import JSONResponse, Response

from bad_debt_app.api.config import (
    COMPUTE_AUTO_RECOVER_STALE,
    COMPUTE_MAX_RUNNING_MINUTES,
    DEFAULT_MODEL_KEY,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    THRESHOLD_HIGH,
    THRESHOLD_LOW,
)
from bad_debt_app.api.service import model_public_info
from bad_debt_app.data.db import TIME_RANGE_OPTIONS, get_data_date_range
from bad_debt_app.data.models import (
    get_latest_job,
    get_local_engine,
    query_customer_risk,
    query_score_results,
    query_top_efl,
)

router = APIRouter()

_WARM_CACHE_RANGES = {"1w", "2w", "1m", "3m"}


def _no_results_response():
    return JSONResponse(
        status_code=404,
        content={
            "error": "No pre-computed scoring results found for these parameters.",
            "hint": "Run POST /db/compute first to generate scoring results.",
        },
    )


def _pagination_meta(page: int, page_size: int, total: int) -> dict:
    return {
        "page": page,
        "page_size": page_size,
        "total_records": total,
        "total_pages": math.ceil(total / page_size) if page_size > 0 else 0,
    }


def _validate_custom_window(
    time_range: str, start_date: str | None, end_date: str | None
):
    if time_range == "custom" and (not start_date or not end_date):
        return JSONResponse(
            status_code=422,
            content={
                "error": "start_date and end_date are required when time_range=custom",
            },
        )
    return None


def _resolve_snapshot_date(snapshot_date: str | None) -> str:
    return snapshot_date or date.today().isoformat()


def _bootstrap_compute_job(
    background_tasks: BackgroundTasks,
    model: str,
    snapshot_date: str,
    time_range: str,
    start_date: str | None,
    end_date: str | None,
):
    """Start compute in background and return a 202 payload for bootstrap clients."""
    from bad_debt_app.api.routes_compute import _run_compute
    from bad_debt_app.api.service import resolve_model, resolve_model_key
    from bad_debt_app.data.models import (
        create_job,
        generate_job_id,
        get_all_jobs,
        has_running_job,
        recover_stale_running_jobs,
    )

    if COMPUTE_AUTO_RECOVER_STALE:
        recover_stale_running_jobs(
            COMPUTE_MAX_RUNNING_MINUTES,
            reason="Auto-recovered stale running job before bootstrap compute",
        )

    if has_running_job():
        running_job_id = None
        for job in get_all_jobs(limit=20):
            if job.get("status") == "running":
                running_job_id = job.get("job_id")
                break
        return JSONResponse(
            status_code=202,
            content={
                "job_id": running_job_id,
                "status": "running",
                "message": "Compute job is already running. Waiting for completion.",
                "status_url": (
                    f"/db/compute/status/{running_job_id}" if running_job_id else None
                ),
            },
        )

    resolved_key = resolve_model_key(model)
    m_info = resolve_model(resolved_key)
    job_id = generate_job_id()
    compute_time_range = "3m" if time_range in _WARM_CACHE_RANGES else time_range

    create_job(
        job_id,
        model_key=resolved_key,
        model_label=m_info.get("label", resolved_key),
        model_flow=m_info.get("training_flow"),
        label_strategy=m_info.get("label_strategy"),
        snapshot_date=snapshot_date,
        time_range=compute_time_range,
        start_date=start_date,
        end_date=end_date,
    )

    background_tasks.add_task(
        _run_compute,
        job_id=job_id,
        model_key=resolved_key,
        snapshot_date=snapshot_date,
        time_range=compute_time_range,
        start_date=start_date,
        end_date=end_date,
    )

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "running",
            "message": (
                "No pre-computed data found. 3-month warm cache compute started automatically."
                if compute_time_range == "3m" and time_range in _WARM_CACHE_RANGES
                else "No pre-computed data found. Compute started automatically."
            ),
            "status_url": f"/db/compute/status/{job_id}",
        },
    )


# ── /models ───────────────────────────────────────────────────────────


@router.get("/models")
def list_models():
    from bad_debt_app.api.config import MODEL_REGISTRY

    dates = get_data_date_range()
    return {
        "models": [model_public_info(k, v) for k, v in MODEL_REGISTRY.items()],
        "time_ranges": [{"key": k, "label": v} for k, v in TIME_RANGE_OPTIONS.items()],
        "min_date": dates.get("min_date"),
        "max_date": dates.get("max_date"),
    }


# ── GET /db/score ─────────────────────────────────────────────────────


@router.get("/db/score")
def db_score(
    background_tasks: BackgroundTasks,
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str | None = Query(None),
    time_range: str = Query("1w"),
    # Pagination
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    # Sorting
    sort_by: str = Query("prob_bad_debt"),
    sort_order: str = Query("desc"),
    # Filtering
    risk_level: str | None = Query(None),
    search: str | None = Query(None),
    # Needed to select the exact compute job when time_range=custom.
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    auto_compute_if_missing: bool = Query(False),
):
    """Paginated scoring results from pre-computed data."""
    from bad_debt_app.api.service import resolve_model_key

    snapshot_date = _resolve_snapshot_date(snapshot_date)

    invalid = _validate_custom_window(time_range, start_date, end_date)
    if invalid is not None:
        return invalid

    resolved_key = resolve_model_key(model)
    job = get_latest_job(
        resolved_key,
        snapshot_date,
        time_range,
        start_date=start_date,
        end_date=end_date,
    )
    if job is None:
        if auto_compute_if_missing:
            return _bootstrap_compute_job(
                background_tasks,
                model=resolved_key,
                snapshot_date=snapshot_date,
                time_range=time_range,
                start_date=start_date,
                end_date=end_date,
            )
        return _no_results_response()

    job_id = job["job_id"]

    records, total = query_score_results(
        job_id=job_id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        risk_level=risk_level,
        search=search,
    )

    top_efl = query_top_efl(job_id, 50)

    return {
        "mode": "snapshot",
        "model_key": job.get("model_key", model),
        "model_flow": job.get("model_flow"),
        "label_strategy": job.get("label_strategy"),
        "snapshot_date": snapshot_date,
        "time_range": time_range,
        "model_label": job.get("model_label", ""),
        "last_computed_at": job.get("completed_at"),
        "job_id": job_id,
        "total_invoices": job.get("total_invoices", 0),
        "pagination": _pagination_meta(page, page_size, total),
        "risk_summary": job.get("risk_summary", {}),
        "high_risk_count": int(job.get("risk_summary", {}).get("HIGH", 0)),
        "preview": records,
        "top_efl_invoices": top_efl,
        "customer_risk_summary": job.get("customer_risk_summary", {}),
    }


# ── GET /db/customer_risk ─────────────────────────────────────────────


@router.get("/db/customer_risk")
def db_customer_risk(
    background_tasks: BackgroundTasks,
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str | None = Query(None),
    time_range: str = Query("1w"),
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    sort_by: str = Query("cust_score_max"),
    sort_order: str = Query("desc"),
    risk_cust: str | None = Query(None),
    search: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    auto_compute_if_missing: bool = Query(False),
):
    """Paginated customer risk aggregation from pre-computed data."""
    from bad_debt_app.api.service import resolve_model_key

    snapshot_date = _resolve_snapshot_date(snapshot_date)

    invalid = _validate_custom_window(time_range, start_date, end_date)
    if invalid is not None:
        return invalid

    resolved_key = resolve_model_key(model)
    job = get_latest_job(
        resolved_key,
        snapshot_date,
        time_range,
        start_date=start_date,
        end_date=end_date,
    )
    if job is None:
        if auto_compute_if_missing:
            return _bootstrap_compute_job(
                background_tasks,
                model=resolved_key,
                snapshot_date=snapshot_date,
                time_range=time_range,
                start_date=start_date,
                end_date=end_date,
            )
        return _no_results_response()

    job_id = job["job_id"]
    records, total = query_customer_risk(
        job_id=job_id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        risk_cust=risk_cust,
        search=search,
    )

    return {
        "mode": "snapshot",
        "snapshot_date": snapshot_date,
        "time_range": time_range,
        "model_key": job.get("model_key", model),
        "last_computed_at": job.get("completed_at"),
        "job_id": job_id,
        "pagination": _pagination_meta(page, page_size, total),
        "customer_risk_summary": job.get("customer_risk_summary", {}),
        "customer_risk": records,
    }


# ── GET /db/alerts ────────────────────────────────────────────────────


@router.get("/db/alerts")
def db_alerts(
    background_tasks: BackgroundTasks,
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str | None = Query(None),
    time_range: str = Query("1w"),
    threshold: float = Query(0.3),
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    sort_by: str = Query("prob_bad_debt"),
    sort_order: str = Query("desc"),
    search: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    auto_compute_if_missing: bool = Query(False),
):
    """Invoices with prob_bad_debt >= threshold, paginated."""
    from bad_debt_app.api.service import resolve_model_key, safe_threshold

    snapshot_date = _resolve_snapshot_date(snapshot_date)

    threshold = safe_threshold(threshold)
    invalid = _validate_custom_window(time_range, start_date, end_date)
    if invalid is not None:
        return invalid

    resolved_key = resolve_model_key(model)
    job = get_latest_job(
        resolved_key,
        snapshot_date,
        time_range,
        start_date=start_date,
        end_date=end_date,
    )
    if job is None:
        if auto_compute_if_missing:
            return _bootstrap_compute_job(
                background_tasks,
                model=resolved_key,
                snapshot_date=snapshot_date,
                time_range=time_range,
                start_date=start_date,
                end_date=end_date,
            )
        return _no_results_response()

    job_id = job["job_id"]

    # Use custom query with threshold filter
    from sqlalchemy import text

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

    where_parts = ["job_id=:jid", "prob_bad_debt>=:thr"]
    params: dict = {"jid": job_id, "thr": threshold}
    if search:
        where_parts.append("CUSTOMER_NAME LIKE :s")
        params["s"] = f"%{search}%"
    wc = " AND ".join(where_parts)

    engine = get_local_engine()
    with engine.connect() as conn:
        alerts_count = (
            conn.execute(
                text(f"SELECT COUNT(*) FROM bad_debt_score_results WHERE {wc}"),
                params,
            ).scalar()
            or 0
        )

        params["lim"] = page_size
        params["off"] = (page - 1) * page_size
        cols = (
            "CUSTOMER_TRX_ID,ACCOUNT_NUMBER,CUSTOMER_NAME,"
            "TRX_DATE,DUE_DATE,days_to_due,"
            "TRX_AMOUNT,TRX_AMOUNT_GROSS,credit_memo_reduction,"
            "prob_bad_debt,risk_level,recommended_action,expected_financial_loss"
        )
        rows = (
            conn.execute(
                text(
                    f"SELECT {cols} FROM bad_debt_score_results WHERE {wc} ORDER BY {sort_by} {sort_order} LIMIT :lim OFFSET :off"
                ),
                params,
            )
            .mappings()
            .fetchall()
        )

    return {
        "mode": "snapshot",
        "threshold": threshold,
        "model_key": job.get("model_key", model),
        "model_flow": job.get("model_flow"),
        "label_strategy": job.get("label_strategy"),
        "snapshot_date": snapshot_date,
        "time_range": time_range,
        "model_label": job.get("model_label", ""),
        "last_computed_at": job.get("completed_at"),
        "job_id": job_id,
        "total_invoices": job.get("total_invoices", 0),
        "alerts_count": int(alerts_count),
        "pagination": _pagination_meta(page, page_size, int(alerts_count)),
        "risk_summary": job.get("risk_summary", {}),
        "alerts": [dict(r) for r in rows],
        "top_efl_invoices": query_top_efl(job_id, 50),
        "customer_risk_summary": job.get("customer_risk_summary", {}),
    }


# ── GET /db/score_csv ─────────────────────────────────────────────────


@router.get("/db/score_csv")
def db_score_csv(
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str | None = Query(None),
    time_range: str = Query("1w"),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
):
    """Export all pre-computed score results as CSV download."""
    from bad_debt_app.api.service import resolve_model_key

    snapshot_date = _resolve_snapshot_date(snapshot_date)

    invalid = _validate_custom_window(time_range, start_date, end_date)
    if invalid is not None:
        return invalid

    resolved_key = resolve_model_key(model)
    job = get_latest_job(
        resolved_key,
        snapshot_date,
        time_range,
        start_date=start_date,
        end_date=end_date,
    )
    if job is None:
        return _no_results_response()

    engine = get_local_engine()
    from sqlalchemy import text

    df = pd.read_sql(
        text("SELECT * FROM bad_debt_score_results WHERE job_id=:jid"),
        engine,
        params={"jid": job["job_id"]},
    )
    # Drop internal columns
    for col in (
        "id",
        "job_id",
        "snapshot_date",
        "time_range",
        "model_key",
        "created_at",
    ):
        if col in df.columns:
            df = df.drop(columns=[col])

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    filename = f"bad_debt_{time_range}_{snapshot_date}.csv".replace(":", "-")
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── GET /db/early_warning/receipt_trigger ─────────────────────────────


@router.get("/db/early_warning/receipt_trigger")
def db_receipt_trigger(
    background_tasks: BackgroundTasks,
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str | None = Query(None),
    time_range: str = Query("1w"),
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    sort_by: str = Query("prob_bad_debt"),
    sort_order: str = Query("desc"),
    risk_level: str | None = Query(None),
    search: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    auto_compute_if_missing: bool = Query(False),
):
    """Early-warning view of pre-computed scoring results, paginated."""
    from bad_debt_app.api.service import resolve_model_key

    snapshot_date = _resolve_snapshot_date(snapshot_date)

    invalid = _validate_custom_window(time_range, start_date, end_date)
    if invalid is not None:
        return invalid

    resolved_key = resolve_model_key(model)
    job = get_latest_job(
        resolved_key,
        snapshot_date,
        time_range,
        start_date=start_date,
        end_date=end_date,
    )
    if job is None:
        if auto_compute_if_missing:
            return _bootstrap_compute_job(
                background_tasks,
                model=resolved_key,
                snapshot_date=snapshot_date,
                time_range=time_range,
                start_date=start_date,
                end_date=end_date,
            )
        return _no_results_response()

    job_id = job["job_id"]

    # All scores (paginated)
    records, total = query_score_results(
        job_id=job_id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        risk_level=risk_level,
        search=search,
    )

    # Alerts count (above THRESHOLD_LOW)
    from sqlalchemy import text

    engine = get_local_engine()
    with engine.connect() as conn:
        alerts_count = (
            conn.execute(
                text(
                    "SELECT COUNT(*) FROM bad_debt_score_results WHERE job_id=:jid AND prob_bad_debt>=:thr"
                ),
                {"jid": job_id, "thr": THRESHOLD_LOW},
            ).scalar()
            or 0
        )
        high_risk_count = (
            conn.execute(
                text(
                    "SELECT COUNT(*) FROM bad_debt_score_results WHERE job_id=:jid AND prob_bad_debt>=:thr"
                ),
                {"jid": job_id, "thr": THRESHOLD_HIGH},
            ).scalar()
            or 0
        )

    return {
        "mode": "early_warning",
        "analysis_type": "Early-Warning (Pre-Due Analysis)",
        "analysis_description": "Mengevaluasi risiko bad debt SEBELUM jatuh tempo.",
        "model_key": job.get("model_key", model),
        "model_flow": job.get("model_flow"),
        "label_strategy": job.get("label_strategy"),
        "model_label": job.get("model_label", ""),
        "snapshot_date": snapshot_date,
        "time_range": time_range,
        "last_computed_at": job.get("completed_at"),
        "job_id": job_id,
        "processed_invoices": job.get("total_invoices", 0),
        "pagination": _pagination_meta(page, page_size, total),
        "risk_summary": job.get("risk_summary", {}),
        "alerts_count": int(alerts_count),
        "high_risk_count": int(high_risk_count),
        "all_scores_preview": records,
        "top_efl_invoices": query_top_efl(job_id, 50),
        "customer_risk_summary": job.get("customer_risk_summary", {}),
    }
