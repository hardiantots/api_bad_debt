"""DB-backed scoring endpoints.

Score/invoice prediction endpoints read from MySQL publish table.
Customer-risk endpoint remains on local SQLite.
Scoring pipeline is triggered via POST /db/compute (routes_compute.py).
"""

from __future__ import annotations

import math
from datetime import date

from fastapi import APIRouter, BackgroundTasks, Query
from fastapi.responses import JSONResponse, Response

from bad_debt_app.api.config import (
    COMPUTE_AUTO_RECOVER_STALE,
    COMPUTE_MAX_RUNNING_MINUTES,
    COMPUTE_PUBLISH_TARGET_TABLE,
    DEFAULT_MODEL_KEY,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    THRESHOLD_HIGH,
    THRESHOLD_LOW,
)
from bad_debt_app.api.service import model_public_info
from bad_debt_app.data.db import (
    TIME_RANGE_OPTIONS,
    fetch_mysql_scores_df_by_source_job,
    get_latest_mysql_source_job,
    get_data_date_range,
    query_mysql_alerts_by_source_job,
    query_mysql_risk_summary_by_source_job,
    query_mysql_score_results_by_source_job,
    query_mysql_top_efl_by_source_job,
)
from bad_debt_app.data.models import (
    get_latest_job,
    query_customer_risk,
)

router = APIRouter()

_WARM_CACHE_RANGES = {"1w", "2w", "1m", "3m"}


def _no_results_response():
    return JSONResponse(
        status_code=404,
        content={
            "error": "No pre-computed scoring results found for these parameters.",
            "hint": "Run POST /db/compute or click Refresh Scoring to generate and publish scoring results.",
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
):
    """Paginated scoring results from pre-computed data."""
    from bad_debt_app.api.service import resolve_model_key

    snapshot_date = _resolve_snapshot_date(snapshot_date)

    invalid = _validate_custom_window(time_range, start_date, end_date)
    if invalid is not None:
        return invalid

    resolved_key = resolve_model_key(model)
    mysql_job = get_latest_mysql_source_job(
        model_key=resolved_key,
        snapshot_date=snapshot_date,
        time_range=time_range,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )
    if mysql_job is None:
        return _no_results_response()

    job_id = str(mysql_job["source_job_id"])

    records, total = query_mysql_score_results_by_source_job(
        source_job_id=job_id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        risk_level=risk_level,
        search=search,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )
    if total == 0:
        return _no_results_response()

    risk_summary = query_mysql_risk_summary_by_source_job(
        source_job_id=job_id,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )

    top_efl = query_mysql_top_efl_by_source_job(
        source_job_id=job_id,
        top_n=50,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )

    return {
        "mode": "snapshot",
        "model_key": resolved_key,
        "model_flow": None,
        "label_strategy": None,
        "snapshot_date": snapshot_date,
        "effective_snapshot_date": str(
            mysql_job.get("source_snapshot_date") or snapshot_date
        ),
        "time_range": time_range,
        "model_label": resolved_key,
        "last_computed_at": mysql_job.get("last_published_at"),
        "job_id": job_id,
        "total_invoices": int(mysql_job.get("total_invoices", 0) or 0),
        "pagination": _pagination_meta(page, page_size, total),
        "risk_summary": risk_summary,
        "high_risk_count": int(risk_summary.get("HIGH", 0)),
        "preview": records,
        "top_efl_invoices": top_efl,
        "customer_risk_summary": {},
    }


# ── GET /db/customer_risk ─────────────────────────────────────────────


@router.get("/db/customer_risk")
def db_customer_risk(
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
):
    """Paginated customer risk aggregation from pre-computed data."""
    from bad_debt_app.api.service import resolve_model_key

    snapshot_date = _resolve_snapshot_date(snapshot_date)

    invalid = _validate_custom_window(time_range, start_date, end_date)
    if invalid is not None:
        return invalid

    resolved_key = resolve_model_key(model)
    try:
        job = get_latest_job(
            resolved_key,
            snapshot_date,
            time_range,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Customer risk local database unavailable.",
                "detail": str(exc),
            },
        )
    if job is None:
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
):
    """Invoices with prob_bad_debt >= threshold, paginated."""
    from bad_debt_app.api.service import resolve_model_key, safe_threshold

    snapshot_date = _resolve_snapshot_date(snapshot_date)

    threshold = safe_threshold(threshold)
    invalid = _validate_custom_window(time_range, start_date, end_date)
    if invalid is not None:
        return invalid

    resolved_key = resolve_model_key(model)
    mysql_job = get_latest_mysql_source_job(
        model_key=resolved_key,
        snapshot_date=snapshot_date,
        time_range=time_range,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )
    if mysql_job is None:
        return _no_results_response()

    job_id = str(mysql_job["source_job_id"])

    rows, alerts_count = query_mysql_alerts_by_source_job(
        source_job_id=job_id,
        threshold=threshold,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        search=search,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )

    risk_summary = query_mysql_risk_summary_by_source_job(
        source_job_id=job_id,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )

    return {
        "mode": "snapshot",
        "threshold": threshold,
        "model_key": resolved_key,
        "model_flow": None,
        "label_strategy": None,
        "snapshot_date": snapshot_date,
        "effective_snapshot_date": str(
            mysql_job.get("source_snapshot_date") or snapshot_date
        ),
        "time_range": time_range,
        "model_label": resolved_key,
        "last_computed_at": mysql_job.get("last_published_at"),
        "job_id": job_id,
        "total_invoices": int(mysql_job.get("total_invoices", 0) or 0),
        "alerts_count": int(alerts_count),
        "pagination": _pagination_meta(page, page_size, int(alerts_count)),
        "risk_summary": risk_summary,
        "alerts": rows,
        "top_efl_invoices": query_mysql_top_efl_by_source_job(
            source_job_id=job_id,
            top_n=50,
            target_table=COMPUTE_PUBLISH_TARGET_TABLE,
        ),
        "customer_risk_summary": {},
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
    mysql_job = get_latest_mysql_source_job(
        model_key=resolved_key,
        snapshot_date=snapshot_date,
        time_range=time_range,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )
    job = None
    if mysql_job is not None:
        job = {"job_id": str(mysql_job["source_job_id"])}
    if job is None:
        return _no_results_response()

    df = fetch_mysql_scores_df_by_source_job(
        source_job_id=job["job_id"],
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )
    if df.empty:
        return _no_results_response()
    # Drop internal columns
    for col in (
        "id",
        "id",
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
):
    """Early-warning view of pre-computed scoring results, paginated."""
    from bad_debt_app.api.service import resolve_model_key

    snapshot_date = _resolve_snapshot_date(snapshot_date)

    invalid = _validate_custom_window(time_range, start_date, end_date)
    if invalid is not None:
        return invalid

    resolved_key = resolve_model_key(model)
    mysql_job = get_latest_mysql_source_job(
        model_key=resolved_key,
        snapshot_date=snapshot_date,
        time_range=time_range,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )
    if mysql_job is None:
        return _no_results_response()

    job_id = str(mysql_job["source_job_id"])

    # All scores (paginated)
    records, total = query_mysql_score_results_by_source_job(
        source_job_id=job_id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        risk_level=risk_level,
        search=search,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )
    if total == 0:
        return _no_results_response()

    risk_summary = query_mysql_risk_summary_by_source_job(
        source_job_id=job_id,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )

    _, alerts_count = query_mysql_alerts_by_source_job(
        source_job_id=job_id,
        threshold=THRESHOLD_LOW,
        page=1,
        page_size=1,
        sort_by="prob_bad_debt",
        sort_order="desc",
        search=search,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )
    _, high_risk_count = query_mysql_alerts_by_source_job(
        source_job_id=job_id,
        threshold=THRESHOLD_HIGH,
        page=1,
        page_size=1,
        sort_by="prob_bad_debt",
        sort_order="desc",
        search=search,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )

    return {
        "mode": "early_warning",
        "analysis_type": "Early-Warning (Pre-Due Analysis)",
        "analysis_description": "Mengevaluasi risiko bad debt SEBELUM jatuh tempo.",
        "model_key": resolved_key,
        "model_flow": None,
        "label_strategy": None,
        "model_label": resolved_key,
        "snapshot_date": snapshot_date,
        "effective_snapshot_date": str(
            mysql_job.get("source_snapshot_date") or snapshot_date
        ),
        "time_range": time_range,
        "last_computed_at": mysql_job.get("last_published_at"),
        "job_id": job_id,
        "processed_invoices": int(mysql_job.get("total_invoices", 0) or 0),
        "pagination": _pagination_meta(page, page_size, total),
        "risk_summary": risk_summary,
        "alerts_count": int(alerts_count),
        "high_risk_count": int(high_risk_count),
        "all_scores_preview": records,
        "top_efl_invoices": query_mysql_top_efl_by_source_job(
            source_job_id=job_id,
            top_n=50,
            target_table=COMPUTE_PUBLISH_TARGET_TABLE,
        ),
        "customer_risk_summary": {},
    }
