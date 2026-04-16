"""DB-backed scoring endpoints.

Score/invoice prediction endpoints read from MySQL hasil_baddebt publish table.
Customer-risk endpoint reads from local SQLite (written by the compute pipeline).
Scoring pipeline is triggered via POST /db/compute (routes_compute.py).
"""

from __future__ import annotations

import math
from datetime import date

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, Response

from bad_debt_app.api.config import (
    COMPUTE_PUBLISH_TARGET_TABLE,
    DEFAULT_MODEL_KEY,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    MODEL_REGISTRY,
    THRESHOLD_HIGH,
    THRESHOLD_LOW,
)
from bad_debt_app.api.service import model_public_info, resolve_model_key, safe_threshold
from bad_debt_app.data.db import (
    TIME_RANGE_OPTIONS,
    fetch_mysql_scores_df,
    get_data_date_range,
    get_latest_mysql_source_job,
    query_mysql_alerts,
    query_mysql_chart_data,
    query_mysql_risk_summary,
    query_mysql_score_results,
    query_mysql_top_efl,
)
from bad_debt_app.data.models import (
    get_latest_job_with_fallback,
    query_customer_risk,
)

router = APIRouter()

# Columns to always drop before emitting CSV output
_CSV_DROP_COLS = {"id", "_rn"}


# ── Private helpers ────────────────────────────────────────────────────


def _resolve_snapshot_date(snapshot_date: str | None) -> str:
    return snapshot_date or date.today().isoformat()


def _no_results_response() -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": "No pre-computed scoring results found for these parameters.",
            "hint": (
                "Run POST /db/compute or click Refresh Scoring "
                "to generate and publish scoring results."
            ),
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
) -> JSONResponse | None:
    """Return a 422 JSONResponse if custom range is missing its date bounds, else None."""
    if time_range == "custom" and (not start_date or not end_date):
        return JSONResponse(
            status_code=422,
            content={
                "error": "start_date and end_date are required when time_range=custom",
            },
        )
    return None


def _require_mysql_job(
    model_key: str,
    snapshot_date: str,
    time_range: str,
) -> tuple[dict | None, JSONResponse | None]:
    """Resolve the MySQL source job for the given parameters.

    Returns (mysql_job, None) on success, or (None, error_response) when not found.
    """
    mysql_job = get_latest_mysql_source_job(
        model_key=model_key,
        snapshot_date=snapshot_date,
        time_range=time_range,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )
    if mysql_job is None:
        return None, _no_results_response()
    return mysql_job, None


# ── GET /models ────────────────────────────────────────────────────────


@router.get("/models")
def list_models():
    """List available models, time range options, and data date boundaries."""
    dates = get_data_date_range()
    return {
        "models": [model_public_info(k, v) for k, v in MODEL_REGISTRY.items()],
        "time_ranges": [{"key": k, "label": v} for k, v in TIME_RANGE_OPTIONS.items()],
        "min_date": dates.get("min_date"),
        "max_date": dates.get("max_date"),
    }


# ── GET /db/chart_data ─────────────────────────────────────────────────


@router.get("/db/chart_data", summary="Lightweight single-request chart payload")
def get_chart_data(
    model: str = Query(DEFAULT_MODEL_KEY, description="Model key (e.g. 'stacked')"),
    snapshot_date: str | None = Query(None, description="YYYY-MM-DD; defaults to today"),
    time_range: str = Query("all", description="1w | 2w | 1m | 3m | 6m | 1y | all | custom"),
    start_date: str | None = Query(None, description="Start date for custom range"),
    end_date: str | None = Query(None, description="End date for custom range"),
):
    """Return a lightweight list of rows suitable for client-side charting."""
    if time_range not in TIME_RANGE_OPTIONS:
        return JSONResponse(
            status_code=400,
            content={"detail": f"time_range must be one of {list(TIME_RANGE_OPTIONS)}"},
        )

    s_date = _resolve_snapshot_date(snapshot_date)
    resolved_key = resolve_model_key(model)

    source_info = get_latest_mysql_source_job(
        model_key=resolved_key,
        snapshot_date=s_date,
        time_range=time_range,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )
    # Graceful fallback: serve from an "all"-range job when a specific range is absent
    if source_info is None:
        source_info = get_latest_mysql_source_job(
            model_key=resolved_key,
            snapshot_date=s_date,
            time_range="all",
            target_table=COMPUTE_PUBLISH_TARGET_TABLE,
        )
    if source_info is None:
        return {"data": []}

    rows = query_mysql_chart_data(
        job_id=source_info["source_job_id"],
        snapshot_date=s_date,
        time_range=time_range,
        custom_start=start_date,
        custom_end=end_date,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )
    return {"data": rows}


# ── GET /db/score ──────────────────────────────────────────────────────


@router.get("/db/score")
def db_score(
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
    """Paginated invoice scoring results from the MySQL publish table."""
    snapshot_date = _resolve_snapshot_date(snapshot_date)

    invalid = _validate_custom_window(time_range, start_date, end_date)
    if invalid is not None:
        return invalid

    resolved_key = resolve_model_key(model)
    mysql_job, err = _require_mysql_job(resolved_key, snapshot_date, time_range)
    if err is not None:
        return err

    job_id = str(mysql_job["source_job_id"])
    common_kwargs = dict(
        job_id=job_id,
        snapshot_date=snapshot_date,
        time_range=time_range,
        custom_start=start_date,
        custom_end=end_date,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )

    records, total = query_mysql_score_results(
        **common_kwargs,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        risk_level=risk_level,
        search=search,
    )
    if total == 0:
        return _no_results_response()

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
        "risk_summary": query_mysql_risk_summary(**common_kwargs),
        "high_risk_count": int(
            query_mysql_risk_summary(**common_kwargs).get("HIGH", 0)
        ),
        "preview": records,
        "top_efl_invoices": query_mysql_top_efl(**common_kwargs, top_n=50),
        "customer_risk_summary": {},
    }


# ── GET /db/customer_risk ──────────────────────────────────────────────


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
    """Paginated customer risk aggregation from the local SQLite store.

    Falls back to a wider-range job if no exact match exists — e.g. a 6m job
    can serve a 1m or 2w request without requiring a separate scoring run.
    When fallback is used, the response includes ``effective_time_range`` and
    ``fallback_notice`` so the frontend can display a contextual message.
    """
    snapshot_date = _resolve_snapshot_date(snapshot_date)

    invalid = _validate_custom_window(time_range, start_date, end_date)
    if invalid is not None:
        return invalid

    resolved_key = resolve_model_key(model)
    try:
        job = get_latest_job_with_fallback(
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

    # Detect if a wider-range fallback job was used
    fallback_used: bool = job.get("_fallback_used", False)
    effective_time_range: str = job.get("effective_time_range", time_range)
    fallback_notice: str | None = None
    if fallback_used and effective_time_range != time_range:
        from bad_debt_app.data.db import TIME_RANGE_OPTIONS
        effective_label = TIME_RANGE_OPTIONS.get(effective_time_range, effective_time_range)
        fallback_notice = (
            f"Data customer risk ditampilkan dari compute periode {effective_label} "
            f"karena compute untuk periode yang dipilih belum tersedia."
        )

    return {
        "mode": "snapshot",
        "snapshot_date": snapshot_date,
        "time_range": time_range,
        "effective_time_range": effective_time_range,
        "fallback_notice": fallback_notice,
        "model_key": job.get("model_key", model),
        "last_computed_at": job.get("completed_at"),
        "job_id": job_id,
        "pagination": _pagination_meta(page, page_size, total),
        "customer_risk_summary": job.get("customer_risk_summary", {}),
        "customer_risk": records,
    }


# ── GET /db/alerts ─────────────────────────────────────────────────────


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
    snapshot_date = _resolve_snapshot_date(snapshot_date)
    threshold = safe_threshold(threshold)

    invalid = _validate_custom_window(time_range, start_date, end_date)
    if invalid is not None:
        return invalid

    resolved_key = resolve_model_key(model)
    mysql_job, err = _require_mysql_job(resolved_key, snapshot_date, time_range)
    if err is not None:
        return err

    job_id = str(mysql_job["source_job_id"])
    common_kwargs = dict(
        job_id=job_id,
        snapshot_date=snapshot_date,
        time_range=time_range,
        custom_start=start_date,
        custom_end=end_date,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )

    rows, alerts_count = query_mysql_alerts(
        **common_kwargs,
        threshold=threshold,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        search=search,
    )
    risk_summary = query_mysql_risk_summary(**common_kwargs)

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
        "top_efl_invoices": query_mysql_top_efl(**common_kwargs, top_n=50),
        "customer_risk_summary": {},
    }


# ── GET /db/score_csv ──────────────────────────────────────────────────


@router.get("/db/score_csv")
def db_score_csv(
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str | None = Query(None),
    time_range: str = Query("1w"),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
):
    """Export all pre-computed score results for the given period as a CSV download."""
    snapshot_date = _resolve_snapshot_date(snapshot_date)

    invalid = _validate_custom_window(time_range, start_date, end_date)
    if invalid is not None:
        return invalid

    resolved_key = resolve_model_key(model)
    mysql_job, err = _require_mysql_job(resolved_key, snapshot_date, time_range)
    if err is not None:
        return err

    df = fetch_mysql_scores_df(
        job_id=str(mysql_job["source_job_id"]),
        snapshot_date=snapshot_date,
        time_range=time_range,
        custom_start=start_date,
        custom_end=end_date,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )
    if df.empty:
        return _no_results_response()

    # Drop internal/system columns before export
    drop_cols = [c for c in df.columns if c in _CSV_DROP_COLS]
    if drop_cols:
        df = df.drop(columns=drop_cols)

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
    snapshot_date = _resolve_snapshot_date(snapshot_date)

    invalid = _validate_custom_window(time_range, start_date, end_date)
    if invalid is not None:
        return invalid

    resolved_key = resolve_model_key(model)
    mysql_job, err = _require_mysql_job(resolved_key, snapshot_date, time_range)
    if err is not None:
        return err

    job_id = str(mysql_job["source_job_id"])
    common_kwargs = dict(
        job_id=job_id,
        snapshot_date=snapshot_date,
        time_range=time_range,
        custom_start=start_date,
        custom_end=end_date,
        target_table=COMPUTE_PUBLISH_TARGET_TABLE,
    )

    records, total = query_mysql_score_results(
        **common_kwargs,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        risk_level=risk_level,
        search=search,
    )
    if total == 0:
        return _no_results_response()

    risk_summary = query_mysql_risk_summary(**common_kwargs)

    _, alerts_count = query_mysql_alerts(
        **common_kwargs,
        threshold=THRESHOLD_LOW,
        page=1,
        page_size=1,
        sort_by="prob_bad_debt",
        sort_order="desc",
        search=search,
    )
    _, high_risk_count = query_mysql_alerts(
        **common_kwargs,
        threshold=THRESHOLD_HIGH,
        page=1,
        page_size=1,
        sort_by="prob_bad_debt",
        sort_order="desc",
        search=search,
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
        "top_efl_invoices": query_mysql_top_efl(**common_kwargs, top_n=50),
        "customer_risk_summary": {},
    }
