"""POST /db/compute endpoint — triggers scoring and stores results locally."""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from datetime import timedelta

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from bad_debt_app.api.config import (
    COMPUTE_AUTO_PUBLISH_SCORE_TO_MYSQL,
    COMPUTE_PUBLISH_REPLACE_PARTITION,
    COMPUTE_AUTO_RECOVER_STALE,
    COMPUTE_KEEP_DAYS,
    COMPUTE_MAX_RUNNING_MINUTES,
    COMPUTE_PUBLISH_TARGET_TABLE,
    DEFAULT_MODEL_KEY,
)
from bad_debt_app.api.service import (
    apply_customer_exclusion,
    build_customer_risk,
    build_customer_risk_summary,
    fetch_raw_from_db,
    filter_excluded_customers,
    resolve_model,
    resolve_model_key,
    score_snapshot,
)
from bad_debt_app.data.models import (
    cleanup_old_jobs,
    complete_job,
    create_job,
    fail_job,
    fetch_score_results_by_job,
    generate_job_id,
    get_all_jobs,
    get_latest_job,
    get_job,
    has_running_job,
    insert_customer_risk,
    insert_score_results,
    recover_stale_running_jobs,
    replace_partition_results_for_job,
)

logger = logging.getLogger("bad_debt_api")
router = APIRouter()

_SHORT_RANGE_DAYS = {
    "1m": 30,
    "2w": 14,
    "1w": 7,
}


def _resolve_snapshot_date(snapshot_date: str | None) -> str:
    return snapshot_date or date.today().isoformat()


def _auto_publish_score_best_effort(
    *,
    job_id: str,
    model_key: str,
    snapshot_date: str,
    time_range: str,
    df_score: pd.DataFrame,
):
    """Publish invoice-level score rows to MySQL target table without breaking compute flow."""
    if not COMPUTE_AUTO_PUBLISH_SCORE_TO_MYSQL:
        return

    from bad_debt_app.data.db import publish_score_to_hasil_baddebt

    try:
        summary = publish_score_to_hasil_baddebt(
            df_score=df_score,
            model_key=model_key,
            snapshot_date=snapshot_date,
            time_range=time_range,
            source_job_id=job_id,
            target_table=COMPUTE_PUBLISH_TARGET_TABLE,
            replace_partition=COMPUTE_PUBLISH_REPLACE_PARTITION,
        )
        logger.info(
            "Auto-published score for job %s to %s (%d rows)",
            job_id,
            summary.get("table", COMPUTE_PUBLISH_TARGET_TABLE),
            int(summary.get("rows_inserted", 0)),
        )
    except Exception:
        logger.exception(
            "Auto-publish score failed for job %s (table=%s)",
            job_id,
            COMPUTE_PUBLISH_TARGET_TABLE,
        )


def _store_partition_results(
    *,
    job_id: str,
    snapshot_date: str,
    time_range: str,
    model_key: str,
    invoice_df: pd.DataFrame,
    customer_risk_df: pd.DataFrame,
) -> tuple[dict, dict]:
    """Persist invoice/customer results and return both summaries."""
    insert_score_results(
        job_id=job_id,
        snapshot_date=snapshot_date,
        time_range=time_range,
        model_key=model_key,
        df=invoice_df,
    )
    insert_customer_risk(
        job_id=job_id,
        snapshot_date=snapshot_date,
        time_range=time_range,
        model_key=model_key,
        df=customer_risk_df,
    )
    return (
        invoice_df["risk_level"].value_counts().to_dict(),
        build_customer_risk_summary(customer_risk_df),
    )


def _replace_old_partition_rows(job_id: str):
    """Best-effort cleanup of older rows in the same partition."""
    try:
        replace_partition_results_for_job(job_id)
    except Exception:
        logger.exception(
            "Replace-partition cleanup failed for job %s; new results are kept",
            job_id,
        )


def _materialize_short_ranges_from_three_months(
    *,
    base_model_key: str,
    snapshot_date: str,
    out: pd.DataFrame,
    df_feat: pd.DataFrame,
    proba,
    model_label: str,
    model_flow: str | None,
    label_strategy: str | None,
):
    """Create pre-computed 1m/2w/1w caches from a 3m compute output.

    This avoids re-running heavy modeling when users switch filters inside 3 months.
    """
    snap = pd.to_datetime(snapshot_date, errors="coerce")
    if pd.isna(snap):
        return

    out_aligned = out.reset_index(drop=True)
    feat_aligned = df_feat.reset_index(drop=True)
    proba_arr = np.asarray(proba)
    if len(proba_arr) != len(out_aligned):
        logger.warning("Skip short-range materialization: proba length mismatch")
        return

    trx_dates = pd.to_datetime(out_aligned.get("TRX_DATE"), errors="coerce")
    if trx_dates.isna().all():
        logger.warning("Skip short-range materialization: TRX_DATE missing/unparseable")
        return

    for tr, days in _SHORT_RANGE_DAYS.items():
        start = snap - timedelta(days=days)
        mask = (trx_dates >= start) & (trx_dates <= snap)
        if int(mask.sum()) == 0:
            continue

        out_sub = out_aligned.loc[mask].copy()
        feat_sub = feat_aligned.loc[mask].copy()
        proba_sub = proba_arr[mask.to_numpy()]

        job_id = generate_job_id()
        create_job(
            job_id,
            model_key=base_model_key,
            model_label=model_label,
            model_flow=model_flow,
            label_strategy=label_strategy,
            snapshot_date=snapshot_date,
            time_range=tr,
        )

        customer_risk_sub = build_customer_risk(feat_sub, proba_sub)
        customer_risk_sub = filter_excluded_customers(
            customer_risk_sub, name_col="CUSTOMER_NAME"
        )

        risk_summary, customer_risk_summary = _store_partition_results(
            job_id=job_id,
            snapshot_date=snapshot_date,
            time_range=tr,
            model_key=base_model_key,
            invoice_df=out_sub,
            customer_risk_df=customer_risk_sub,
        )

        complete_job(
            job_id,
            total_invoices=int(out_sub.shape[0]),
            total_customers=int(customer_risk_sub.shape[0]),
            risk_summary=risk_summary,
            customer_risk_summary=customer_risk_summary,
            duration_sec=0.0,
        )
        _replace_old_partition_rows(job_id)
        logger.info(
            "Materialized short-range cache %s: %d invoices, %d customers",
            tr,
            out_sub.shape[0],
            customer_risk_sub.shape[0],
        )


# ── Background scoring task ──────────────────────────────────────────


def _run_compute(
    job_id: str,
    model_key: str,
    snapshot_date: str,
    time_range: str,
    start_date: str | None,
    end_date: str | None,
):
    start = time.time()
    try:
        raw = fetch_raw_from_db(
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            snapshot_date=snapshot_date,
        )
        if isinstance(raw, JSONResponse):
            error_detail = "DB fetch failed"
            try:
                payload = json.loads(raw.body.decode("utf-8"))
                error_detail = payload.get("error", error_detail)
            except Exception:
                pass
            fail_job(job_id, error_detail, time.time() - start)
            return
        if raw.invoice.empty:
            fail_job(
                job_id, "No invoices found for the given filters", time.time() - start
            )
            return

        scored = score_snapshot(
            raw=raw, snapshot_date=snapshot_date, model_key=model_key
        )
        if isinstance(scored, JSONResponse):
            fail_job(job_id, "Scoring pipeline failed", time.time() - start)
            return

        out, df_feat, proba, m_info = scored
        out, df_feat, proba = apply_customer_exclusion(out, df_feat, proba)

        customer_risk = build_customer_risk(df_feat, proba)
        customer_risk = filter_excluded_customers(
            customer_risk, name_col="CUSTOMER_NAME"
        )

        risk_summary, cr_summary = _store_partition_results(
            job_id=job_id,
            snapshot_date=snapshot_date,
            time_range=time_range,
            model_key=model_key,
            invoice_df=out,
            customer_risk_df=customer_risk,
        )

        complete_job(
            job_id,
            total_invoices=int(out.shape[0]),
            total_customers=int(customer_risk.shape[0]),
            risk_summary=risk_summary,
            customer_risk_summary=cr_summary,
            duration_sec=time.time() - start,
        )
        _replace_old_partition_rows(job_id)
        _auto_publish_score_best_effort(
            job_id=job_id,
            model_key=model_key,
            snapshot_date=snapshot_date,
            time_range=time_range,
            df_score=out,
        )

        if time_range == "3m":
            _materialize_short_ranges_from_three_months(
                base_model_key=model_key,
                snapshot_date=snapshot_date,
                out=out,
                df_feat=df_feat,
                proba=proba,
                model_label=m_info.get("label", model_key),
                model_flow=m_info.get("training_flow"),
                label_strategy=m_info.get("label_strategy"),
            )

        logger.info(
            "Compute job %s completed: %d invoices, %d customers in %.1fs",
            job_id,
            out.shape[0],
            customer_risk.shape[0],
            time.time() - start,
        )

        try:
            cleanup_old_jobs(COMPUTE_KEEP_DAYS)
        except Exception:
            logger.exception("Cleanup old jobs failed after compute %s", job_id)

    except Exception as exc:
        logger.exception("Compute job %s failed", job_id)
        fail_job(job_id, str(exc), time.time() - start)


# ── Endpoints ─────────────────────────────────────────────────────────


@router.post("/db/compute")
async def compute(
    background_tasks: BackgroundTasks,
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str | None = Query(None),
    time_range: str = Query("1w"),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
):
    """Trigger scoring pipeline. Runs in background; results stored to local DB."""
    snapshot_date = _resolve_snapshot_date(snapshot_date)

    if time_range == "custom" and (not start_date or not end_date):
        return JSONResponse(
            status_code=422,
            content={
                "error": "start_date and end_date are required when time_range=custom"
            },
        )

    if COMPUTE_AUTO_RECOVER_STALE:
        recover_stale_running_jobs(
            COMPUTE_MAX_RUNNING_MINUTES,
            reason="Auto-recovered stale running job before manual compute trigger",
        )

    if has_running_job():
        return JSONResponse(
            status_code=409,
            content={
                "error": "A compute job is already running. Please wait for it to complete."
            },
        )

    resolved_key = resolve_model_key(model)
    m_info = resolve_model(resolved_key)

    job_id = generate_job_id()
    create_job(
        job_id,
        model_key=resolved_key,
        model_label=m_info.get("label", resolved_key),
        model_flow=m_info.get("training_flow"),
        label_strategy=m_info.get("label_strategy"),
        snapshot_date=snapshot_date,
        time_range=time_range,
        start_date=start_date,
        end_date=end_date,
    )

    background_tasks.add_task(
        _run_compute,
        job_id=job_id,
        model_key=resolved_key,
        snapshot_date=snapshot_date,
        time_range=time_range,
        start_date=start_date,
        end_date=end_date,
    )

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "running",
            "message": "Scoring computation started in background.",
            "status_url": f"/db/compute/status/{job_id}",
        },
    )


@router.get("/db/compute/status")
def compute_latest_status():
    """Get the most recent compute job status."""
    jobs = get_all_jobs(limit=1)
    if not jobs:
        return {"message": "No compute jobs found.", "jobs": []}
    return jobs[0]


@router.get("/db/compute/status/{job_id}")
def compute_status(job_id: str):
    """Get status of a specific compute job."""
    job = get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Job not found."})
    job.pop("risk_summary_json", None)
    job.pop("customer_risk_summary_json", None)
    return job


@router.get("/db/compute/history")
def compute_history(limit: int = Query(20, ge=1, le=100)):
    """List recent compute jobs."""
    return {"jobs": get_all_jobs(limit=limit)}


@router.post("/db/compute/publish")
def publish_compute_result(
    job_id: str | None = Query(None),
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str | None = Query(None),
    time_range: str = Query("1w"),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    table_name: str = Query("hasil_baddebt"),
    replace_partition: bool = Query(True),
):
    """Publish completed compute score rows from local SQLite to MySQL table."""
    from bad_debt_app.data.db import publish_score_to_hasil_baddebt

    snapshot_date = _resolve_snapshot_date(snapshot_date)

    target_job = None
    if job_id:
        target_job = get_job(job_id)
        if target_job is None:
            return JSONResponse(status_code=404, content={"error": "Job not found."})
    else:
        resolved_key = resolve_model_key(model)
        target_job = get_latest_job(
            resolved_key,
            snapshot_date,
            time_range,
            start_date=start_date,
            end_date=end_date,
        )
        if target_job is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "No completed compute result found for given parameters."
                },
            )

    if target_job.get("status") != "completed":
        return JSONResponse(
            status_code=409,
            content={
                "error": "Selected compute job is not completed yet.",
                "job_id": target_job.get("job_id"),
                "status": target_job.get("status"),
            },
        )

    df_score = fetch_score_results_by_job(target_job["job_id"])

    try:
        publish_summary = publish_score_to_hasil_baddebt(
            df_score=df_score,
            model_key=target_job.get("model_key", model),
            snapshot_date=target_job.get("snapshot_date", snapshot_date),
            time_range=target_job.get("time_range", time_range),
            source_job_id=target_job["job_id"],
            target_table=table_name,
            replace_partition=replace_partition,
        )
    except Exception as exc:
        logger.exception("Failed to publish compute job %s", target_job.get("job_id"))
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to publish compute result: {exc}"},
        )

    return {
        "message": "Compute result published to MySQL.",
        "job_id": target_job.get("job_id"),
        "model_key": target_job.get("model_key"),
        "snapshot_date": target_job.get("snapshot_date"),
        "time_range": target_job.get("time_range"),
        "publish": publish_summary,
    }
