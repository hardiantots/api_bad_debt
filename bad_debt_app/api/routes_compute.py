"""POST /db/compute endpoint — triggers scoring and publishes score results to MySQL."""

from __future__ import annotations

import json
import logging
import time
from datetime import date

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from bad_debt_app.api.config import (
    COMPUTE_AUTO_PUBLISH_SCORE_TO_MYSQL,
    COMPUTE_AUTO_RECOVER_STALE,
    COMPUTE_DEFAULT_TIME_RANGE,
    COMPUTE_KEEP_DAYS,
    COMPUTE_MAX_RUNNING_MINUTES,
    COMPUTE_MYSQL_KEEP_DAYS,
    COMPUTE_PUBLISH_REPLACE_PARTITION,
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
    generate_job_id,
    get_all_jobs,
    get_job,
    get_latest_job,
    has_running_job,
    insert_customer_risk,
    recover_stale_running_jobs,
    replace_partition_results_for_job,
)

logger = logging.getLogger("bad_debt_api")
router = APIRouter()


# ── Private helpers ────────────────────────────────────────────────────


def _resolve_snapshot_date(snapshot_date: str | None) -> str:
    return snapshot_date or date.today().isoformat()


def _auto_publish_score_best_effort(
    *,
    job_id: str,
    model_key: str,
    snapshot_date: str,
    time_range: str,
    df_score: pd.DataFrame,
) -> None:
    """Publish invoice-level score rows to MySQL target table.

    Best-effort: any failure is logged but does not break the compute flow.
    No-op when COMPUTE_AUTO_PUBLISH_SCORE_TO_MYSQL is disabled.
    """
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


def _cleanup_mysql_best_effort(job_id: str) -> None:
    """Remove MySQL hasil_baddebt rows older than COMPUTE_MYSQL_KEEP_DAYS.

    Best-effort: any failure (e.g. missing DELETE privilege) is logged only.
    No-op when COMPUTE_AUTO_PUBLISH_SCORE_TO_MYSQL is disabled.
    """
    if not COMPUTE_AUTO_PUBLISH_SCORE_TO_MYSQL:
        return

    from bad_debt_app.data.db import cleanup_mysql_old_scoring_results

    try:
        cleanup_mysql_old_scoring_results(
            target_table=COMPUTE_PUBLISH_TARGET_TABLE,
            keep_days=COMPUTE_MYSQL_KEEP_DAYS,
        )
    except Exception:
        logger.exception(
            "MySQL cleanup failed after compute %s (non-fatal)", job_id
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
    """Persist customer-risk rows to local SQLite and return compute summaries."""
    insert_customer_risk(
        job_id=job_id,
        snapshot_date=snapshot_date,
        time_range=time_range,
        model_key=model_key,
        df=customer_risk_df,
    )
    risk_summary = invoice_df["risk_level"].value_counts().to_dict()
    cr_summary = build_customer_risk_summary(customer_risk_df)
    return risk_summary, cr_summary


def _replace_old_partition_rows(job_id: str) -> None:
    """Best-effort cleanup of older SQLite rows in the same partition."""
    try:
        replace_partition_results_for_job(job_id)
    except Exception:
        logger.exception(
            "Replace-partition cleanup failed for job %s; new results are kept",
            job_id,
        )


def _run_compute(
    job_id: str,
    model_key: str,
    snapshot_date: str,
    time_range: str,
    start_date: str | None,
    end_date: str | None,
) -> None:
    """Core scoring pipeline executed in a background thread.

    Steps:
      1. Fetch raw invoice/receipt data from MySQL.
      2. Run model scoring pipeline.
      3. Build and persist customer-risk aggregates to local SQLite.
      4. Mark job as completed.
      5. Replace old partition rows (SQLite).
      6. Publish invoice scores to MySQL hasil_baddebt.
      7. Cleanup old MySQL rows (> COMPUTE_MYSQL_KEEP_DAYS).
      8. Cleanup old SQLite jobs (> COMPUTE_KEEP_DAYS).
    """
    start = time.time()
    try:
        # ── Step 1: fetch raw data ──────────────────────────────────────
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
                job_id,
                "No invoices found for the given filters",
                time.time() - start,
            )
            return

        # ── Step 2: score ───────────────────────────────────────────────
        scored = score_snapshot(
            raw=raw, snapshot_date=snapshot_date, model_key=model_key
        )
        if isinstance(scored, JSONResponse):
            fail_job(job_id, "Scoring pipeline failed", time.time() - start)
            return

        out, df_feat, proba, _ = scored
        out, df_feat, proba = apply_customer_exclusion(out, df_feat, proba)

        # ── Step 3: build and persist customer risk ─────────────────────
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

        # ── Step 4: mark job complete ───────────────────────────────────
        complete_job(
            job_id,
            total_invoices=int(out.shape[0]),
            total_customers=int(customer_risk.shape[0]),
            risk_summary=risk_summary,
            customer_risk_summary=cr_summary,
            duration_sec=time.time() - start,
        )
        logger.info(
            "Compute job %s completed: %d invoices, %d customers in %.1fs",
            job_id,
            out.shape[0],
            customer_risk.shape[0],
            time.time() - start,
        )

        # ── Step 5: replace old SQLite partition rows ───────────────────
        _replace_old_partition_rows(job_id)

        # ── Step 6: publish to MySQL ────────────────────────────────────
        _auto_publish_score_best_effort(
            job_id=job_id,
            model_key=model_key,
            snapshot_date=snapshot_date,
            time_range=time_range,
            df_score=out,
        )

        # ── Step 7: cleanup old MySQL rows ──────────────────────────────
        _cleanup_mysql_best_effort(job_id)

        # ── Step 8: cleanup old SQLite jobs ────────────────────────────
        try:
            cleanup_old_jobs(COMPUTE_KEEP_DAYS)
        except Exception:
            logger.exception("Cleanup old SQLite jobs failed after compute %s", job_id)

    except Exception as exc:
        logger.exception("Compute job %s failed unexpectedly", job_id)
        fail_job(job_id, str(exc), time.time() - start)


# ── Endpoints ──────────────────────────────────────────────────────────


@router.post("/db/compute")
async def compute(
    background_tasks: BackgroundTasks,
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str | None = Query(None),
    time_range: str = Query(COMPUTE_DEFAULT_TIME_RANGE),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
):
    """Trigger the scoring pipeline.

    Runs asynchronously in the background; results are stored to local SQLite
    and optionally published to MySQL hasil_baddebt.
    Returns HTTP 202 immediately with a job_id for status polling.
    """
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
    """Get the status of a specific compute job by ID."""
    job = get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Job not found."})
    # Remove internal JSON blobs from the response
    job.pop("risk_summary_json", None)
    job.pop("customer_risk_summary_json", None)
    return job


@router.get("/db/compute/history")
def compute_history(limit: int = Query(20, ge=1, le=100)):
    """List recent compute jobs, newest first."""
    return {"jobs": get_all_jobs(limit=limit)}


@router.post("/db/compute/publish")
def publish_compute_result(
    job_id: str | None = Query(None),
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str | None = Query(None),
    time_range: str = Query(COMPUTE_DEFAULT_TIME_RANGE),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    table_name: str = Query("hasil_baddebt"),
    replace_partition: bool = Query(True),
):
    """Legacy endpoint — score publish now happens automatically during compute.

    Kept for API compatibility; always returns HTTP 410 Gone.
    """
    snapshot_date = _resolve_snapshot_date(snapshot_date)

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

    return JSONResponse(
        status_code=410,
        content={
            "error": (
                "Manual publish from local score storage is no longer supported. "
                "Score results are published to MySQL during compute execution."
            ),
            "job_id": target_job.get("job_id"),
            "target_table": table_name,
            "replace_partition": replace_partition,
        },
    )
