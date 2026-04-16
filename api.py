from __future__ import annotations

import logging
import os
import secrets
import time
from datetime import date

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from bad_debt_app.api.config import (
    ALLOW_ORIGINS,
    API_KEY,
    APP_TITLE,
    BASE_DIR,
    COMPUTE_AUTO_ENABLED,
    COMPUTE_AUTO_PUBLISH_SCORE_TO_MYSQL,
    COMPUTE_AUTO_RECOVER_STALE,
    COMPUTE_DEFAULT_TIME_RANGE,
    COMPUTE_KEEP_DAYS,
    COMPUTE_MAX_RUNNING_MINUTES,
    COMPUTE_PUBLISH_REPLACE_PARTITION,
    COMPUTE_PUBLISH_TARGET_TABLE,
    COMPUTE_SCHEDULE_HOUR,
    DEFAULT_MODEL_KEY,
)
from bad_debt_app.api.routes_compute import router as compute_router
from bad_debt_app.api.routes_db import router as db_router
from bad_debt_app.api.routes_system import router as system_router
from bad_debt_app.api.routes_upload import router as upload_router

logger = logging.getLogger("bad_debt_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

web_dir = BASE_DIR / "web"
if web_dir.is_dir():
    app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")


@app.middleware("http")
async def add_timing(request: Request, call_next):
    if API_KEY:
        public_paths = {"/health", "/docs", "/openapi.json", "/redoc", "/"}
        if (
            not request.url.path.startswith("/web")
            and request.url.path not in public_paths
        ):
            req_api_key = (
                request.headers.get("x-api-key")
                or request.headers.get("authorization", "")
                .removeprefix("Bearer ")
                .strip()
            )
            if not req_api_key or not secrets.compare_digest(req_api_key, API_KEY):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"error": "Invalid or missing API key."},
                )

    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    response.headers["X-Process-Time"] = f"{duration:.3f}s"
    logger.info(
        "%s %s -> %s in %.3fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


app.include_router(system_router)
app.include_router(upload_router)
app.include_router(db_router)
app.include_router(compute_router)


# ── Startup: ensure tables + scheduler ────────────────────────────────

_scheduler = None


def _auto_compute():
    """Called by scheduler to run scoring with default parameters."""
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
        has_running_job,
        insert_customer_risk,
        recover_stale_running_jobs,
    )

    if COMPUTE_AUTO_RECOVER_STALE:
        recover_stale_running_jobs(
            COMPUTE_MAX_RUNNING_MINUTES,
            reason="Auto-recovered stale running job before scheduler compute",
        )

    if has_running_job():
        logger.info("Scheduler: skipping, a compute job is already running")
        return

    model_key = resolve_model_key(DEFAULT_MODEL_KEY)
    m_info = resolve_model(model_key)
    job_id = generate_job_id()
    tr = COMPUTE_DEFAULT_TIME_RANGE
    sd = date.today().isoformat()

    logger.info(
        "Scheduler: starting auto-compute job %s (model=%s, range=%s, snapshot=%s)",
        job_id,
        model_key,
        tr,
        sd,
    )
    create_job(
        job_id,
        model_key=model_key,
        model_label=m_info.get("label", model_key),
        model_flow=m_info.get("training_flow"),
        label_strategy=m_info.get("label_strategy"),
        snapshot_date=sd,
        time_range=tr,
    )

    start = time.time()
    try:
        raw = fetch_raw_from_db(
            time_range=tr, start_date=None, end_date=None, snapshot_date=sd
        )
        if isinstance(raw, JSONResponse):
            fail_job(job_id, "DB fetch failed", time.time() - start)
            return
        if raw.invoice.empty:
            fail_job(job_id, "No invoices found", time.time() - start)
            return

        scored = score_snapshot(raw=raw, snapshot_date=sd, model_key=model_key)
        if isinstance(scored, JSONResponse):
            fail_job(job_id, "Scoring failed", time.time() - start)
            return

        out, df_feat, proba, _ = scored
        out, df_feat, proba = apply_customer_exclusion(out, df_feat, proba)
        customer_risk = build_customer_risk(df_feat, proba)
        customer_risk = filter_excluded_customers(
            customer_risk, name_col="CUSTOMER_NAME"
        )

        insert_customer_risk(job_id, sd, tr, model_key, customer_risk)

        if COMPUTE_AUTO_PUBLISH_SCORE_TO_MYSQL:
            try:
                from bad_debt_app.data.db import publish_score_to_hasil_baddebt

                summary = publish_score_to_hasil_baddebt(
                    df_score=out,
                    model_key=model_key,
                    snapshot_date=sd,
                    time_range=tr,
                    source_job_id=job_id,
                    target_table=COMPUTE_PUBLISH_TARGET_TABLE,
                    replace_partition=COMPUTE_PUBLISH_REPLACE_PARTITION,
                )
                logger.info(
                    "Scheduler auto-published score for job %s to %s (%d rows)",
                    job_id,
                    summary.get("table", COMPUTE_PUBLISH_TARGET_TABLE),
                    int(summary.get("rows_inserted", 0)),
                )
            except Exception:
                logger.exception(
                    "Scheduler auto-publish score failed for job %s (table=%s)",
                    job_id,
                    COMPUTE_PUBLISH_TARGET_TABLE,
                )

        complete_job(
            job_id,
            total_invoices=int(out.shape[0]),
            total_customers=int(customer_risk.shape[0]),
            risk_summary=out["risk_level"].value_counts().to_dict(),
            customer_risk_summary=build_customer_risk_summary(customer_risk),
            duration_sec=time.time() - start,
        )
        logger.info("Scheduler: job %s completed in %.1fs", job_id, time.time() - start)
    except Exception as exc:
        logger.exception("Scheduler: job %s failed", job_id)
        fail_job(job_id, str(exc), time.time() - start)

    # Cleanup old data
    try:
        cleanup_old_jobs(COMPUTE_KEEP_DAYS)
    except Exception:
        logger.exception("Scheduler: cleanup failed")


@app.on_event("startup")
async def startup():
    from bad_debt_app.data.models import ensure_tables, recover_stale_running_jobs

    ensure_tables()
    logger.info("Local SQLite tables initialized")

    if COMPUTE_AUTO_RECOVER_STALE:
        recovered = recover_stale_running_jobs(
            COMPUTE_MAX_RUNNING_MINUTES,
            reason="Auto-recovered stale running job during startup",
        )
        if recovered > 0:
            logger.warning(
                "Startup recovery marked %d stale running job(s) as failed", recovered
            )

    if COMPUTE_AUTO_ENABLED:
        try:
            # Use BackgroundScheduler (runs in a daemon thread) so that
            # _auto_compute (a synchronous, long-running function) does NOT
            # block the FastAPI/uvicorn asyncio event loop.
            from apscheduler.schedulers.background import BackgroundScheduler

            global _scheduler
            _scheduler = BackgroundScheduler()
            _scheduler.add_job(
                _auto_compute,
                trigger="cron",
                hour=COMPUTE_SCHEDULE_HOUR,
                minute=0,
                id="daily_scoring",
                replace_existing=True,
                misfire_grace_time=3600,  # allow up to 1h late start
            )
            _scheduler.start()

            # Confirm next scheduled run in the log
            job = _scheduler.get_job("daily_scoring")
            next_run = job.next_run_time if job else "unknown"
            logger.info(
                "APScheduler (BackgroundScheduler) started: auto-compute daily at %02d:00 — next run: %s",
                COMPUTE_SCHEDULE_HOUR,
                next_run,
            )
        except ImportError:
            logger.warning(
                "apscheduler not installed — auto-compute scheduler disabled. "
                "Install with: pip install apscheduler"
            )


@app.on_event("shutdown")
async def shutdown():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        logger.info("APScheduler shut down")
