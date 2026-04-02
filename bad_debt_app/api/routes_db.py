from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, Response

from bad_debt_app.api.config import (
    DEFAULT_MODEL_KEY,
    DEFAULT_SNAPSHOT_DATE,
    THRESHOLD_HIGH,
    THRESHOLD_LOW,
)
from bad_debt_app.api.service import (
    apply_customer_exclusion,
    build_customer_risk,
    build_customer_risk_summary,
    fetch_raw_from_db,
    filter_excluded_customers,
    get_top_efl,
    load_schema,
    model_public_info,
    records,
    safe_threshold,
    score_snapshot,
)
from bad_debt_app.data.db import TIME_RANGE_OPTIONS, get_data_date_range

router = APIRouter()


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


@router.get("/db/score")
def db_score(
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str = Query(DEFAULT_SNAPSHOT_DATE),
    time_range: str = Query("1w"),
    start_date: str = Query(None),
    end_date: str = Query(None),
):
    raw = fetch_raw_from_db(
        time_range=time_range,
        start_date=start_date,
        end_date=end_date,
        snapshot_date=snapshot_date,
    )
    if isinstance(raw, JSONResponse):
        return raw
    if raw.invoice.empty:
        return JSONResponse(
            status_code=200,
            content={"warning": "No invoices found.", "total_invoices": 0},
        )

    scored = score_snapshot(raw=raw, snapshot_date=snapshot_date, model_key=model)
    if isinstance(scored, JSONResponse):
        return scored

    out, df_feat, proba, m_info = scored
    out, df_feat, proba = apply_customer_exclusion(out, df_feat, proba)
    customer_risk = build_customer_risk(df_feat, proba)
    customer_risk = filter_excluded_customers(customer_risk, name_col="CUSTOMER_NAME")
    risk_summary = out["risk_level"].value_counts().to_dict()

    return {
        "mode": "snapshot",
        "model_key": m_info.get("key", model),
        "model_flow": m_info.get("training_flow"),
        "label_strategy": m_info.get("label_strategy"),
        "snapshot_date": snapshot_date,
        "time_range": time_range,
        "model_label": m_info["label"],
        "total_invoices": int(out.shape[0]),
        "risk_summary": risk_summary,
        "high_risk_count": int(risk_summary.get("HIGH", 0)),
        "preview": records(out),
        "top_efl_invoices": get_top_efl(out, 50),
        "customer_risk_summary": build_customer_risk_summary(customer_risk),
        "customer_risk": records(customer_risk),
    }


@router.get("/db/score_csv")
def db_score_csv(
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str = Query(DEFAULT_SNAPSHOT_DATE),
    time_range: str = Query("1w"),
    start_date: str = Query(None),
    end_date: str = Query(None),
):
    raw = fetch_raw_from_db(
        time_range=time_range,
        start_date=start_date,
        end_date=end_date,
        snapshot_date=snapshot_date,
    )
    if isinstance(raw, JSONResponse):
        return raw

    scored = score_snapshot(raw=raw, snapshot_date=snapshot_date, model_key=model)
    if isinstance(scored, JSONResponse):
        return scored

    out, df_feat, proba, _ = scored
    out, _, _ = apply_customer_exclusion(out, df_feat, proba)
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    filename = f"bad_debt_{time_range}_{snapshot_date}.csv".replace(":", "-")
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/db/alerts")
def db_alerts(
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str = Query(DEFAULT_SNAPSHOT_DATE),
    time_range: str = Query("1w"),
    threshold: float = Query(0.3),
    start_date: str = Query(None),
    end_date: str = Query(None),
):
    threshold = safe_threshold(threshold)
    raw = fetch_raw_from_db(
        time_range=time_range,
        start_date=start_date,
        end_date=end_date,
        snapshot_date=snapshot_date,
    )
    if isinstance(raw, JSONResponse):
        return raw
    if raw.invoice.empty:
        return JSONResponse(
            status_code=200, content={"warning": "No invoices.", "alerts_count": 0}
        )

    scored = score_snapshot(raw=raw, snapshot_date=snapshot_date, model_key=model)
    if isinstance(scored, JSONResponse):
        return scored

    out, df_feat, proba, m_info = scored
    out, df_feat, proba = apply_customer_exclusion(out, df_feat, proba)
    alerts_df = out[out["prob_bad_debt"] >= threshold].sort_values(
        "prob_bad_debt", ascending=False
    )
    customer_risk = build_customer_risk(df_feat, proba)
    customer_risk = filter_excluded_customers(customer_risk, name_col="CUSTOMER_NAME")

    return {
        "mode": "snapshot",
        "threshold": threshold,
        "model_key": m_info.get("key", model),
        "model_flow": m_info.get("training_flow"),
        "label_strategy": m_info.get("label_strategy"),
        "snapshot_date": snapshot_date,
        "time_range": time_range,
        "model_label": m_info["label"],
        "total_invoices": int(out.shape[0]),
        "alerts_count": int(alerts_df.shape[0]),
        "risk_summary": (
            alerts_df["risk_level"].value_counts().to_dict()
            if not alerts_df.empty
            else {}
        ),
        "alerts": records(alerts_df),
        "top_efl_invoices": get_top_efl(out, 50),
        "customer_risk_summary": build_customer_risk_summary(customer_risk),
        "customer_risk": records(customer_risk),
    }


@router.get("/db/early_warning/receipt_trigger")
def db_receipt_trigger(
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str = Query(DEFAULT_SNAPSHOT_DATE),
    time_range: str = Query("1w"),
    start_date: str = Query(None),
    end_date: str = Query(None),
):
    raw = fetch_raw_from_db(
        time_range=time_range,
        start_date=start_date,
        end_date=end_date,
        snapshot_date=snapshot_date,
    )
    if isinstance(raw, JSONResponse):
        return raw
    if raw.invoice.empty:
        return JSONResponse(
            status_code=200,
            content={"warning": "No invoices.", "processed_invoices": 0},
        )

    scored = score_snapshot(raw=raw, snapshot_date=snapshot_date, model_key=model)
    if isinstance(scored, JSONResponse):
        return scored

    df_result, df_feat, proba, m_info = scored
    df_result, df_feat, proba = apply_customer_exclusion(df_result, df_feat, proba)
    alerts_df = df_result[df_result["prob_bad_debt"] >= THRESHOLD_LOW].sort_values(
        "prob_bad_debt", ascending=False
    )
    customer_risk = build_customer_risk(df_feat, proba)
    customer_risk = filter_excluded_customers(customer_risk, name_col="CUSTOMER_NAME")
    feature_cols = load_schema(m_info["schema_path"])

    return {
        "mode": "early_warning",
        "analysis_type": "Early-Warning (Pre-Due Analysis)",
        "analysis_description": "Mengevaluasi risiko bad debt SEBELUM jatuh tempo.",
        "model_key": m_info.get("key", model),
        "model_flow": m_info.get("training_flow"),
        "label_strategy": m_info.get("label_strategy"),
        "model_label": m_info["label"],
        "feature_count": len(feature_cols),
        "snapshot_date": snapshot_date,
        "time_range": time_range,
        "processed_invoices": int(df_result.shape[0]),
        "risk_summary": df_result["risk_level"].value_counts().to_dict(),
        "alerts_count": int(alerts_df.shape[0]),
        "high_risk_count": int((proba >= THRESHOLD_HIGH).sum()),
        "alerts": records(alerts_df),
        "all_scores_preview": records(df_result),
        "top_efl_invoices": get_top_efl(df_result, 50),
        "customer_risk_summary": build_customer_risk_summary(customer_risk),
        "customer_risk": records(customer_risk),
    }
