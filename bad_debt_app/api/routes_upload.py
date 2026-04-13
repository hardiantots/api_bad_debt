from __future__ import annotations

from typing import Optional
from datetime import date

from fastapi import APIRouter, File, Form, UploadFile

from bad_debt_app.api.config import (
    DEFAULT_MODEL_KEY,
    THRESHOLD_HIGH,
    THRESHOLD_LOW,
)
from bad_debt_app.api.service import (
    build_customer_risk,
    build_customer_risk_summary,
    build_raw_inputs,
    get_top_efl,
    load_schema,
    read_upload_bundle,
    records,
    safe_threshold,
    score_snapshot,
)

router = APIRouter()


def _resolve_snapshot_date(snapshot_date: str | None) -> str:
    return snapshot_date or date.today().isoformat()


@router.post("/score")
async def score(
    invoice_csv: UploadFile = File(...),
    receipt_csv: UploadFile = File(...),
    customer_json: Optional[UploadFile] = File(None),
    model: str = Form(DEFAULT_MODEL_KEY),
    snapshot_date: str | None = Form(None),
    customer_format: Optional[str] = Form(None),
):
    snapshot_date = _resolve_snapshot_date(snapshot_date)

    upload_result = await read_upload_bundle(invoice_csv, receipt_csv, customer_json)
    if not isinstance(upload_result, tuple):
        return upload_result

    inv_bytes, rcp_bytes, cust_bytes = upload_result
    raw = build_raw_inputs(
        inv_bytes,
        rcp_bytes,
        cust_bytes,
        customer_name=customer_json.filename if customer_json else None,
        customer_format=customer_format,
    )

    scored = score_snapshot(raw=raw, snapshot_date=snapshot_date, model_key=model)
    if not isinstance(scored, tuple):
        return scored

    out, df_feat, proba, m_info = scored
    customer_risk = build_customer_risk(df_feat, proba)
    risk_summary = out["risk_level"].value_counts().to_dict()
    customer_risk_summary = build_customer_risk_summary(customer_risk)

    return {
        "mode": "snapshot",
        "model_key": m_info.get("key", model),
        "model_flow": m_info.get("training_flow"),
        "label_strategy": m_info.get("label_strategy"),
        "snapshot_date": snapshot_date,
        "total_invoices": int(out.shape[0]),
        "risk_summary": risk_summary,
        "high_risk_count": int(risk_summary.get("HIGH", 0)),
        "preview": records(out.head(20)),
        "top_efl_invoices": get_top_efl(out, 50),
        "customer_risk_summary": customer_risk_summary,
        "customer_risk": records(customer_risk),
    }


@router.post("/score_csv")
async def score_csv(
    invoice_csv: UploadFile = File(...),
    receipt_csv: UploadFile = File(...),
    customer_json: Optional[UploadFile] = File(None),
    model: str = Form(DEFAULT_MODEL_KEY),
    snapshot_date: str | None = Form(None),
    customer_format: Optional[str] = Form(None),
):
    from fastapi.responses import Response

    snapshot_date = _resolve_snapshot_date(snapshot_date)

    upload_result = await read_upload_bundle(invoice_csv, receipt_csv, customer_json)
    if not isinstance(upload_result, tuple):
        return upload_result

    inv_bytes, rcp_bytes, cust_bytes = upload_result
    raw = build_raw_inputs(
        inv_bytes,
        rcp_bytes,
        cust_bytes,
        customer_name=customer_json.filename if customer_json else None,
        customer_format=customer_format,
    )

    scored = score_snapshot(raw=raw, snapshot_date=snapshot_date, model_key=model)
    if not isinstance(scored, tuple):
        return scored

    out, _, _, _ = scored
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    filename = f"bad_debt_snapshot_{snapshot_date}.csv".replace(":", "-")

    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/alerts")
async def alerts(
    invoice_csv: UploadFile = File(...),
    receipt_csv: UploadFile = File(...),
    customer_json: Optional[UploadFile] = File(None),
    model: str = Form(DEFAULT_MODEL_KEY),
    snapshot_date: str | None = Form(None),
    threshold: float = Form(0.3),
    customer_format: Optional[str] = Form(None),
):
    snapshot_date = _resolve_snapshot_date(snapshot_date)

    threshold = safe_threshold(threshold)

    upload_result = await read_upload_bundle(invoice_csv, receipt_csv, customer_json)
    if not isinstance(upload_result, tuple):
        return upload_result

    inv_bytes, rcp_bytes, cust_bytes = upload_result
    raw = build_raw_inputs(
        inv_bytes,
        rcp_bytes,
        cust_bytes,
        customer_name=customer_json.filename if customer_json else None,
        customer_format=customer_format,
    )

    scored = score_snapshot(raw=raw, snapshot_date=snapshot_date, model_key=model)
    if not isinstance(scored, tuple):
        return scored

    out, df_feat, proba, m_info = scored
    alerts_df = out[out["prob_bad_debt"] >= threshold].sort_values(
        "prob_bad_debt", ascending=False
    )

    customer_risk = build_customer_risk(df_feat, proba)
    customer_risk_summary = build_customer_risk_summary(customer_risk)

    return {
        "mode": "snapshot",
        "threshold": threshold,
        "model_key": m_info.get("key", model),
        "model_flow": m_info.get("training_flow"),
        "label_strategy": m_info.get("label_strategy"),
        "snapshot_date": snapshot_date,
        "total_invoices": int(out.shape[0]),
        "alerts_count": int(alerts_df.shape[0]),
        "risk_summary": (
            alerts_df["risk_level"].value_counts().to_dict()
            if not alerts_df.empty
            else {}
        ),
        "alerts": records(alerts_df),
        "top_efl_invoices": get_top_efl(out, 50),
        "customer_risk_summary": customer_risk_summary,
        "customer_risk": records(customer_risk),
    }


@router.post("/early_warning/receipt_trigger")
async def receipt_trigger(
    invoice_csv: UploadFile = File(...),
    receipt_csv: UploadFile = File(...),
    customer_json: Optional[UploadFile] = File(None),
    model: str = Form(DEFAULT_MODEL_KEY),
    snapshot_date: str | None = Form(None),
    customer_format: Optional[str] = Form(None),
):
    snapshot_date = _resolve_snapshot_date(snapshot_date)

    upload_result = await read_upload_bundle(invoice_csv, receipt_csv, customer_json)
    if not isinstance(upload_result, tuple):
        return upload_result

    inv_bytes, rcp_bytes, cust_bytes = upload_result
    raw = build_raw_inputs(
        inv_bytes,
        rcp_bytes,
        cust_bytes,
        customer_name=customer_json.filename if customer_json else None,
        customer_format=customer_format,
    )

    scored = score_snapshot(raw=raw, snapshot_date=snapshot_date, model_key=model)
    if not isinstance(scored, tuple):
        return scored

    df_result, df_feat, proba, m_info = scored
    feature_cols = load_schema(m_info["schema_path"])
    alerts_df = df_result[df_result["prob_bad_debt"] >= THRESHOLD_LOW].sort_values(
        "prob_bad_debt", ascending=False
    )

    customer_risk = build_customer_risk(df_feat, proba)
    customer_risk_summary = build_customer_risk_summary(customer_risk)

    return {
        "mode": "early_warning",
        "analysis_type": "Early-Warning (Pre-Due Analysis)",
        "model_key": m_info.get("key", model),
        "model_flow": m_info.get("training_flow"),
        "label_strategy": m_info.get("label_strategy"),
        "model_label": m_info["label"],
        "feature_count": len(feature_cols),
        "snapshot_date": snapshot_date,
        "processed_invoices": int(df_result.shape[0]),
        "risk_summary": df_result["risk_level"].value_counts().to_dict(),
        "alerts_count": int(alerts_df.shape[0]),
        "high_risk_count": int((proba >= THRESHOLD_HIGH).sum()),
        "alerts": records(alerts_df),
        "all_scores_preview": records(df_result),
        "top_efl_invoices": get_top_efl(df_result, 50),
        "customer_risk_summary": customer_risk_summary,
        "customer_risk": records(customer_risk),
    }
