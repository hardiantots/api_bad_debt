from __future__ import annotations

import io
import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

# ── Logging ──
logger = logging.getLogger("bad_debt_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

import sklearn

# ── Global Sklearn Config ──
# Ensures sklearn transformers (like SMOTETomek, MinMaxScaler) output pandas DataFrames
# avoiding "X does not have valid feature names" warnings down the pipeline.
sklearn.set_config(transform_output="pandas")

from bad_debt_app.features import (
    load_raw_inputs,
    prepare_snapshot_features,
)
from bad_debt_app.db import TIME_RANGE_OPTIONS, fetch_raw_inputs

APP_TITLE = "Bad Debt Early-Warning API"

# ── Model & Schema Paths ──
_BASE_DIR = Path(__file__).resolve().parent
MODEL_REGISTRY = {
    "stacked": {
        "model_path": str(_BASE_DIR / "artifacts/stacked_recall_driven_model.joblib"),
        "schema_path": str(_BASE_DIR / "artifacts/feature_cols_stacked.json"),
        "label": "Model Stacked (LightGBM + LR)",
        "training_flow": "Test Stacked SMOTE, Auto Search Parameter & Updated",
        "label_strategy": "y_bad_debt_ever (3 kondisi)",
        "credit_memo_policy": "netting PREVIOUS_CUSTOMER_TRX_ID sampai snapshot_date",
    },
    "lgbm_hyper_smote": {
        "model_path": str(
            _BASE_DIR
            / "artifacts/bad_debt_snapshot_lgbm_hyper_smote_16_features.joblib"
        ),
        "schema_path": str(
            _BASE_DIR / "artifacts/feature_cols_snapshot_16_features.json"
        ),
        "label": "LightGBM",
        "training_flow": "Test_New_CM_SMOTE",
        "label_strategy": "y_bad_debt_ever (3 kondisi)",
        "credit_memo_policy": "netting PREVIOUS_CUSTOMER_TRX_ID sampai snapshot_date",
    },
}
DEFAULT_MODEL_KEY = "stacked"
DEFAULT_SNAPSHOT_DATE = os.getenv("BAD_DEBT_SNAPSHOT_DATE", "2026-03-15")

# ── Auto-fallback customer file ──
DEFAULT_CUSTOMER_CSV = _BASE_DIR / "artifacts" / "OracleCustomer_slim.csv"

# ── Risk Thresholds ──
THRESHOLD_LOW = float(os.getenv("THRESHOLD_LOW", "0.3"))
THRESHOLD_HIGH = float(os.getenv("THRESHOLD_HIGH", "0.6"))

# ── Upload size limit ──
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

# ── Optional API key ──
API_KEY = os.getenv("API_KEY")  # Set in .env to enable auth

app = FastAPI(title=APP_TITLE)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount web/ folder for static frontend ──
_WEB_DIR = _BASE_DIR / "web"
if _WEB_DIR.is_dir():
    app.mount("/web", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")


# ── Request timing middleware ──
@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    response.headers["X-Process-Time"] = f"{duration:.3f}s"
    logger.info(
        "%s %s → %s in %.3fs",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


# ═══════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════


def _resolve_model(model_key: str | None) -> dict:
    if not model_key or model_key not in MODEL_REGISTRY:
        return MODEL_REGISTRY[DEFAULT_MODEL_KEY]
    return MODEL_REGISTRY[model_key]


def _resolve_model_key(model_key: str | None) -> str:
    if not model_key or model_key not in MODEL_REGISTRY:
        return DEFAULT_MODEL_KEY
    return model_key


def _model_public_info(model_key: str, model_info: dict) -> dict:
    return {
        "key": model_key,
        "label": model_info.get("label", model_key),
        "training_flow": model_info.get("training_flow"),
        "label_strategy": model_info.get("label_strategy"),
        "credit_memo_policy": model_info.get("credit_memo_policy"),
    }


def _load_schema(path: str) -> list[str]:
    """Load and cache feature column schema from JSON."""
    return _load_schema_cached(path)


@lru_cache(maxsize=4)
def _load_schema_cached(path: str) -> list[str]:
    with open(path, "r") as f:
        return json.load(f)


@lru_cache(maxsize=2)
def _load_model_cached(path: str):
    """Load and cache model artifact from disk."""
    logger.info("Loading model from %s", Path(path).name)
    return joblib.load(path)


def _validate_snapshot_date(date_str: str) -> pd.Timestamp | JSONResponse:
    """Validate and parse snapshot_date string."""
    try:
        return pd.Timestamp(date_str)
    except (ValueError, TypeError):
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Invalid snapshot_date format: '{date_str}'. Use YYYY-MM-DD."
            },
        )


async def _read_upload_with_limit(file: UploadFile) -> bytes | JSONResponse:
    """Read upload file bytes, enforcing MAX_UPLOAD_BYTES limit."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "error": f"File '{file.filename}' exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit."
            },
        )
    return data


def _records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame rows to JSON-safe list of dicts.

    pandas `.to_dict(orient='records')` keeps Python float('nan') which is
    *not* valid JSON.  Going through ``to_json`` → ``json.loads`` maps NaN
    and NaT to ``null``, avoiding the ``ValueError`` that FastAPI / Starlette
    raises when it tries ``json.dumps`` on the raw dict.
    """
    return json.loads(df.to_json(orient="records"))


def _classify_risk(prob: float) -> str:
    """Classify probability into risk level."""
    if prob >= THRESHOLD_HIGH:
        return "HIGH"
    elif prob >= THRESHOLD_LOW:
        return "MEDIUM"
    return "LOW"


def _recommend_action(risk: str) -> str:
    """Return recommended action based on risk level."""
    actions = {
        "HIGH": "ESKALASI - Follow-up tim collection segera",
        "MEDIUM": "WATCHLIST - Monitor & kirim reminder pembayaran",
        "LOW": "OK - Monitoring rutin",
    }
    return actions.get(risk, "Unknown")


def _score_snapshot(
    *, raw, snapshot_date: str, model_key: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict] | JSONResponse:
    """Run snapshot scoring and return (df_result, df_feat, proba, model_info)."""
    snap_result = _validate_snapshot_date(snapshot_date)
    if isinstance(snap_result, JSONResponse):
        return snap_result
    snap = snap_result

    resolved_model_key = _resolve_model_key(model_key)
    model_info = dict(_resolve_model(resolved_model_key))
    model_info["key"] = resolved_model_key
    model_path = model_info["model_path"]
    schema_path = model_info["schema_path"]
    df_feat, _ = prepare_snapshot_features(raw, snapshot_date=snap)

    feature_cols = _load_schema(schema_path)
    for c in feature_cols:
        if c not in df_feat.columns:
            df_feat[c] = np.nan  # use np.nan for sklearn compatibility

    X = df_feat[feature_cols].copy()
    # Ensure all NA variants are converted to np.nan for sklearn
    X = X.fillna(value=np.nan).infer_objects(copy=False)
    try:
        if str(model_path).endswith(".keras"):
            import tensorflow as tf

            model = tf.keras.models.load_model(model_path)
            preprocess_path = str(model_path).replace(".keras", "_preprocess.joblib")
            if os.path.exists(preprocess_path):
                preprocess = joblib.load(preprocess_path)
                X_processed = preprocess.transform(X)
                if hasattr(X_processed, "toarray"):
                    X_processed = X_processed.toarray()
            else:
                X_processed = X.values
            X_lstm = X_processed.reshape(
                (X_processed.shape[0], 1, X_processed.shape[1])
            )
            proba = model.predict(X_lstm).flatten()

        else:
            model_obj = _load_model_cached(model_path)

            # Check if this is our new Stacked Dictionary Artifact
            if isinstance(model_obj, dict) and "meta_model" in model_obj:
                lgb_b = model_obj["lgb_model_B"]
                lr_m = model_obj["lr_model"]
                meta = model_obj["meta_model"]
                prep = model_obj.get("preprocessor")

                # Preprocess if scaler exists
                if prep is not None:
                    try:
                        X_prep = prep.transform(X)
                    except (ValueError, TypeError):
                        logger.warning(
                            "Preprocessor transform failed, falling back to fillna(0)"
                        )
                        X_prep = prep.transform(X.fillna(0))
                else:
                    X_prep = X.values

                # Base predictions
                oof_lgb = lgb_b.predict_proba(X)[:, 1]
                # LogisticRegression expects numpy array (no feature names)
                oof_lr = lr_m.predict_proba(np.asarray(X_prep))[:, 1]

                # Meta features
                X_meta = pd.DataFrame({"oof_lgb_B": oof_lgb, "oof_lr": oof_lr})

                # Meta predict (meta model was fitted without names for stack)
                raw_proba = meta.predict_proba(np.asarray(X_meta))[:, 1]

                # Calibrate if exists
                cal_method = model_obj.get("calibration_method")
                if cal_method == "platt" and model_obj.get("platt_model") is not None:
                    proba = model_obj["platt_model"].predict_proba(
                        raw_proba.reshape(-1, 1)
                    )[:, 1]
                elif (
                    cal_method == "isotonic" and model_obj.get("iso_model") is not None
                ):
                    proba = model_obj["iso_model"].predict(raw_proba)
                else:
                    proba = raw_proba

            else:
                # Standard Scikit-Learn / LightGBM fallback
                model = model_obj
                proba = model.predict_proba(X)[:, 1]

    except FileNotFoundError:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Model file not found.",
                "hint": "Pastikan model sudah di-train dan disimpan ke artifacts/",
            },
        )
    except ModuleNotFoundError as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to load model due to missing dependency.",
                "hint": "pip install lightgbm / tensorflow",
            },
        )
    except Exception as e:
        logger.exception("Scoring failed unexpectedly")
        return JSONResponse(
            status_code=500,
            content={"error": "Scoring failed. Check server logs for details."},
        )

    cols: dict[str, object] = {}
    if "ACCOUNT_NUMBER" in df_feat.columns and df_feat["ACCOUNT_NUMBER"].notna().any():
        cols["ACCOUNT_NUMBER"] = df_feat["ACCOUNT_NUMBER"]
    if "CUSTOMER_NAME" in df_feat.columns and df_feat["CUSTOMER_NAME"].notna().any():
        cols["CUSTOMER_NAME"] = df_feat["CUSTOMER_NAME"]
    cols["CUSTOMER_TRX_ID"] = df_feat.get("CUSTOMER_TRX_ID")

    if "TRX_DATE" in df_feat.columns:
        try:
            cols["TRX_DATE"] = pd.to_datetime(
                df_feat["TRX_DATE"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        except Exception:
            cols["TRX_DATE"] = df_feat["TRX_DATE"].astype(str)

    if "DUE_DATE" in df_feat.columns:
        try:
            cols["DUE_DATE"] = pd.to_datetime(
                df_feat["DUE_DATE"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        except Exception:
            cols["DUE_DATE"] = df_feat["DUE_DATE"].astype(str)

    if "days_to_due" in df_feat.columns:
        cols["days_to_due"] = df_feat["days_to_due"]

    if "TRX_AMOUNT" in df_feat.columns:
        if "TRX_AMOUNT_GROSS" in df_feat.columns:
            cols["TRX_AMOUNT_GROSS"] = df_feat["TRX_AMOUNT_GROSS"]
        if "credit_memo_reduction" in df_feat.columns:
            cols["credit_memo_reduction"] = df_feat["credit_memo_reduction"]
        cols["TRX_AMOUNT"] = df_feat["TRX_AMOUNT"]
        try:
            amt_num = (
                pd.to_numeric(df_feat["TRX_AMOUNT"], errors="coerce").fillna(0).abs()
            )
            cols["expected_financial_loss"] = amt_num * proba
        except Exception:
            logger.warning("EFL calculation failed, defaulting to zero")
            cols["expected_financial_loss"] = [0] * len(proba)
    cols["prob_bad_debt"] = proba
    cols["risk_level"] = [_classify_risk(p) for p in proba]
    cols["recommended_action"] = [_recommend_action(_classify_risk(p)) for p in proba]

    df_result = pd.DataFrame(cols)

    # -------- Two-Pass Output Filtering --------
    # If the request used two-pass (DB mode), filter to the target requested invoices
    # so we don't output scores for 3 years of historical background invoices.
    if getattr(raw, "target_trx_ids", None) is not None:
        target_ids = (
            pd.Series(raw.target_trx_ids)
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .astype("int64")
            .tolist()
        )
        mask = df_result["CUSTOMER_TRX_ID"].isin(target_ids)
        df_result = df_result[mask].copy()
        df_feat = df_feat[mask].copy()
        proba = proba[mask.values].copy()

    return df_result, df_feat, proba, model_info


def _compute_scores(*, raw, snapshot_date: str) -> pd.DataFrame | JSONResponse:
    """Core scoring function for early-warning snapshot mode (SMOTE model)."""
    scored = _score_snapshot(raw=raw, snapshot_date=snapshot_date)
    if isinstance(scored, JSONResponse):
        return scored
    df_result, _, _, _ = scored
    return df_result


def _build_customer_risk(df_feat: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Aggregate invoice scores into customer risk metrics."""
    if df_feat.empty:
        return pd.DataFrame()

    df = df_feat.copy()
    df["prob_bad_debt"] = proba

    id_cols: list[str] = []
    if "PARTY_ID" in df.columns:
        id_cols.append("PARTY_ID")
    for col in ["ACCOUNT_NUMBER", "CUSTOMER_NAME"]:
        if col in df.columns:
            id_cols.append(col)

    if not id_cols:
        return pd.DataFrame()

    amt = pd.to_numeric(df.get("TRX_AMOUNT"), errors="coerce").fillna(0).abs()
    paid = pd.to_numeric(df.get("amt_paid_pre_due"), errors="coerce").fillna(0)
    paid_ratio = pd.to_numeric(df.get("paid_ratio_pre_due"), errors="coerce")

    gap_flag = None
    if "count_gaps_gt_90_pre_due" in df.columns:
        gap_flag = (
            pd.to_numeric(df["count_gaps_gt_90_pre_due"], errors="coerce")
            .fillna(0)
            .astype(float)
            > 0
        )
    elif "max_gap_pre_due" in df.columns:
        gap_flag = (
            pd.to_numeric(df["max_gap_pre_due"], errors="coerce")
            .fillna(0)
            .astype(float)
            > 90
        )

    def _first_nonnull(series: pd.Series):
        s = series.dropna()
        return s.iloc[0] if len(s) else np.nan

    def _wavg_prob(frame: pd.DataFrame) -> float:
        w = pd.to_numeric(frame.get("TRX_AMOUNT"), errors="coerce").fillna(0).abs()
        p = pd.to_numeric(frame.get("prob_bad_debt"), errors="coerce").fillna(0)
        if float(w.sum()) <= 0:
            return float(p.mean()) if len(p) else np.nan
        return float(np.average(p, weights=w))

    def _paid_ratio_total(frame: pd.DataFrame) -> float:
        if "amt_paid_pre_due" not in frame.columns or "TRX_AMOUNT" not in frame.columns:
            return np.nan
        amt_paid = pd.to_numeric(frame.get("amt_paid_pre_due"), errors="coerce").fillna(
            0
        )
        trx_amt = (
            pd.to_numeric(frame.get("TRX_AMOUNT"), errors="coerce").fillna(0).abs()
        )
        denom = float(trx_amt.sum())
        if denom <= 0:
            return np.nan
        return float((amt_paid.sum() / denom) * 100.0)

    def _gap90_pct(frame: pd.DataFrame) -> float:
        if gap_flag is None:
            return np.nan
        return float(gap_flag.loc[frame.index].mean() * 100.0)

    group = df.groupby(id_cols, dropna=False)
    out = (
        group.apply(
            lambda x: pd.Series(
                {
                    "cust_score_max": float(
                        pd.to_numeric(x["prob_bad_debt"], errors="coerce").max()
                    ),
                    "cust_score_mean": float(
                        pd.to_numeric(x["prob_bad_debt"], errors="coerce").mean()
                    ),
                    "cust_score_wavg_amount": _wavg_prob(x),
                    "invoice_cnt": (
                        int(x["CUSTOMER_TRX_ID"].nunique())
                        if "CUSTOMER_TRX_ID" in x.columns
                        else int(len(x))
                    ),
                    "total_amount": float(
                        pd.to_numeric(x.get("TRX_AMOUNT", 0), errors="coerce").sum()
                    ),
                    "total_amt_paid_pre_due": (
                        float(paid.loc[x.index].sum())
                        if "amt_paid_pre_due" in x.columns
                        else np.nan
                    ),
                    "paid_ratio_pre_due_total": _paid_ratio_total(x),
                    "paid_ratio_pre_due_mean": (
                        float(paid_ratio.loc[x.index].mean())
                        if "paid_ratio_pre_due" in x.columns
                        else np.nan
                    ),
                    "pct_invoices_gap_gt_90_pre_due": _gap90_pct(x),
                    "party_prior_invoice_cnt": (
                        float(
                            _first_nonnull(
                                pd.to_numeric(
                                    x.get("party_prior_invoice_cnt", np.nan),
                                    errors="coerce",
                                )
                            )
                        )
                        if "party_prior_invoice_cnt" in x.columns
                        else np.nan
                    ),
                    "party_prior_gap90_cnt": (
                        float(
                            _first_nonnull(
                                pd.to_numeric(
                                    x.get("party_prior_gap90_cnt", np.nan),
                                    errors="coerce",
                                )
                            )
                        )
                        if "party_prior_gap90_cnt" in x.columns
                        else np.nan
                    ),
                    "party_prior_gap90_rate": (
                        float(
                            _first_nonnull(
                                pd.to_numeric(
                                    x.get("party_prior_gap90_rate", np.nan),
                                    errors="coerce",
                                )
                            )
                        )
                        if "party_prior_gap90_rate" in x.columns
                        else np.nan
                    ),
                }
            ),
            include_groups=False,
        )
    ).reset_index()

    out["risk_cust"] = out["cust_score_max"].apply(_classify_risk)
    return out


def _get_top_efl(df: pd.DataFrame, top_n: int = 50) -> list[dict]:
    """Helper to return the top N invoices by expected financial loss."""
    if "expected_financial_loss" not in df.columns:
        return []
    top_df = df.sort_values("expected_financial_loss", ascending=False).head(top_n)
    return _records(top_df)


def _build_raw_inputs(
    inv_bytes: bytes,
    rcp_bytes: bytes,
    cust_bytes: bytes | None,
    customer_name: str | None,
    customer_format: str | None,
) -> "RawInputFrames":
    """Build RawInputFrames, auto-falling back to OracleCustomer_slim.csv when
    no customer file is uploaded and the slim file exists on disk."""
    cust_io: io.BytesIO | str | None = None
    if cust_bytes is not None:
        cust_io = io.BytesIO(cust_bytes)
    elif DEFAULT_CUSTOMER_CSV.exists():
        # Auto-fallback: use pre-slimmed customer master
        cust_io = str(DEFAULT_CUSTOMER_CSV)
        customer_name = DEFAULT_CUSTOMER_CSV.name
        customer_format = "csv"

    return load_raw_inputs(
        io.BytesIO(inv_bytes),
        io.BytesIO(rcp_bytes),
        cust_io,
        customer_name=customer_name,
        customer_format=customer_format,
    )


# ═══════════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════════


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL_KEY,
        "snapshot_date": DEFAULT_SNAPSHOT_DATE,
        "thresholds": {"low": THRESHOLD_LOW, "high": THRESHOLD_HIGH},
        "models": [_model_public_info(k, v) for k, v in MODEL_REGISTRY.items()],
    }


@app.get("/", response_class=HTMLResponse)
def ui_index():
    """Redirect to the full web UI."""
    return (
        '<!doctype html><html><head><meta http-equiv="refresh" content="0;url=/web/" /></head>'
        '<body><p>Redirecting to <a href="/web/">/web/</a>...</p></body></html>'
    )


@app.post("/score")
async def score(
    invoice_csv: UploadFile = File(...),
    receipt_csv: UploadFile = File(...),
    customer_json: Optional[UploadFile] = File(None),
    model: str = Form(DEFAULT_MODEL_KEY),
    snapshot_date: str = Form(DEFAULT_SNAPSHOT_DATE),
    customer_format: Optional[str] = Form(None),
):
    """Batch scoring - preview JSON."""
    inv_bytes = await _read_upload_with_limit(invoice_csv)
    if isinstance(inv_bytes, JSONResponse):
        return inv_bytes
    rcp_bytes = await _read_upload_with_limit(receipt_csv)
    if isinstance(rcp_bytes, JSONResponse):
        return rcp_bytes
    cust_bytes = None
    if customer_json is not None:
        cust_bytes = await _read_upload_with_limit(customer_json)
        if isinstance(cust_bytes, JSONResponse):
            return cust_bytes

    raw = _build_raw_inputs(
        inv_bytes,
        rcp_bytes,
        cust_bytes,
        customer_name=customer_json.filename if customer_json else None,
        customer_format=customer_format,
    )

    scored = _score_snapshot(raw=raw, snapshot_date=snapshot_date, model_key=model)
    if isinstance(scored, JSONResponse):
        return scored
    out, df_feat, proba, m_info = scored

    customer_risk = _build_customer_risk(df_feat, proba)
    risk_summary = out["risk_level"].value_counts().to_dict()
    customer_risk_summary = (
        customer_risk["risk_cust"].value_counts().to_dict()
        if not customer_risk.empty and "risk_cust" in customer_risk.columns
        else {}
    )
    return {
        "mode": "snapshot",
        "model_key": m_info.get("key", model),
        "model_flow": m_info.get("training_flow"),
        "label_strategy": m_info.get("label_strategy"),
        "snapshot_date": snapshot_date,
        "total_invoices": int(out.shape[0]),
        "risk_summary": risk_summary,
        "high_risk_count": int(risk_summary.get("HIGH", 0)),
        "preview": _records(out.head(20)),
        "top_efl_invoices": _get_top_efl(out, 50),
        "customer_risk_summary": customer_risk_summary,
        "customer_risk": _records(customer_risk),
    }


@app.post("/score_csv")
async def score_csv(
    invoice_csv: UploadFile = File(...),
    receipt_csv: UploadFile = File(...),
    customer_json: Optional[UploadFile] = File(None),
    model: str = Form(DEFAULT_MODEL_KEY),
    snapshot_date: str = Form(DEFAULT_SNAPSHOT_DATE),
    customer_format: Optional[str] = Form(None),
):
    """Batch scoring - download CSV."""
    inv_bytes = await _read_upload_with_limit(invoice_csv)
    if isinstance(inv_bytes, JSONResponse):
        return inv_bytes
    rcp_bytes = await _read_upload_with_limit(receipt_csv)
    if isinstance(rcp_bytes, JSONResponse):
        return rcp_bytes
    cust_bytes = None
    if customer_json is not None:
        cust_bytes = await _read_upload_with_limit(customer_json)
        if isinstance(cust_bytes, JSONResponse):
            return cust_bytes

    raw = _build_raw_inputs(
        inv_bytes,
        rcp_bytes,
        cust_bytes,
        customer_name=customer_json.filename if customer_json else None,
        customer_format=customer_format,
    )

    scored = _score_snapshot(raw=raw, snapshot_date=snapshot_date, model_key=model)
    if isinstance(scored, JSONResponse):
        return scored
    out, _, _, m_info = scored

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    filename = f"bad_debt_snapshot_{snapshot_date}.csv".replace(":", "-")
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/alerts")
async def alerts(
    invoice_csv: UploadFile = File(...),
    receipt_csv: UploadFile = File(...),
    customer_json: Optional[UploadFile] = File(None),
    model: str = Form(DEFAULT_MODEL_KEY),
    snapshot_date: str = Form(DEFAULT_SNAPSHOT_DATE),
    threshold: float = Form(0.3),
    customer_format: Optional[str] = Form(None),
):
    """Return only invoices above the risk threshold (default: MEDIUM+)."""
    inv_bytes = await _read_upload_with_limit(invoice_csv)
    if isinstance(inv_bytes, JSONResponse):
        return inv_bytes
    rcp_bytes = await _read_upload_with_limit(receipt_csv)
    if isinstance(rcp_bytes, JSONResponse):
        return rcp_bytes
    cust_bytes = None
    if customer_json is not None:
        cust_bytes = await _read_upload_with_limit(customer_json)
        if isinstance(cust_bytes, JSONResponse):
            return cust_bytes

    raw = _build_raw_inputs(
        inv_bytes,
        rcp_bytes,
        cust_bytes,
        customer_name=customer_json.filename if customer_json else None,
        customer_format=customer_format,
    )

    scored = _score_snapshot(raw=raw, snapshot_date=snapshot_date, model_key=model)
    if isinstance(scored, JSONResponse):
        return scored
    out, df_feat, proba, m_info = scored

    alerts_df = out[out["prob_bad_debt"] >= threshold].sort_values(
        "prob_bad_debt", ascending=False
    )

    customer_risk = _build_customer_risk(df_feat, proba)
    customer_risk_summary = (
        customer_risk["risk_cust"].value_counts().to_dict()
        if not customer_risk.empty and "risk_cust" in customer_risk.columns
        else {}
    )

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
        "alerts": _records(alerts_df),
        "top_efl_invoices": _get_top_efl(out, 50),
        "customer_risk_summary": customer_risk_summary,
        "customer_risk": _records(customer_risk),
    }


@app.post("/early_warning/receipt_trigger")
async def receipt_trigger(
    invoice_csv: UploadFile = File(...),
    receipt_csv: UploadFile = File(...),
    customer_json: Optional[UploadFile] = File(None),
    model: str = Form(DEFAULT_MODEL_KEY),
    snapshot_date: str = Form(DEFAULT_SNAPSHOT_DATE),
    customer_format: Optional[str] = Form(None),
):
    """
    Early-Warning: Simulasi alur ketika receipt baru masuk.
    Upload invoice & receipt CSV (dan optional customer JSON).
    """
    inv_bytes = await _read_upload_with_limit(invoice_csv)
    if isinstance(inv_bytes, JSONResponse):
        return inv_bytes
    rcp_bytes = await _read_upload_with_limit(receipt_csv)
    if isinstance(rcp_bytes, JSONResponse):
        return rcp_bytes
    cust_bytes = None
    if customer_json is not None:
        cust_bytes = await _read_upload_with_limit(customer_json)
        if isinstance(cust_bytes, JSONResponse):
            return cust_bytes

    raw = _build_raw_inputs(
        inv_bytes,
        rcp_bytes,
        cust_bytes,
        customer_name=customer_json.filename if customer_json else None,
        customer_format=customer_format,
    )

    scored = _score_snapshot(raw=raw, snapshot_date=snapshot_date, model_key=model)
    if isinstance(scored, JSONResponse):
        return scored
    df_result, df_feat, proba, m_info = scored

    feature_cols = _load_schema(m_info["schema_path"])
    alerts_df = df_result[df_result["prob_bad_debt"] >= THRESHOLD_LOW].sort_values(
        "prob_bad_debt", ascending=False
    )
    customer_risk = _build_customer_risk(df_feat, proba)
    customer_risk_summary = (
        customer_risk["risk_cust"].value_counts().to_dict()
        if not customer_risk.empty and "risk_cust" in customer_risk.columns
        else {}
    )

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
        "alerts": _records(alerts_df),
        "all_scores_preview": _records(df_result),
        "top_efl_invoices": _get_top_efl(df_result, 50),
        "customer_risk_summary": customer_risk_summary,
        "customer_risk": _records(customer_risk),
    }


# =========================================================================
# DB-backed Endpoints (no file upload needed)
# =========================================================================


@app.get("/models")
def list_models():
    """Return available time ranges and models for the frontend."""
    from bad_debt_app.db import get_data_date_range

    dates = get_data_date_range()
    return {
        "models": [_model_public_info(k, v) for k, v in MODEL_REGISTRY.items()],
        "time_ranges": [{"key": k, "label": v} for k, v in TIME_RANGE_OPTIONS.items()],
        "min_date": dates.get("min_date"),
        "max_date": dates.get("max_date"),
    }


@app.get("/db/score")
def db_score(
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str = Query(DEFAULT_SNAPSHOT_DATE),
    time_range: str = Query("1w"),
    start_date: str = Query(None),
    end_date: str = Query(None),
):
    """Score invoices from database using the stacked model."""
    try:
        raw = fetch_raw_inputs(
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            snapshot_date=snapshot_date,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"DB error: {e}"})
    if raw.invoice.empty:
        return JSONResponse(
            status_code=200,
            content={"warning": "No invoices found.", "total_invoices": 0},
        )

    scored = _score_snapshot(raw=raw, snapshot_date=snapshot_date, model_key=model)
    if isinstance(scored, JSONResponse):
        return scored
    out, df_feat, proba, m_info = scored
    customer_risk = _build_customer_risk(df_feat, proba)
    risk_summary = out["risk_level"].value_counts().to_dict()
    customer_risk_summary = (
        customer_risk["risk_cust"].value_counts().to_dict()
        if not customer_risk.empty and "risk_cust" in customer_risk.columns
        else {}
    )
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
        "preview": _records(out),
        "top_efl_invoices": _get_top_efl(out, 50),
        "customer_risk_summary": customer_risk_summary,
        "customer_risk": _records(customer_risk),
    }


@app.get("/db/score_csv")
def db_score_csv(
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str = Query(DEFAULT_SNAPSHOT_DATE),
    time_range: str = Query("1w"),
    start_date: str = Query(None),
    end_date: str = Query(None),
):
    """Score invoices from database - CSV download."""
    try:
        raw = fetch_raw_inputs(
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            snapshot_date=snapshot_date,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"DB error: {e}"})
    scored = _score_snapshot(raw=raw, snapshot_date=snapshot_date, model_key=model)
    if isinstance(scored, JSONResponse):
        return scored
    out, _, _, m_info = scored
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    filename = f"bad_debt_{time_range}_{snapshot_date}.csv".replace(":", "-")
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/db/alerts")
def db_alerts(
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str = Query(DEFAULT_SNAPSHOT_DATE),
    time_range: str = Query("1w"),
    threshold: float = Query(0.3),
    start_date: str = Query(None),
    end_date: str = Query(None),
):
    """Return only invoices above risk threshold from database."""
    try:
        raw = fetch_raw_inputs(
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            snapshot_date=snapshot_date,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"DB error: {e}"})
    if raw.invoice.empty:
        return JSONResponse(
            status_code=200, content={"warning": "No invoices.", "alerts_count": 0}
        )
    scored = _score_snapshot(raw=raw, snapshot_date=snapshot_date, model_key=model)
    if isinstance(scored, JSONResponse):
        return scored
    out, df_feat, proba, m_info = scored
    alerts_df = out[out["prob_bad_debt"] >= threshold].sort_values(
        "prob_bad_debt", ascending=False
    )
    customer_risk = _build_customer_risk(df_feat, proba)
    customer_risk_summary = (
        customer_risk["risk_cust"].value_counts().to_dict()
        if not customer_risk.empty and "risk_cust" in customer_risk.columns
        else {}
    )
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
        "alerts": _records(alerts_df),
        "top_efl_invoices": _get_top_efl(out, 50),
        "customer_risk_summary": customer_risk_summary,
        "customer_risk": _records(customer_risk),
    }


@app.get("/db/early_warning/receipt_trigger")
def db_receipt_trigger(
    model: str = Query(DEFAULT_MODEL_KEY),
    snapshot_date: str = Query(DEFAULT_SNAPSHOT_DATE),
    time_range: str = Query("1w"),
    start_date: str = Query(None),
    end_date: str = Query(None),
):
    """Early-Warning receipt trigger analysis from database."""
    try:
        raw = fetch_raw_inputs(
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            snapshot_date=snapshot_date,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"DB error: {e}"})
    if raw.invoice.empty:
        return JSONResponse(
            status_code=200,
            content={"warning": "No invoices.", "processed_invoices": 0},
        )
    scored = _score_snapshot(raw=raw, snapshot_date=snapshot_date, model_key=model)
    if isinstance(scored, JSONResponse):
        return scored
    df_result, df_feat, proba, m_info = scored
    alerts_df = df_result[df_result["prob_bad_debt"] >= THRESHOLD_LOW].sort_values(
        "prob_bad_debt", ascending=False
    )
    customer_risk = _build_customer_risk(df_feat, proba)
    customer_risk_summary = (
        customer_risk["risk_cust"].value_counts().to_dict()
        if not customer_risk.empty and "risk_cust" in customer_risk.columns
        else {}
    )
    feature_cols = _load_schema(m_info["schema_path"])
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
        "alerts": _records(alerts_df),
        "all_scores_preview": _records(df_result),
        "top_efl_invoices": _get_top_efl(df_result, 50),
        "customer_risk_summary": customer_risk_summary,
        "customer_risk": _records(customer_risk),
    }
