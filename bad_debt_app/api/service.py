from __future__ import annotations

import io
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

import sklearn

from bad_debt_app.api.config import (
    BASE_DIR,
    DEFAULT_MODEL_KEY,
    MAX_UPLOAD_BYTES,
    MODEL_REGISTRY,
    THRESHOLD_HIGH,
    THRESHOLD_LOW,
)
from bad_debt_app.data.db import fetch_raw_inputs
from bad_debt_app.feature_engineering.pipeline import (
    RawInputFrames,
    load_raw_inputs,
    prepare_snapshot_features,
)

logger = logging.getLogger("bad_debt_api")

# Keep feature names through sklearn pipeline
sklearn.set_config(transform_output="pandas")


def resolve_model(model_key: str | None) -> dict:
    if not model_key or model_key not in MODEL_REGISTRY:
        return MODEL_REGISTRY[DEFAULT_MODEL_KEY]
    return MODEL_REGISTRY[model_key]


def resolve_model_key(model_key: str | None) -> str:
    if not model_key or model_key not in MODEL_REGISTRY:
        return DEFAULT_MODEL_KEY
    return model_key


def model_public_info(model_key: str, model_info: dict) -> dict:
    return {
        "key": model_key,
        "label": model_info.get("label", model_key),
        "training_flow": model_info.get("training_flow"),
        "label_strategy": model_info.get("label_strategy"),
        "credit_memo_policy": model_info.get("credit_memo_policy"),
    }


@lru_cache(maxsize=4)
def load_schema(path: str) -> list[str]:
    with open(path, "r") as f:
        return json.load(f)


@lru_cache(maxsize=2)
def load_model_cached(path: str):
    logger.info("Loading model from %s", Path(path).name)
    return joblib.load(path)


def validate_snapshot_date(date_str: str) -> pd.Timestamp | JSONResponse:
    try:
        return pd.Timestamp(date_str)
    except (ValueError, TypeError):
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Invalid snapshot_date format: '{date_str}'. Use YYYY-MM-DD."
            },
        )


def safe_threshold(threshold: float) -> float:
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="threshold must be between 0.0 and 1.0",
        )
    return threshold


async def read_upload_with_limit(file: UploadFile) -> bytes | JSONResponse:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "error": f"File '{file.filename}' exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit."
            },
        )
    return data


async def read_upload_bundle(
    invoice_csv: UploadFile,
    receipt_csv: UploadFile,
    customer_json: UploadFile | None,
) -> tuple[bytes, bytes, bytes | None] | JSONResponse:
    inv_bytes = await read_upload_with_limit(invoice_csv)
    if isinstance(inv_bytes, JSONResponse):
        return inv_bytes

    rcp_bytes = await read_upload_with_limit(receipt_csv)
    if isinstance(rcp_bytes, JSONResponse):
        return rcp_bytes

    cust_bytes = None
    if customer_json is not None:
        cust_bytes = await read_upload_with_limit(customer_json)
        if isinstance(cust_bytes, JSONResponse):
            return cust_bytes

    return inv_bytes, rcp_bytes, cust_bytes


def build_raw_inputs(
    inv_bytes: bytes,
    rcp_bytes: bytes,
    cust_bytes: bytes | None,
    customer_name: str | None,
    customer_format: str | None,
) -> RawInputFrames:
    cust_io: io.BytesIO | str | None = None
    if cust_bytes is not None:
        cust_io = io.BytesIO(cust_bytes)

    return load_raw_inputs(
        io.BytesIO(inv_bytes),
        io.BytesIO(rcp_bytes),
        cust_io,
        customer_name=customer_name,
        customer_format=customer_format,
    )


def fetch_raw_from_db(
    *, time_range: str, snapshot_date: str, start_date: str | None, end_date: str | None
):
    try:
        return fetch_raw_inputs(
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            snapshot_date=snapshot_date,
        )
    except Exception as exc:
        logger.exception("Failed to fetch data from DB")
        return JSONResponse(status_code=500, content={"error": f"DB error: {exc}"})


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def normalize_customer_name(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    normalized = re.sub(r"[^A-Z0-9]+", "", str(value).upper().strip())
    if normalized.isdigit() and len(normalized) <= 2:
        return ""
    return normalized


@lru_cache(maxsize=1)
def load_excluded_customer_keys() -> set[str]:
    csv_path = BASE_DIR / "artifacts" / "list_all_cust_affiliate_kalla.csv"
    if not csv_path.exists():
        logger.warning("Exclusion list file not found: %s", csv_path)
        return set()

    try:
        raw_df = pd.read_csv(csv_path, header=None, dtype=str, keep_default_na=False)
    except Exception:
        logger.exception("Failed to read exclusion list CSV")
        return set()

    keys: set[str] = set()
    for col in raw_df.columns:
        for item in raw_df[col].tolist():
            key = normalize_customer_name(item)
            if key:
                keys.add(key)
    logger.info("Loaded %d excluded customer names", len(keys))
    return keys


def filter_excluded_customers(
    df: pd.DataFrame, name_col: str = "CUSTOMER_NAME"
) -> pd.DataFrame:
    if df.empty or name_col not in df.columns:
        return df
    excluded_keys = load_excluded_customer_keys()
    if not excluded_keys:
        return df

    normalized = df[name_col].map(normalize_customer_name)
    return df.loc[~normalized.isin(excluded_keys)].copy()


def apply_customer_exclusion(
    df_result: pd.DataFrame,
    df_feat: pd.DataFrame,
    proba: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    filtered_result = filter_excluded_customers(df_result, name_col="CUSTOMER_NAME")
    if filtered_result.shape[0] == df_result.shape[0]:
        return df_result, df_feat, proba

    if (
        "CUSTOMER_TRX_ID" not in df_result.columns
        or "CUSTOMER_TRX_ID" not in df_feat.columns
    ):
        return filtered_result, df_feat, proba

    keep_ids = set(
        pd.to_numeric(filtered_result["CUSTOMER_TRX_ID"], errors="coerce")
        .dropna()
        .astype("int64")
        .tolist()
    )

    feat_ids = pd.to_numeric(df_feat["CUSTOMER_TRX_ID"], errors="coerce")
    feat_mask = feat_ids.isin(keep_ids)
    filtered_feat = df_feat.loc[feat_mask].copy()

    filtered_proba = proba
    if len(proba) == len(df_feat):
        filtered_proba = proba[feat_mask.values].copy()

    return filtered_result, filtered_feat, filtered_proba


def classify_risk(prob: float) -> str:
    if prob >= THRESHOLD_HIGH:
        return "HIGH"
    if prob >= THRESHOLD_LOW:
        return "MEDIUM"
    return "LOW"


def recommend_action(risk: str) -> str:
    actions = {
        "HIGH": "ESKALASI - Follow-up tim collection segera",
        "MEDIUM": "WATCHLIST - Monitor & kirim reminder pembayaran",
        "LOW": "OK - Monitoring rutin",
    }
    return actions.get(risk, "Unknown")


def score_snapshot(
    *, raw, snapshot_date: str, model_key: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict] | JSONResponse:
    snap_result = validate_snapshot_date(snapshot_date)
    if isinstance(snap_result, JSONResponse):
        return snap_result
    snap = snap_result

    resolved_model_key = resolve_model_key(model_key)
    model_info = dict(resolve_model(resolved_model_key))
    model_info["key"] = resolved_model_key
    model_path = model_info["model_path"]
    schema_path = model_info["schema_path"]

    df_feat, _ = prepare_snapshot_features(raw, snapshot_date=snap)
    feature_cols = load_schema(schema_path)

    for c in feature_cols:
        if c not in df_feat.columns:
            df_feat[c] = np.nan

    X = df_feat[feature_cols].copy().fillna(value=np.nan).infer_objects(copy=False)

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
            model_obj = load_model_cached(model_path)
            if isinstance(model_obj, dict) and "meta_model" in model_obj:
                lgb_b = model_obj["lgb_model_B"]
                lr_m = model_obj["lr_model"]
                meta = model_obj["meta_model"]
                prep = model_obj.get("preprocessor")

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

                oof_lgb = lgb_b.predict_proba(X)[:, 1]
                oof_lr = lr_m.predict_proba(np.asarray(X_prep))[:, 1]
                X_meta = pd.DataFrame({"oof_lgb_B": oof_lgb, "oof_lr": oof_lr})
                raw_proba = meta.predict_proba(np.asarray(X_meta))[:, 1]

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
                proba = model_obj.predict_proba(X)[:, 1]

    except FileNotFoundError:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Model file not found.",
                "hint": "Pastikan model sudah di-train dan disimpan ke artifacts/",
            },
        )
    except ModuleNotFoundError:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to load model due to missing dependency.",
                "hint": "pip install lightgbm / tensorflow",
            },
        )
    except Exception:
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
    if "SBU" in df_feat.columns and df_feat["SBU"].notna().any():
        cols["SBU"] = df_feat["SBU"]
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
    cols["risk_level"] = [classify_risk(p) for p in proba]
    cols["recommended_action"] = [recommend_action(classify_risk(p)) for p in proba]

    df_result = pd.DataFrame(cols)

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


def build_customer_risk(df_feat: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
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

    out["risk_cust"] = out["cust_score_max"].apply(classify_risk)
    return out


def build_customer_risk_summary(customer_risk: pd.DataFrame) -> dict:
    if customer_risk.empty or "risk_cust" not in customer_risk.columns:
        return {}
    return customer_risk["risk_cust"].value_counts().to_dict()


def get_top_efl(df: pd.DataFrame, top_n: int = 50) -> list[dict]:
    if "expected_financial_loss" not in df.columns:
        return []
    top_df = df.sort_values("expected_financial_loss", ascending=False).head(top_n)
    return records(top_df)
