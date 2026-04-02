from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from bad_debt_app.feature_engineering.io import RawInputFrames
from bad_debt_app.feature_engineering.base import prepare_base_tables
from bad_debt_app.feature_engineering.history import add_party_history_features
from bad_debt_app.feature_engineering.pre_due import make_features_pre_due


def make_receipt_agg(df_pay_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate receipt/payment data per invoice (gap_variance, is_partial, etc.)."""
    df = df_pay_raw.copy()
    if "INVOICE_ID" in df.columns:
        df = df.rename(columns={"INVOICE_ID": "CUSTOMER_TRX_ID"})

    df["PAYMENT_DATE"] = pd.to_datetime(df.get("PAYMENT_DATE"), errors="coerce")
    df["AMOUNT_APPLIED"] = pd.to_numeric(
        df.get("AMOUNT_APPLIED"), errors="coerce"
    ).fillna(0)
    df = df[df["PAYMENT_DATE"].notna()].copy()
    df = df.sort_values(["CUSTOMER_TRX_ID", "PAYMENT_DATE"])

    df["gap_days"] = df.groupby("CUSTOMER_TRX_ID")["PAYMENT_DATE"].diff().dt.days

    agg = df.groupby("CUSTOMER_TRX_ID", as_index=False).agg(
        receipt_count=("PAYMENT_DATE", "count"),
        first_pay_date=("PAYMENT_DATE", "min"),
        last_pay_date=("PAYMENT_DATE", "max"),
        total_paid=("AMOUNT_APPLIED", "sum"),
        gap_variance=("gap_days", "var"),
    )
    agg["is_partial"] = (agg["receipt_count"] > 1).astype(int)
    return agg


def build_invoice_history(
    df_inv: pd.DataFrame, df_receipt_agg: pd.DataFrame
) -> pd.DataFrame:
    """Build invoice history with payment-derived columns (is_paid, actual_dpd, etc.)."""
    cols = [
        c
        for c in [
            "PARTY_ID",
            "CUSTOMER_TRX_ID",
            "TRX_DATE",
            "DUE_DATE",
            "TRX_AMOUNT",
            "ACCOUNT_NUMBER",
            "CUSTOMER_NAME",
        ]
        if c in df_inv.columns
    ]
    df = df_inv[cols].copy()
    df["TRX_DATE"] = pd.to_datetime(df["TRX_DATE"], errors="coerce")
    df["DUE_DATE"] = pd.to_datetime(df["DUE_DATE"], errors="coerce")
    df["TRX_AMOUNT"] = pd.to_numeric(df["TRX_AMOUNT"], errors="coerce").fillna(0)

    df = df.merge(df_receipt_agg, on="CUSTOMER_TRX_ID", how="left")
    df["actual_dpd"] = (df["last_pay_date"] - df["DUE_DATE"]).dt.days
    df["is_late"] = (df["actual_dpd"] > 0).astype(int)
    df["is_paid"] = df["last_pay_date"].notna().astype(int)

    df["paid_ratio"] = np.where(
        df["TRX_AMOUNT"] > 0,
        (df["total_paid"].fillna(0) / df["TRX_AMOUNT"]).clip(0, 1),
        0,
    )

    df = df.sort_values(["PARTY_ID", "TRX_DATE", "CUSTOMER_TRX_ID"]).reset_index(
        drop=True
    )
    return df


def make_new_model_features(df_inv_full: pd.DataFrame) -> pd.DataFrame:
    """Compute historical features for the new-featured model."""
    _df = df_inv_full.copy()

    if "PARTY_ID" not in _df.columns:
        _df["PARTY_ID"] = -1
    _df["PARTY_ID"] = pd.to_numeric(_df["PARTY_ID"], errors="coerce").fillna(-1)

    _df["_first_trx"] = _df.groupby("PARTY_ID")["TRX_DATE"].transform("first")
    _df["customer_maturity_days"] = (_df["TRX_DATE"] - _df["_first_trx"]).dt.days
    _df.loc[_df.groupby("PARTY_ID").cumcount() == 0, "customer_maturity_days"] = 0

    _df["_cumsum_amt"] = _df.groupby("PARTY_ID")["TRX_AMOUNT"].cumsum()
    _df["business_scale_proxy"] = _df["_cumsum_amt"] - _df["TRX_AMOUNT"]

    _df["_dpd_for_avg"] = _df["actual_dpd"].where(_df["is_paid"] == 1)
    _grp = _df.groupby("PARTY_ID")

    _df["historical_avg_dpd"] = (
        _grp["_dpd_for_avg"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    _df["_late_for_ratio"] = _df["is_late"].where(_df["is_paid"] == 1)
    _df["historical_late_payment_ratio"] = (
        _grp["_late_for_ratio"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    ) * 100

    _df["_unpaid_amt"] = _df["TRX_AMOUNT"].where(_df["is_paid"] == 0, 0)
    _df["_cumsum_unpaid"] = _df.groupby("PARTY_ID")["_unpaid_amt"].cumsum()
    _df["_cumsum_unpaid_shifted"] = (
        _df.groupby("PARTY_ID")["_cumsum_unpaid"].shift(1).fillna(0)
    )
    _df["unpaid_exposure_ratio"] = np.where(
        _df["business_scale_proxy"] > 0,
        _df["_cumsum_unpaid_shifted"] / (_df["business_scale_proxy"] + 1),
        0,
    )

    _gv = _df.get("gap_variance")
    if _gv is not None:
        _df["_gapvar_valid"] = _gv.where(_df["is_paid"] == 1)
    else:
        _df["_gapvar_valid"] = np.nan

    _df["payment_consistency_variance"] = (
        _grp["_gapvar_valid"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    _is_partial = _df.get("is_partial")
    if _is_partial is not None:
        _df["_partial_valid"] = _is_partial.where(_df["is_paid"] == 1).astype(float)
    else:
        _df["_partial_valid"] = np.nan

    _df["historical_partial_payment_freq"] = (
        _grp["_partial_valid"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    ) * 100

    _df["_pr_valid"] = _df["paid_ratio"].where(_df["is_paid"] == 1)
    _df["hist_avg_payment_completion_ratio"] = (
        _grp["_pr_valid"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    _df["_outstanding_amt"] = (_df["TRX_AMOUNT"] - _df["total_paid"].fillna(0)).clip(
        lower=0
    )
    _df["total_outstanding_at_trx"] = (
        _grp["_outstanding_amt"]
        .apply(lambda s: s.shift(1).expanding().sum())
        .reset_index(level=0, drop=True)
    ).fillna(0)

    out_cols = [
        "CUSTOMER_TRX_ID",
        "customer_maturity_days",
        "business_scale_proxy",
        "historical_avg_dpd",
        "historical_late_payment_ratio",
        "unpaid_exposure_ratio",
        "payment_consistency_variance",
        "historical_partial_payment_freq",
        "hist_avg_payment_completion_ratio",
        "total_outstanding_at_trx",
    ]
    return _df[[c for c in out_cols if c in _df.columns]].copy()


def prepare_new_featured_snapshot(
    raw: RawInputFrames,
    snapshot_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare feature table for the new-featured model."""
    df_inv, df_pay_raw = prepare_base_tables(raw, snapshot_date=snapshot_date)
    receipt_agg = make_receipt_agg(df_pay_raw)
    df_inv_full = build_invoice_history(df_inv, receipt_agg)
    df_new_features = make_new_model_features(df_inv_full)

    df_feat = make_features_pre_due(df_inv, df_pay_raw)
    df_feat = add_party_history_features(
        df_feat, df_inv, df_pay_raw, snapshot_date=snapshot_date
    )

    df_feat = df_feat.merge(df_new_features, on="CUSTOMER_TRX_ID", how="left")
    df_feat["TRX_AMOUNT_log1p"] = np.log1p(
        pd.to_numeric(df_feat["TRX_AMOUNT"], errors="coerce").clip(lower=0).fillna(0)
    )

    return df_feat, df_inv
