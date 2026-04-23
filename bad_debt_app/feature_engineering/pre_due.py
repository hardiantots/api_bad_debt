from __future__ import annotations

import numpy as np
import pandas as pd


def make_features_pre_due(
    df_inv: pd.DataFrame, df_pay_raw: pd.DataFrame
) -> pd.DataFrame:
    """Build pre-due features using payments within [TRX_DATE, AS_OF_DATE]."""

    inv_cols = [
        c
        for c in [
            "CUSTOMER_TRX_ID",
            "TRX_DATE",
            "DUE_DATE",
            "AS_OF_DATE",
            "TRX_AMOUNT",
            "PARTY_ID",
            "ACCOUNT_NUMBER",
            "CUSTOMER_NAME",
            "SBU",
            "cust_master_missing",
            "TRANS_TYPE",
            "CURRENCY_CODE",
            "ORG_ID",
            "TRX_AMOUNT_GROSS",
            "credit_memo_reduction",
        ]
        if c in df_inv.columns
    ]

    base = df_inv[inv_cols].copy()

    df_pay = df_pay_raw.merge(
        base[["CUSTOMER_TRX_ID", "TRX_DATE", "DUE_DATE", "AS_OF_DATE", "TRX_AMOUNT"]],
        left_on="INVOICE_ID",
        right_on="CUSTOMER_TRX_ID",
        how="inner",
    )

    df_win = df_pay[
        df_pay["PAYMENT_DATE"].notna()
        & (df_pay["PAYMENT_DATE"] >= df_pay["TRX_DATE"])
        & (df_pay["PAYMENT_DATE"] <= df_pay["AS_OF_DATE"])
    ].copy()

    df_agg = (
        df_win.sort_values(["CUSTOMER_TRX_ID", "PAYMENT_DATE"])
        .groupby("CUSTOMER_TRX_ID", as_index=False)
        .agg(
            n_pay_pre_due=("PAYMENT_DATE", "count"),
            amt_paid_pre_due=("AMOUNT_APPLIED", "sum"),
            first_pay_pre_due=("PAYMENT_DATE", "min"),
            last_pay_pre_due=("PAYMENT_DATE", "max"),
        )
    )

    df_win = df_win.sort_values(["CUSTOMER_TRX_ID", "PAYMENT_DATE"])
    df_win["gap_days"] = (
        df_win.groupby("CUSTOMER_TRX_ID")["PAYMENT_DATE"].diff().dt.days
    )
    df_win["gap_gt_90_flag"] = (df_win["gap_days"] > 90).fillna(False).astype(int)

    df_gap = df_win.groupby("CUSTOMER_TRX_ID", as_index=False).agg(
        max_gap_pre_due=("gap_days", "max"),
        count_gaps_gt_90_pre_due=("gap_gt_90_flag", "sum"),
    )

    out = base.merge(df_agg, on="CUSTOMER_TRX_ID", how="left").merge(
        df_gap, on="CUSTOMER_TRX_ID", how="left"
    )

    out["n_pay_pre_due"] = out["n_pay_pre_due"].fillna(0).astype(int)
    out["amt_paid_pre_due"] = out["amt_paid_pre_due"].fillna(0.0)
    out["paid_ratio_pre_due"] = np.where(
        out["TRX_AMOUNT"].abs() > 0,
        (out["amt_paid_pre_due"] / out["TRX_AMOUNT"].abs()) * 100,
        0.0,
    )
    out["paid_ratio_pre_due"] = pd.Series(
        out["paid_ratio_pre_due"], index=out.index
    ).clip(0, 100)

    out["days_trx_to_first_pay_pre_due"] = (
        out["first_pay_pre_due"] - out["TRX_DATE"]
    ).dt.days
    out["days_before_due_last_pay"] = (
        out["DUE_DATE"] - out["last_pay_pre_due"]
    ).dt.days

    out["max_gap_pre_due"] = out["max_gap_pre_due"].fillna(0.0)
    out["count_gaps_gt_90_pre_due"] = (
        out["count_gaps_gt_90_pre_due"].fillna(0).astype(int)
    )

    out["days_to_due"] = (out["DUE_DATE"] - out["TRX_DATE"]).dt.days
    out["trx_month"] = out["TRX_DATE"].dt.month
    out["trx_weekday"] = out["TRX_DATE"].dt.weekday

    return out


def make_features_asof(
    df_inv_base: pd.DataFrame, df_pay_base: pd.DataFrame, asof_col: str, prefix: str
) -> pd.DataFrame:
    """General AS-OF feature builder (ported from notebook)."""

    inv_cols = [
        "CUSTOMER_TRX_ID",
        "TRX_DATE",
        "DUE_DATE",
        "TRX_AMOUNT",
        "TRANS_TYPE",
        "CURRENCY_CODE",
        "ORG_ID",
        asof_col,
    ]
    inv_cols = [c for c in inv_cols if c in df_inv_base.columns]

    df_inv_tmp = df_inv_base[inv_cols].copy()
    df_inv_tmp["TRX_DATE"] = pd.to_datetime(df_inv_tmp["TRX_DATE"], errors="coerce")
    df_inv_tmp["DUE_DATE"] = pd.to_datetime(df_inv_tmp["DUE_DATE"], errors="coerce")
    df_inv_tmp[asof_col] = pd.to_datetime(df_inv_tmp[asof_col], errors="coerce")

    df_pay_tmp = df_pay_base.copy()
    df_pay_tmp = df_pay_tmp.merge(
        df_inv_tmp[["CUSTOMER_TRX_ID", "TRX_DATE", "DUE_DATE", "TRX_AMOUNT", asof_col]],
        left_on="INVOICE_ID",
        right_on="CUSTOMER_TRX_ID",
        how="inner",
    )

    df_win = df_pay_tmp[
        df_pay_tmp["PAYMENT_DATE"].notna()
        & (df_pay_tmp["PAYMENT_DATE"] >= df_pay_tmp["TRX_DATE"])
        & (df_pay_tmp["PAYMENT_DATE"] <= df_pay_tmp[asof_col])
    ].copy()

    df_agg = (
        df_win.sort_values(["CUSTOMER_TRX_ID", "PAYMENT_DATE"])
        .groupby("CUSTOMER_TRX_ID", as_index=False)
        .agg(
            **{
                f"n_pay_{prefix}": ("PAYMENT_DATE", "count"),
                f"amt_paid_{prefix}": ("AMOUNT_APPLIED", "sum"),
                f"first_pay_{prefix}": ("PAYMENT_DATE", "min"),
                f"last_pay_{prefix}": ("PAYMENT_DATE", "max"),
            }
        )
    )

    df_win = df_win.sort_values(["CUSTOMER_TRX_ID", "PAYMENT_DATE"])
    df_win[f"gap_days_{prefix}"] = (
        df_win.groupby("CUSTOMER_TRX_ID")["PAYMENT_DATE"].diff().dt.days
    )
    df_win[f"gap_gt_90_flag_{prefix}"] = (
        (df_win[f"gap_days_{prefix}"] > 90).fillna(False).astype(int)
    )

    df_gap = df_win.groupby("CUSTOMER_TRX_ID", as_index=False).agg(
        **{
            f"max_gap_{prefix}": (f"gap_days_{prefix}", "max"),
            f"count_gaps_gt_90_{prefix}": (f"gap_gt_90_flag_{prefix}", "sum"),
        }
    )

    out = df_inv_tmp.merge(df_agg, on="CUSTOMER_TRX_ID", how="left").merge(
        df_gap, on="CUSTOMER_TRX_ID", how="left"
    )

    out[f"n_pay_{prefix}"] = out[f"n_pay_{prefix}"].fillna(0).astype(int)
    out[f"amt_paid_{prefix}"] = out[f"amt_paid_{prefix}"].fillna(0.0)
    out[f"max_gap_{prefix}"] = out[f"max_gap_{prefix}"].fillna(0.0)
    out[f"count_gaps_gt_90_{prefix}"] = (
        out[f"count_gaps_gt_90_{prefix}"].fillna(0).astype(int)
    )

    out[f"paid_ratio_{prefix}"] = (
        (out[f"amt_paid_{prefix}"] / out["TRX_AMOUNT"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    out[f"days_trx_to_first_pay_{prefix}"] = (
        out[f"first_pay_{prefix}"] - out["TRX_DATE"]
    ).dt.days

    out[f"days_since_last_pay_{prefix}"] = (
        out[asof_col] - out[f"last_pay_{prefix}"]
    ).dt.days
    out.loc[out[f"last_pay_{prefix}"].isna(), f"days_since_last_pay_{prefix}"] = (
        out.loc[out[f"last_pay_{prefix}"].isna(), asof_col]
        - out.loc[out[f"last_pay_{prefix}"].isna(), "TRX_DATE"]
    ).dt.days

    reached_90 = out[asof_col] >= (out["TRX_DATE"] + pd.Timedelta(days=90))
    out[f"no_payment_by_90_{prefix}"] = (
        reached_90 & out[f"first_pay_{prefix}"].isna()
    ).astype(int)
    out[f"first_pay_gt_90_{prefix}"] = (
        (out[f"days_trx_to_first_pay_{prefix}"] > 90).fillna(False).astype(int)
    )

    out["days_to_due"] = (out["DUE_DATE"] - out["TRX_DATE"]).dt.days
    out["trx_month"] = out["TRX_DATE"].dt.month
    out["trx_weekday"] = out["TRX_DATE"].dt.weekday

    return out
