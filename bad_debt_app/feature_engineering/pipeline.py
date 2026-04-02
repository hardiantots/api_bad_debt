from __future__ import annotations

from typing import Tuple

import pandas as pd

from bad_debt_app.feature_engineering.io import (
    RawInputFrames,
    _to_dt_naive,
    fix_year_month,
    load_raw_inputs,
)
from bad_debt_app.feature_engineering.base import (
    apply_credit_memo_netting,
    prepare_base_tables,
)
from bad_debt_app.feature_engineering.history import add_party_history_features
from bad_debt_app.feature_engineering.new_model import (
    build_invoice_history,
    make_new_model_features,
    make_receipt_agg,
    prepare_new_featured_snapshot,
)
from bad_debt_app.feature_engineering.pre_due import (
    make_features_asof,
    make_features_pre_due,
)

EPS_DEFAULT = 0.01


def prepare_snapshot_features(
    raw: RawInputFrames,
    snapshot_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare snapshot/pre-due feature table and return (features_df, df_inv)."""

    df_inv, df_pay_raw = prepare_base_tables(raw, snapshot_date=snapshot_date)
    df_feat = make_features_pre_due(df_inv, df_pay_raw)
    df_feat = add_party_history_features(
        df_feat, df_inv, df_pay_raw, snapshot_date=snapshot_date
    )

    return df_feat, df_inv


def prepare_monitoring_features(
    raw: RawInputFrames,
    snapshot_date: pd.Timestamp,
    monitor_observe_min_days: int = 180,
    monitor_asof_to_snapshot: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare monitoring stage-2 feature table and return (features_df, df_inv)."""

    df_inv, df_pay_raw = prepare_base_tables(raw, snapshot_date=snapshot_date)

    df_inv_mon = df_inv.copy()
    if monitor_asof_to_snapshot:
        df_inv_mon["AS_OF_DATE_MON"] = snapshot_date
    else:
        df_inv_mon["AS_OF_DATE_MON"] = df_inv_mon["TRX_DATE"] + pd.Timedelta(
            days=monitor_observe_min_days
        )
        df_inv_mon.loc[
            df_inv_mon["AS_OF_DATE_MON"] > snapshot_date, "AS_OF_DATE_MON"
        ] = snapshot_date

    df_feat = make_features_asof(
        df_inv_mon, df_pay_raw, asof_col="AS_OF_DATE_MON", prefix="mon"
    )

    attach_cols = [
        c
        for c in [
            "CUSTOMER_TRX_ID",
            "PARTY_ID",
            "ACCOUNT_NUMBER",
            "CUSTOMER_NAME",
            "cust_master_missing",
            "TRX_AMOUNT_GROSS",
            "credit_memo_reduction",
        ]
        if c in df_inv.columns
    ]
    df_feat = df_feat.merge(df_inv[attach_cols], on="CUSTOMER_TRX_ID", how="left")

    df_feat = add_party_history_features(
        df_feat, df_inv, df_pay_raw, snapshot_date=snapshot_date
    )

    trx_dt = (
        _to_dt_naive(df_feat["TRX_DATE"])
        if "TRX_DATE" in df_feat.columns
        else pd.Series(pd.NaT, index=df_feat.index)
    )
    df_feat["mon_observable"] = trx_dt <= (
        snapshot_date - pd.Timedelta(days=monitor_observe_min_days)
    )

    return df_feat, df_inv


__all__ = [
    "RawInputFrames",
    "EPS_DEFAULT",
    "load_raw_inputs",
    "_to_dt_naive",
    "fix_year_month",
    "apply_credit_memo_netting",
    "prepare_base_tables",
    "make_features_pre_due",
    "make_features_asof",
    "add_party_history_features",
    "prepare_snapshot_features",
    "prepare_monitoring_features",
    "make_receipt_agg",
    "build_invoice_history",
    "make_new_model_features",
    "prepare_new_featured_snapshot",
]
