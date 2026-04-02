from __future__ import annotations

import numpy as np
import pandas as pd

from bad_debt_app.feature_engineering.io import _to_dt_naive


def _party_counts_before(
    targets: pd.DataFrame,
    events: pd.DataFrame,
    event_date_col: str,
    flag_col: str,
    out_col: str,
) -> pd.Series:
    """Count per PARTY_ID number of events with event_date < TRX_DATE of target invoice."""

    res = pd.Series(0, index=targets.index, dtype="int64", name=out_col)

    if events.empty or "PARTY_ID" not in events.columns:
        return res

    ev = events[["PARTY_ID", event_date_col, flag_col]].copy()
    ev["PARTY_ID"] = pd.to_numeric(ev["PARTY_ID"], errors="coerce").astype("Int64")
    ev[event_date_col] = _to_dt_naive(ev[event_date_col])
    ev = ev[
        (ev["PARTY_ID"].notna()) & (ev[flag_col] == 1) & ev[event_date_col].notna()
    ].copy()
    if ev.empty:
        return res

    ev = ev.sort_values(["PARTY_ID", event_date_col])
    ev_dict = {
        int(pid): grp[event_date_col].to_numpy(dtype="datetime64[ns]")
        for pid, grp in ev.groupby("PARTY_ID")
    }

    tgt = targets[["PARTY_ID", "TRX_DATE"]].copy()
    tgt["PARTY_ID"] = pd.to_numeric(tgt["PARTY_ID"], errors="coerce").astype("Int64")
    tgt["TRX_DATE"] = _to_dt_naive(tgt["TRX_DATE"])
    tgt = tgt[(tgt["PARTY_ID"].notna()) & tgt["TRX_DATE"].notna()].copy()
    if tgt.empty:
        return res

    for pid, g in tgt.groupby("PARTY_ID", sort=False):
        arr = ev_dict.get(int(pid))
        if arr is None or len(arr) == 0:
            continue
        td = g["TRX_DATE"].to_numpy(dtype="datetime64[ns]")
        res.loc[g.index] = np.searchsorted(arr, td, side="left")

    return res


def add_party_history_features(
    df_feat: pd.DataFrame,
    df_inv: pd.DataFrame,
    df_pay_raw: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    eps: float = 0.01,
) -> pd.DataFrame:
    """Attach PARTY/customer history signals (anti-leakage)."""

    out = df_feat.copy()

    if "PARTY_ID" not in out.columns:
        out["PARTY_ID"] = np.nan

    out["party_missing"] = out["PARTY_ID"].isna().astype(int)

    inv_hist = df_inv[
        [
            c
            for c in [
                "CUSTOMER_TRX_ID",
                "PARTY_ID",
                "TRX_DATE",
                "DUE_DATE",
                "TRX_AMOUNT",
            ]
            if c in df_inv.columns
        ]
    ].copy()
    inv_hist = inv_hist[
        inv_hist["PARTY_ID"].notna() & inv_hist["TRX_DATE"].notna()
    ].copy()
    inv_hist = inv_hist.sort_values(["PARTY_ID", "TRX_DATE", "CUSTOMER_TRX_ID"])
    inv_hist["party_prior_invoice_cnt"] = inv_hist.groupby("PARTY_ID").cumcount()

    prior_cnt_map = inv_hist.set_index("CUSTOMER_TRX_ID")["party_prior_invoice_cnt"]
    out["party_prior_invoice_cnt"] = (
        out["CUSTOMER_TRX_ID"].map(prior_cnt_map).fillna(0).astype(int)
    )
    out["party_is_new_customer"] = (
        (out["party_missing"] == 0) & (out["party_prior_invoice_cnt"] == 0)
    ).astype(int)

    pay_hist = df_pay_raw.copy()
    pay_hist = pay_hist.rename(columns={"INVOICE_ID": "CUSTOMER_TRX_ID"})
    pay_hist["PAYMENT_DATE"] = _to_dt_naive(pay_hist["PAYMENT_DATE"])
    pay_hist = pay_hist[
        pay_hist["PAYMENT_DATE"].notna() & (pay_hist["PAYMENT_DATE"] <= snapshot_date)
    ].copy()
    pay_hist["AMOUNT_APPLIED"] = pd.to_numeric(
        pay_hist["AMOUNT_APPLIED"], errors="coerce"
    ).fillna(0.0)

    inv_bd = inv_hist.copy()
    inv_bd["bd90_event_date"] = inv_bd["DUE_DATE"] + pd.Timedelta(days=90)
    inv_bd = inv_bd[
        inv_bd["bd90_event_date"].notna() & (inv_bd["bd90_event_date"] <= snapshot_date)
    ].copy()

    if not inv_bd.empty and not pay_hist.empty:
        pay_bd = pay_hist.merge(
            inv_bd[["CUSTOMER_TRX_ID", "bd90_event_date"]],
            on="CUSTOMER_TRX_ID",
            how="inner",
        )
        pay_bd = pay_bd[pay_bd["PAYMENT_DATE"] <= pay_bd["bd90_event_date"]].copy()
        paid_to_bd = (
            pay_bd.groupby("CUSTOMER_TRX_ID", as_index=False)["AMOUNT_APPLIED"]
            .sum()
            .rename(columns={"AMOUNT_APPLIED": "amt_paid_to_bd90"})
        )
        inv_bd = inv_bd.merge(paid_to_bd, on="CUSTOMER_TRX_ID", how="left")
        inv_bd["amt_paid_to_bd90"] = inv_bd["amt_paid_to_bd90"].fillna(0.0)
    else:
        inv_bd["amt_paid_to_bd90"] = 0.0

    inv_bd["TRX_AMOUNT"] = pd.to_numeric(inv_bd["TRX_AMOUNT"], errors="coerce")
    inv_bd["bd90_flag"] = (
        (inv_bd["TRX_AMOUNT"] - inv_bd["amt_paid_to_bd90"]) > eps
    ).astype(int)

    if not pay_hist.empty:
        first_pay = (
            pay_hist.groupby("CUSTOMER_TRX_ID", as_index=False)["PAYMENT_DATE"]
            .min()
            .rename(columns={"PAYMENT_DATE": "first_pay_date"})
        )
    else:
        first_pay = pd.DataFrame(columns=["CUSTOMER_TRX_ID", "first_pay_date"])

    inv_fp = inv_hist.merge(first_pay, on="CUSTOMER_TRX_ID", how="left")
    inv_fp["days_trx_to_first_pay"] = (
        inv_fp["first_pay_date"] - inv_fp["TRX_DATE"]
    ).dt.days
    inv_fp["late_firstpay90_flag"] = (
        (inv_fp["days_trx_to_first_pay"] > 90).fillna(False).astype(int)
    )

    if not pay_hist.empty:
        pay_sorted = pay_hist.sort_values(["CUSTOMER_TRX_ID", "PAYMENT_DATE"]).copy()
        pay_sorted["gap_days"] = (
            pay_sorted.groupby("CUSTOMER_TRX_ID")["PAYMENT_DATE"].diff().dt.days
        )
        gap90 = (
            pay_sorted[pay_sorted["gap_days"] > 90]
            .groupby("CUSTOMER_TRX_ID", as_index=False)["PAYMENT_DATE"]
            .min()
            .rename(columns={"PAYMENT_DATE": "gap90_event_date"})
        )
    else:
        gap90 = pd.DataFrame(columns=["CUSTOMER_TRX_ID", "gap90_event_date"])

    inv_gap = inv_hist.merge(gap90, on="CUSTOMER_TRX_ID", how="left")
    inv_gap["gap90_flag"] = inv_gap["gap90_event_date"].notna().astype(int)

    out = out.sort_values(["CUSTOMER_TRX_ID"]).copy()
    out["party_prior_bd90_cnt"] = _party_counts_before(
        out, inv_bd, "bd90_event_date", "bd90_flag", "party_prior_bd90_cnt"
    )
    out["party_prior_late_firstpay90_cnt"] = _party_counts_before(
        out,
        inv_fp,
        "first_pay_date",
        "late_firstpay90_flag",
        "party_prior_late_firstpay90_cnt",
    )
    out["party_prior_gap90_cnt"] = _party_counts_before(
        out, inv_gap, "gap90_event_date", "gap90_flag", "party_prior_gap90_cnt"
    )

    for c in [
        "party_prior_bd90_cnt",
        "party_prior_late_firstpay90_cnt",
        "party_prior_gap90_cnt",
    ]:
        out[c] = out[c].fillna(0).astype(int)

    _pay = df_pay_raw.copy()
    _pay = _pay.rename(columns={"INVOICE_ID": "CUSTOMER_TRX_ID"})
    _pay["PAYMENT_DATE"] = pd.to_datetime(_pay["PAYMENT_DATE"], errors="coerce")
    _pay["AMOUNT_APPLIED"] = pd.to_numeric(
        _pay["AMOUNT_APPLIED"], errors="coerce"
    ).fillna(0.0)
    _pay = _pay[_pay["PAYMENT_DATE"].notna()].copy()
    _pay = _pay.sort_values(["CUSTOMER_TRX_ID", "PAYMENT_DATE"])
    _pay["gap_days"] = _pay.groupby("CUSTOMER_TRX_ID")["PAYMENT_DATE"].diff().dt.days

    _receipt_agg = _pay.groupby("CUSTOMER_TRX_ID", as_index=False).agg(
        receipt_count=("PAYMENT_DATE", "count"),
        first_pay_date=("PAYMENT_DATE", "min"),
        last_pay_date=("PAYMENT_DATE", "max"),
        total_paid=("AMOUNT_APPLIED", "sum"),
        gap_variance=("gap_days", "var"),
    )
    _receipt_agg["is_partial"] = (_receipt_agg["receipt_count"] > 1).astype(int)

    _inv_cols = [
        c
        for c in ["PARTY_ID", "CUSTOMER_TRX_ID", "TRX_DATE", "DUE_DATE", "TRX_AMOUNT"]
        if c in df_inv.columns
    ]
    _df_inv_full = df_inv[_inv_cols].copy()
    _df_inv_full["TRX_DATE"] = pd.to_datetime(_df_inv_full["TRX_DATE"], errors="coerce")
    _df_inv_full["DUE_DATE"] = pd.to_datetime(_df_inv_full["DUE_DATE"], errors="coerce")
    _df_inv_full["TRX_AMOUNT"] = pd.to_numeric(
        _df_inv_full["TRX_AMOUNT"], errors="coerce"
    ).fillna(0)

    _df_inv_full = _df_inv_full.merge(_receipt_agg, on="CUSTOMER_TRX_ID", how="left")
    _df_inv_full["actual_dpd"] = (
        _df_inv_full["last_pay_date"] - _df_inv_full["DUE_DATE"]
    ).dt.days
    _df_inv_full["is_late"] = (_df_inv_full["actual_dpd"] > 0).astype(int)
    _df_inv_full["is_paid"] = _df_inv_full["last_pay_date"].notna().astype(int)
    _df_inv_full["paid_ratio"] = np.where(
        _df_inv_full["TRX_AMOUNT"] > 0,
        (_df_inv_full["total_paid"].fillna(0) / _df_inv_full["TRX_AMOUNT"]).clip(0, 1),
        0,
    )

    _df_inv_full = _df_inv_full.sort_values(
        ["PARTY_ID", "TRX_DATE", "CUSTOMER_TRX_ID"]
    ).reset_index(drop=True)

    _df = _df_inv_full.copy()
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
    _df["_cumsum_amt_shifted"] = _df["business_scale_proxy"]
    _df["unpaid_exposure_ratio"] = np.where(
        _df["_cumsum_amt_shifted"] > 0,
        _df["_cumsum_unpaid_shifted"] / (_df["_cumsum_amt_shifted"] + 1),
        0,
    )

    _df["_gapvar_valid"] = _df["gap_variance"].where(_df["is_paid"] == 1)
    _df["payment_consistency_variance"] = (
        _grp["_gapvar_valid"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    _df["_partial_valid"] = _df["is_partial"].where(_df["is_paid"] == 1).astype(float)
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

    _surv = _df_inv_full[
        [
            "PARTY_ID",
            "CUSTOMER_TRX_ID",
            "TRX_DATE",
            "DUE_DATE",
            "actual_dpd",
            "is_paid",
            "is_late",
        ]
    ].copy()
    _surv = _surv.sort_values(["PARTY_ID", "TRX_DATE", "CUSTOMER_TRX_ID"]).reset_index(
        drop=True
    )

    _surv["paid_within_30d"] = (
        (_surv["actual_dpd"] <= 30) & (_surv["is_paid"] == 1)
    ).astype(float)
    _surv["paid_within_60d"] = (
        (_surv["actual_dpd"] <= 60) & (_surv["is_paid"] == 1)
    ).astype(float)
    _surv["bad_debt_flag"] = (
        (_surv["actual_dpd"] > 90) | (_surv["is_paid"] == 0)
    ).astype(float)

    _grp_surv = _surv.groupby("PARTY_ID")

    _surv["prob_paid_within_30d"] = (
        _grp_surv["paid_within_30d"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    ).fillna(0.5)

    _surv["prob_paid_within_60d"] = (
        _grp_surv["paid_within_60d"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    ).fillna(0.5)

    _surv["_dpd_paid"] = _surv["actual_dpd"].where(_surv["is_paid"] == 1)
    _surv["median_payment_delay"] = (
        _grp_surv["_dpd_paid"]
        .apply(lambda s: s.shift(1).expanding().median())
        .reset_index(level=0, drop=True)
    ).fillna(0)

    _surv["hazard_approx"] = (
        _grp_surv["bad_debt_flag"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    ).fillna(0)

    _hist_cols = [
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
    _df_hist = _df[_hist_cols].drop_duplicates("CUSTOMER_TRX_ID")

    _surv_cols = [
        "CUSTOMER_TRX_ID",
        "prob_paid_within_30d",
        "prob_paid_within_60d",
        "median_payment_delay",
        "hazard_approx",
    ]
    _df_surv = _surv[_surv_cols].drop_duplicates("CUSTOMER_TRX_ID")

    out = out.merge(_df_hist, on="CUSTOMER_TRX_ID", how="left")
    out = out.merge(_df_surv, on="CUSTOMER_TRX_ID", how="left")

    _cold_start = {
        "customer_maturity_days": 0,
        "business_scale_proxy": 0.0,
        "historical_avg_dpd": 0.0,
        "historical_late_payment_ratio": 0.0,
        "unpaid_exposure_ratio": 0.0,
        "payment_consistency_variance": 0.0,
        "historical_partial_payment_freq": 0.0,
        "hist_avg_payment_completion_ratio": 0.0,
        "total_outstanding_at_trx": 0.0,
        "prob_paid_within_30d": 0.5,
        "prob_paid_within_60d": 0.5,
        "median_payment_delay": 0,
        "hazard_approx": 0,
    }
    out.fillna(value=_cold_start, inplace=True)

    out["TRX_AMOUNT_log1p"] = np.log1p(
        pd.to_numeric(out["TRX_AMOUNT"], errors="coerce").fillna(0).clip(lower=0)
    )
    out["business_scale_proxy"] = np.log1p(out["business_scale_proxy"].clip(lower=0))

    return out
