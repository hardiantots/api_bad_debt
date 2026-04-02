from __future__ import annotations

from dataclasses import dataclass
from typing import IO, Any, Optional, Tuple

import numpy as np
import pandas as pd


EPS_DEFAULT = 0.01


@dataclass(frozen=True)
class RawInputFrames:
    invoice: pd.DataFrame
    receipt: pd.DataFrame
    customer: Optional[pd.DataFrame] = None
    target_trx_ids: Optional[list] = None


def _to_dt_naive(s: pd.Series) -> pd.Series:
    # Always return tz-naive datetime64[ns]
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)


def fix_year_month(s: pd.Series, snapshot_date: pd.Timestamp) -> pd.Series:
    """Fix common future-date anomalies (ported from notebook logic)."""
    s = _to_dt_naive(s)

    y = s.dt.year
    m = s.dt.month
    d = s.dt.day

    # 22xx -> 20xx
    y = np.where((y >= 2200) & (y < 2300), y - 200, y)

    # future within next ~20 years often means off-by-10
    y = np.where((y > snapshot_date.year) & (y <= snapshot_date.year + 20), y - 10, y)

    out = pd.to_datetime({"year": y, "month": m, "day": d}, errors="coerce")

    # If it's just slightly in the future (<=45d), shift one month back
    mask = (
        out.notna()
        & (out > snapshot_date)
        & (out <= snapshot_date + pd.Timedelta(days=45))
    )
    out.loc[mask] = out.loc[mask] - pd.DateOffset(months=1)

    return out


def _infer_format(source_name: str | None) -> str | None:
    if not source_name:
        return None
    name = source_name.lower()
    if name.endswith(".parquet") or name.endswith(".pq"):
        return "parquet"
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".json"):
        return "json"
    if name.endswith(".jsonl"):
        return "jsonl"
    return None


def _read_customer(
    customer_input: IO[bytes] | str,
    *,
    customer_name: str | None = None,
    customer_format: str | None = None,
) -> pd.DataFrame:
    fmt = customer_format or _infer_format(customer_name)

    if fmt == "csv":
        return pd.read_csv(customer_input, low_memory=False)
    if fmt == "parquet":
        try:
            return pd.read_parquet(customer_input)
        except Exception as exc:  # pragma: no cover - depends on pyarrow/fastparquet
            raise ValueError(
                "Failed to read parquet customer file. Install pyarrow or use CSV/JSON instead."
            ) from exc
    if fmt == "jsonl":
        return pd.read_json(customer_input, lines=True)
    # default json
    return pd.read_json(customer_input)


def load_raw_inputs(
    invoice_csv: IO[bytes] | str,
    receipt_csv: IO[bytes] | str,
    customer_json: IO[bytes] | str | None = None,
    *,
    customer_name: str | None = None,
    customer_format: str | None = None,
) -> RawInputFrames:
    """Load raw invoice/receipt/customer inputs into DataFrames."""
    # low_memory=False avoids dtype inference issues on wide CSVs
    df_invoice = pd.read_csv(invoice_csv, low_memory=False)
    df_receipt = pd.read_csv(receipt_csv, low_memory=False)

    df_customer = None
    if customer_json is not None:
        df_customer = _read_customer(
            customer_json,
            customer_name=customer_name,
            customer_format=customer_format,
        )

    return RawInputFrames(invoice=df_invoice, receipt=df_receipt, customer=df_customer)


def _maybe_filter_negative_unpaid_invoices(
    df_invoice: pd.DataFrame, df_receipt: pd.DataFrame
) -> pd.DataFrame:
    """Drop negative/zero invoices only if there is no payment record at all.

    This matches the notebook intent: avoid dropping credit notes that actually have receipts.
    """
    if (
        "CUSTOMER_TRX_ID" not in df_invoice.columns
        or "TRX_AMOUNT" not in df_invoice.columns
    ):
        return df_invoice

    df = df_invoice.copy()
    df["TRX_AMOUNT"] = pd.to_numeric(df["TRX_AMOUNT"], errors="coerce")

    # Determine invoices with any receipt row
    receipt_key = None
    for candidate in ["APPLIED_CUSTOMER_TRX_ID", "CUSTOMER_TRX_ID", "INVOICE_ID"]:
        if candidate in df_receipt.columns:
            receipt_key = candidate
            break

    if receipt_key is None:
        return df

    has_receipt = set(
        pd.to_numeric(df_receipt[receipt_key], errors="coerce")
        .dropna()
        .astype("int64")
        .tolist()
    )

    inv_id = pd.to_numeric(df["CUSTOMER_TRX_ID"], errors="coerce")
    mask_drop = ((df["TRX_AMOUNT"] < 0) | (df["TRX_AMOUNT"] == 0.0)) & (
        ~inv_id.isin(has_receipt)
    )

    return df.loc[~mask_drop].copy()


def apply_credit_memo_netting(
    df_invoice: pd.DataFrame, snapshot_date: pd.Timestamp
) -> pd.DataFrame:
    """Apply Credit Memo (CM) netting to reduce TRX_AMOUNT.

    If PREVIOUS_CUSTOMER_TRX_ID is present, those invoices are treated as credit memos
    reducing the exposure of their parent invoice (up to the snapshot date).
    """
    if df_invoice.empty:
        df_invoice["TRX_AMOUNT_GROSS"] = df_invoice.get("TRX_AMOUNT", 0.0)
        df_invoice["credit_memo_reduction"] = 0.0
        df_invoice["cm_count"] = 0
        df_invoice["cm_first_date"] = pd.NaT
        return df_invoice

    df = df_invoice.copy()

    # Save original gross amount
    if "TRX_AMOUNT" in df.columns:
        df["TRX_AMOUNT_GROSS"] = pd.to_numeric(
            df["TRX_AMOUNT"], errors="coerce"
        ).fillna(0.0)
    else:
        df["TRX_AMOUNT_GROSS"] = 0.0

    df["TRX_AMOUNT"] = df["TRX_AMOUNT_GROSS"]

    prev_col = None
    for cand in ["previous_customer_trx_id", "PREVIOUS_CUSTOMER_TRX_ID"]:
        if cand in df.columns:
            prev_col = cand
            break

    # If no PREVIOUS_CUSTOMER_TRX_ID column, return early (backward compatible)
    if prev_col is None:
        df["credit_memo_reduction"] = 0.0
        df["cm_count"] = 0
        df["cm_first_date"] = pd.NaT
        return df

    # Find Credit Memos (CM)
    cm_src = df[[prev_col, "TRX_AMOUNT", "TRX_DATE"]].copy()
    cm_src = cm_src[cm_src[prev_col].notna()].copy()

    if cm_src.empty:
        df["credit_memo_reduction"] = 0.0
        df["cm_count"] = 0
        df["cm_first_date"] = pd.NaT
        return df

    # Standardize types for CM source
    cm_src[prev_col] = pd.to_numeric(cm_src[prev_col], errors="coerce").astype("Int64")
    cm_src["TRX_AMOUNT"] = pd.to_numeric(cm_src["TRX_AMOUNT"], errors="coerce").fillna(
        0.0
    )
    cm_src["TRX_DATE_dt"] = _to_dt_naive(cm_src["TRX_DATE"])

    # Anti-leakage: only consider CMs available at or before snapshot_date
    cm_src = cm_src[
        cm_src["TRX_DATE_dt"].notna() & (cm_src["TRX_DATE_dt"] <= snapshot_date)
    ].copy()

    # Aggregate by parent invoice ID
    c = (
        cm_src.groupby(prev_col, as_index=False)
        .agg(
            cm_amount_raw_sum=("TRX_AMOUNT", "sum"),  # Usually negative
            cm_count=("TRX_AMOUNT", "size"),
            cm_first_date=("TRX_DATE_dt", "min"),
        )
        .rename(columns={prev_col: "CUSTOMER_TRX_ID"})
    )
    c["CUSTOMER_TRX_ID"] = pd.to_numeric(c["CUSTOMER_TRX_ID"], errors="coerce").astype(
        "Int64"
    )

    # Effective CM only reduces exposure (if positive, it's not a valid CM, cap at 0)
    c["cm_amount_effective"] = c["cm_amount_raw_sum"].where(
        c["cm_amount_raw_sum"] < 0, 0.0
    )
    c["credit_memo_reduction"] = (-c["cm_amount_effective"]).clip(lower=0.0)

    # Merge back to parent invoices
    df["CUSTOMER_TRX_ID_JOIN"] = pd.to_numeric(
        df["CUSTOMER_TRX_ID"], errors="coerce"
    ).astype("Int64")

    df = df.merge(
        c[
            [
                "CUSTOMER_TRX_ID",
                "cm_amount_effective",
                "credit_memo_reduction",
                "cm_count",
                "cm_first_date",
            ]
        ],
        left_on="CUSTOMER_TRX_ID_JOIN",
        right_on="CUSTOMER_TRX_ID",
        how="left",
        suffixes=("", "_cm"),
    )

    df["cm_amount_effective"] = df["cm_amount_effective"].fillna(0.0)
    df["credit_memo_reduction"] = df["credit_memo_reduction"].fillna(0.0)
    df["cm_count"] = df["cm_count"].fillna(0).astype(int)

    # Netting: net = max(gross + effective_cm, 0)
    # Note: effective_cm is negative here
    net = (df["TRX_AMOUNT_GROSS"] + df["cm_amount_effective"]).clip(lower=0.0)

    df["TRX_AMOUNT"] = net

    df.drop(
        columns=["CUSTOMER_TRX_ID_JOIN", "CUSTOMER_TRX_ID_cm", "cm_amount_effective"],
        inplace=True,
        errors="ignore",
    )

    return df


def prepare_base_tables(
    raw: RawInputFrames,
    snapshot_date: pd.Timestamp,
    apply_fix_year_month: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare df_inv (invoice table) and df_pay_raw (payment table) from raw inputs."""

    df_invoice = _maybe_filter_negative_unpaid_invoices(raw.invoice, raw.receipt)

    # Apply Credit Memo Netting
    df_invoice = apply_credit_memo_netting(df_invoice, snapshot_date)

    # Basic invoice schema
    required_invoice = ["CUSTOMER_TRX_ID", "TRX_DATE", "DUE_DATE", "TRX_AMOUNT"]
    missing_invoice = [c for c in required_invoice if c not in df_invoice.columns]
    if missing_invoice:
        raise ValueError(f"Invoice CSV missing required columns: {missing_invoice}")

    df_invoice = df_invoice.copy()
    df_invoice["CUSTOMER_TRX_ID"] = pd.to_numeric(
        df_invoice["CUSTOMER_TRX_ID"], errors="coerce"
    )
    df_invoice = df_invoice[df_invoice["CUSTOMER_TRX_ID"].notna()].copy()
    df_invoice["CUSTOMER_TRX_ID"] = df_invoice["CUSTOMER_TRX_ID"].astype("int64")

    df_invoice["TRX_DATE"] = _to_dt_naive(df_invoice["TRX_DATE"])
    df_invoice["DUE_DATE"] = _to_dt_naive(df_invoice["DUE_DATE"])
    df_invoice["TRX_AMOUNT"] = pd.to_numeric(df_invoice["TRX_AMOUNT"], errors="coerce")

    # Optional PARTY_ID
    if "PARTY_ID" in df_invoice.columns:
        df_invoice["PARTY_ID"] = pd.to_numeric(df_invoice["PARTY_ID"], errors="coerce")

    # Optional TRX_NUMBER (useful for customer mapping when PARTY_ID is missing/bad)
    if "TRX_NUMBER" in df_invoice.columns:
        df_invoice["TRX_NUMBER"] = df_invoice["TRX_NUMBER"].astype(str)

    # Merge customer master if provided
    df_invoice["cust_master_missing"] = 1  # default unknown/missing
    if (
        raw.customer is not None
        and "PARTY_ID" in df_invoice.columns
        and "PARTY_ID" in raw.customer.columns
    ):
        df_cust = raw.customer.copy()
        df_cust["PARTY_ID"] = pd.to_numeric(df_cust["PARTY_ID"], errors="coerce")
        df_cust = df_cust[df_cust["PARTY_ID"].notna()].copy()
        # Prefer mapping by PARTY_ID, but fall back to TRX_NUMBER mapping if available.
        df_cust = df_cust.drop_duplicates(subset=["PARTY_ID"], keep="first")
        df_cust["_in_customer_master_party"] = 1
        cust_keep_party = [
            c
            for c in [
                "PARTY_ID",
                "_in_customer_master_party",
                "ACCOUNT_NUMBER",
                "CUSTOMER_NAME",
            ]
            if c in df_cust.columns
        ]
        df_invoice = df_invoice.merge(
            df_cust[cust_keep_party], on="PARTY_ID", how="left"
        )

        # Optional TRX_NUMBER mapping (OracleCustomer.json contains TRX_NUMBER -> ACCOUNT_NUMBER)
        if "TRX_NUMBER" in df_invoice.columns and "TRX_NUMBER" in raw.customer.columns:
            df_cust_trx = raw.customer.copy()
            df_cust_trx["TRX_NUMBER"] = df_cust_trx["TRX_NUMBER"].astype(str)
            df_cust_trx = df_cust_trx.drop_duplicates(
                subset=["TRX_NUMBER"], keep="first"
            )
            df_cust_trx["_in_customer_master_trx"] = 1
            cust_keep_trx = [
                c
                for c in ["TRX_NUMBER", "_in_customer_master_trx", "ACCOUNT_NUMBER"]
                if c in df_cust_trx.columns
            ]
            df_invoice = df_invoice.merge(
                df_cust_trx[cust_keep_trx].rename(
                    columns={"ACCOUNT_NUMBER": "ACCOUNT_NUMBER_trx"}
                ),
                on="TRX_NUMBER",
                how="left",
            )
        else:
            df_invoice["_in_customer_master_trx"] = pd.NA
            df_invoice["ACCOUNT_NUMBER_trx"] = pd.NA

        # Combine ACCOUNT_NUMBER from PARTY mapping first, else TRX mapping
        if "ACCOUNT_NUMBER" in df_invoice.columns:
            df_invoice["ACCOUNT_NUMBER"] = df_invoice["ACCOUNT_NUMBER"].combine_first(
                df_invoice.get("ACCOUNT_NUMBER_trx")
            )
        else:
            df_invoice["ACCOUNT_NUMBER"] = df_invoice.get("ACCOUNT_NUMBER_trx")

        in_master = df_invoice["_in_customer_master_party"].fillna(0).astype(int)
        in_master_trx = df_invoice.get("_in_customer_master_trx")
        if in_master_trx is not None:
            in_master_trx = in_master_trx.fillna(0).astype(int)
            df_invoice["cust_master_missing"] = (
                (in_master + in_master_trx) == 0
            ).astype(int)
        else:
            df_invoice["cust_master_missing"] = (in_master == 0).astype(int)

        df_invoice.drop(
            columns=[
                "_in_customer_master_party",
                "_in_customer_master_trx",
                "ACCOUNT_NUMBER_trx",
            ],
            inplace=True,
            errors="ignore",
        )
    elif "PARTY_ID" in df_invoice.columns:
        # PARTY_ID exists but no master provided: treat as unknown master
        df_invoice["cust_master_missing"] = 1
    else:
        # No PARTY_ID at all
        df_invoice["cust_master_missing"] = 1

    # Deduplicate invoice header per CUSTOMER_TRX_ID
    keep_cols = [
        c
        for c in [
            "CUSTOMER_TRX_ID",
            "TRX_DATE",
            "DUE_DATE",
            "TRX_AMOUNT",
            "PARTY_ID",
            "ACCOUNT_NUMBER",
            "CUSTOMER_NAME",
            "cust_master_missing",
            "TRANS_TYPE",
            "CURRENCY_CODE",
            "ORG_ID",
            "TRX_AMOUNT_GROSS",
            "credit_memo_reduction",
            "cm_count",
            "cm_first_date",
        ]
        if c in df_invoice.columns
    ]

    df_inv = (
        df_invoice.sort_values(["CUSTOMER_TRX_ID", "TRX_DATE"])
        .drop_duplicates("CUSTOMER_TRX_ID")
        .loc[:, keep_cols]
        .copy()
    )

    df_inv["AS_OF_DATE"] = df_inv["DUE_DATE"].copy()
    df_inv.loc[df_inv["AS_OF_DATE"] > snapshot_date, "AS_OF_DATE"] = snapshot_date

    # Payments raw
    df_receipt = raw.receipt.copy()

    receipt_required = ["APPLIED_CUSTOMER_TRX_ID", "AMOUNT_APPLIED"]
    missing_receipt = [c for c in receipt_required if c not in df_receipt.columns]
    if missing_receipt:
        raise ValueError(f"Receipt CSV missing required columns: {missing_receipt}")

    df_pay_raw = df_receipt.rename(
        columns={"APPLIED_CUSTOMER_TRX_ID": "INVOICE_ID"}
    ).copy()
    df_pay_raw["INVOICE_ID"] = pd.to_numeric(df_pay_raw["INVOICE_ID"], errors="coerce")
    df_pay_raw = df_pay_raw[df_pay_raw["INVOICE_ID"].notna()].copy()
    df_pay_raw["INVOICE_ID"] = df_pay_raw["INVOICE_ID"].astype("int64")

    # Optional fix for receipt/apply dates
    if apply_fix_year_month:
        for col in ["RECEIPT_DATE", "APPLY_DATE"]:
            if col in df_pay_raw.columns:
                df_pay_raw[col] = fix_year_month(df_pay_raw[col], snapshot_date)

    apply_dt = (
        _to_dt_naive(df_pay_raw["APPLY_DATE"])
        if "APPLY_DATE" in df_pay_raw.columns
        else pd.Series(pd.NaT, index=df_pay_raw.index)
    )
    rcpt_dt = (
        _to_dt_naive(df_pay_raw["RECEIPT_DATE"])
        if "RECEIPT_DATE" in df_pay_raw.columns
        else pd.Series(pd.NaT, index=df_pay_raw.index)
    )
    df_pay_raw["PAYMENT_DATE"] = apply_dt.combine_first(rcpt_dt)

    df_pay_raw["AMOUNT_APPLIED"] = pd.to_numeric(
        df_pay_raw["AMOUNT_APPLIED"], errors="coerce"
    ).fillna(0.0)

    if "RECEIPT_STATUS" in df_pay_raw.columns:
        df_pay_raw = df_pay_raw[
            (df_pay_raw["RECEIPT_STATUS"].isna())
            | (df_pay_raw["RECEIPT_STATUS"] == "APP")
        ].copy()

    return (
        df_inv,
        df_pay_raw[
            [
                c
                for c in ["INVOICE_ID", "PAYMENT_DATE", "AMOUNT_APPLIED"]
                if c in df_pay_raw.columns
            ]
        ].copy(),
    )


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
    # Match notebook training scale: percentage (0-100), use abs() & clip
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

    # Use a deterministic, index-safe loop instead of groupby.apply to avoid
    # pandas index edge cases on some inputs.
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
    eps: float = EPS_DEFAULT,
) -> pd.DataFrame:
    """Attach PARTY/customer history signals (anti-leakage)."""

    out = df_feat.copy()

    if "PARTY_ID" not in out.columns:
        out["PARTY_ID"] = np.nan

    out["party_missing"] = out["PARTY_ID"].isna().astype(int)

    # Prior invoice count per PARTY_ID
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

    # Payments up to snapshot
    pay_hist = df_pay_raw.copy()
    pay_hist = pay_hist.rename(columns={"INVOICE_ID": "CUSTOMER_TRX_ID"})
    pay_hist["PAYMENT_DATE"] = _to_dt_naive(pay_hist["PAYMENT_DATE"])
    pay_hist = pay_hist[
        pay_hist["PAYMENT_DATE"].notna() & (pay_hist["PAYMENT_DATE"] <= snapshot_date)
    ].copy()
    pay_hist["AMOUNT_APPLIED"] = pd.to_numeric(
        pay_hist["AMOUNT_APPLIED"], errors="coerce"
    ).fillna(0.0)

    # Event 1: BD90 per invoice (event_date = DUE_DATE+90, flag if not paid by then)
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

    # Event 2: late first payment > 90 (event_date = first payment date)
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

    # Event 3: payment gap >90 (event_date = later payment where the gap is observed)
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

    # Count events strictly before each target invoice TRX_DATE
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

    # Ensure ints
    for c in [
        "party_prior_bd90_cnt",
        "party_prior_late_firstpay90_cnt",
        "party_prior_gap90_cnt",
    ]:
        out[c] = out[c].fillna(0).astype(int)

    # ==========================================================
    # Notebook-Aligned Features for 16-feature models
    # Ported from: Test_New_Test_SMOTE + Auto Search Parameter 16 Features.ipynb
    # ==========================================================

    # --- Step 1: Build historical invoice table (_df_inv_full) ---
    # Aggregate payments per invoice to get actual_dpd, is_paid, etc.
    _pay = df_pay_raw.copy()
    _pay = _pay.rename(columns={"INVOICE_ID": "CUSTOMER_TRX_ID"})
    _pay["PAYMENT_DATE"] = pd.to_datetime(_pay["PAYMENT_DATE"], errors="coerce")
    _pay["AMOUNT_APPLIED"] = pd.to_numeric(
        _pay["AMOUNT_APPLIED"], errors="coerce"
    ).fillna(0.0)
    _pay = _pay[_pay["PAYMENT_DATE"].notna()].copy()
    _pay = _pay.sort_values(["CUSTOMER_TRX_ID", "PAYMENT_DATE"])

    # Gap days between payments per invoice
    _pay["gap_days"] = _pay.groupby("CUSTOMER_TRX_ID")["PAYMENT_DATE"].diff().dt.days

    # Aggregate per invoice
    _receipt_agg = _pay.groupby("CUSTOMER_TRX_ID", as_index=False).agg(
        receipt_count=("PAYMENT_DATE", "count"),
        first_pay_date=("PAYMENT_DATE", "min"),
        last_pay_date=("PAYMENT_DATE", "max"),
        total_paid=("AMOUNT_APPLIED", "sum"),
        gap_variance=("gap_days", "var"),
    )
    _receipt_agg["is_partial"] = (_receipt_agg["receipt_count"] > 1).astype(int)

    # Build _df_inv_full from df_inv (full invoice table)
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

    # --- Step 2: Compute 9 Historical Features ---
    _df = _df_inv_full.copy()

    # Fitur 1: customer_maturity_days
    _df["_first_trx"] = _df.groupby("PARTY_ID")["TRX_DATE"].transform("first")
    _df["customer_maturity_days"] = (_df["TRX_DATE"] - _df["_first_trx"]).dt.days
    _df.loc[_df.groupby("PARTY_ID").cumcount() == 0, "customer_maturity_days"] = 0

    # Fitur 2: business_scale_proxy
    _df["_cumsum_amt"] = _df.groupby("PARTY_ID")["TRX_AMOUNT"].cumsum()
    _df["business_scale_proxy"] = _df["_cumsum_amt"] - _df["TRX_AMOUNT"]

    # Expanding stats (strictly-before = shift(1).expanding())
    _df["_dpd_for_avg"] = _df["actual_dpd"].where(_df["is_paid"] == 1)
    _grp = _df.groupby("PARTY_ID")

    # Fitur 3: historical_avg_dpd
    _df["historical_avg_dpd"] = (
        _grp["_dpd_for_avg"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # Fitur 4: historical_late_payment_ratio
    _df["_late_for_ratio"] = _df["is_late"].where(_df["is_paid"] == 1)
    _df["historical_late_payment_ratio"] = (
        _grp["_late_for_ratio"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    ) * 100

    # Fitur 5: unpaid_exposure_ratio
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

    # Fitur 6: payment_consistency_variance
    _df["_gapvar_valid"] = _df["gap_variance"].where(_df["is_paid"] == 1)
    _df["payment_consistency_variance"] = (
        _grp["_gapvar_valid"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # Fitur 7: historical_partial_payment_freq
    _df["_partial_valid"] = _df["is_partial"].where(_df["is_paid"] == 1).astype(float)
    _df["historical_partial_payment_freq"] = (
        _grp["_partial_valid"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    ) * 100

    # Fitur 8: hist_avg_payment_completion_ratio
    _df["_pr_valid"] = _df["paid_ratio"].where(_df["is_paid"] == 1)
    _df["hist_avg_payment_completion_ratio"] = (
        _grp["_pr_valid"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # Fitur 9: total_outstanding_at_trx
    _df["_outstanding_amt"] = (_df["TRX_AMOUNT"] - _df["total_paid"].fillna(0)).clip(
        lower=0
    )
    _df["total_outstanding_at_trx"] = (
        _grp["_outstanding_amt"]
        .apply(lambda s: s.shift(1).expanding().sum())
        .reset_index(level=0, drop=True)
    ).fillna(0)

    # --- Step 3: Compute 4 Survival-Based Features ---
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

    # --- Step 4: Merge computed features back into `out` ---
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

    # Cold-start defaults (notebook convention)
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

    # Log transforms (notebook convention)
    out["TRX_AMOUNT_log1p"] = np.log1p(
        pd.to_numeric(out["TRX_AMOUNT"], errors="coerce").fillna(0).clip(lower=0)
    )
    out["business_scale_proxy"] = np.log1p(out["business_scale_proxy"].clip(lower=0))
    # ==========================================================

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

    # Re-attach PARTY and cust flags (mon features come from df_inv_tmp, so PARTY_ID not in there)
    # We keep PARTY_id/cust flags from df_inv.
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

    # Add mon_observable flag (optional for evaluation/training)
    trx_dt = (
        _to_dt_naive(df_feat["TRX_DATE"])
        if "TRX_DATE" in df_feat.columns
        else pd.Series(pd.NaT, index=df_feat.index)
    )
    df_feat["mon_observable"] = trx_dt <= (
        snapshot_date - pd.Timedelta(days=monitor_observe_min_days)
    )

    return df_feat, df_inv


# ═══════════════════════════════════════════════════════════════════════
# New-Featured Model: Additional Feature Engineering
# (ported from notebook Cell 66-69 for lgbm_smote_randomsearch_new_featured)
# ═══════════════════════════════════════════════════════════════════════


def make_receipt_agg(df_pay_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate receipt/payment data per invoice (gap_variance, is_partial, etc.).

    Mirrors notebook Cell 66: builds df_receipt_agg from raw payment table.
    """
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
    """Build _df_inv_full with payment-derived columns (is_paid, actual_dpd, etc.).

    Mirrors notebook Cell 67.
    """
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
    """Compute 9 historical features for the new-featured model.

    Mirrors notebook Cell 68. All features are anti-leakage
    (shift(1).expanding) so only past data is used.
    """
    _df = df_inv_full.copy()

    # Ensure PARTY_ID is present (fill with -1 if missing)
    if "PARTY_ID" not in _df.columns:
        _df["PARTY_ID"] = -1
    _df["PARTY_ID"] = pd.to_numeric(_df["PARTY_ID"], errors="coerce").fillna(-1)

    # Fitur 1: customer_maturity_days
    _df["_first_trx"] = _df.groupby("PARTY_ID")["TRX_DATE"].transform("first")
    _df["customer_maturity_days"] = (_df["TRX_DATE"] - _df["_first_trx"]).dt.days
    _df.loc[_df.groupby("PARTY_ID").cumcount() == 0, "customer_maturity_days"] = 0

    # Fitur 2: business_scale_proxy
    _df["_cumsum_amt"] = _df.groupby("PARTY_ID")["TRX_AMOUNT"].cumsum()
    _df["business_scale_proxy"] = _df["_cumsum_amt"] - _df["TRX_AMOUNT"]

    # Prepare grouped
    _df["_dpd_for_avg"] = _df["actual_dpd"].where(_df["is_paid"] == 1)
    _grp = _df.groupby("PARTY_ID")

    # Fitur 3: historical_avg_dpd
    _df["historical_avg_dpd"] = (
        _grp["_dpd_for_avg"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # Fitur 4: historical_late_payment_ratio
    _df["_late_for_ratio"] = _df["is_late"].where(_df["is_paid"] == 1)
    _df["historical_late_payment_ratio"] = (
        _grp["_late_for_ratio"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    ) * 100

    # Fitur 5: unpaid_exposure_ratio
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

    # Fitur 6: payment_consistency_variance
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

    # Fitur 7: historical_partial_payment_freq
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

    # Fitur 8: hist_avg_payment_completion_ratio
    _df["_pr_valid"] = _df["paid_ratio"].where(_df["is_paid"] == 1)
    _df["hist_avg_payment_completion_ratio"] = (
        _grp["_pr_valid"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # Fitur 9: total_outstanding_at_trx
    _df["_outstanding_amt"] = (_df["TRX_AMOUNT"] - _df["total_paid"].fillna(0)).clip(
        lower=0
    )
    _df["total_outstanding_at_trx"] = (
        _grp["_outstanding_amt"]
        .apply(lambda s: s.shift(1).expanding().sum())
        .reset_index(level=0, drop=True)
    ).fillna(0)

    # Return only the needed columns
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
    """Prepare feature table for the new-featured model.

    This combines the standard pre-due features with the 9 new
    historical features + TRX_AMOUNT_log1p.
    """
    # Step 1: Standard base tables
    df_inv, df_pay_raw = prepare_base_tables(raw, snapshot_date=snapshot_date)

    # Step 2: Receipt aggregation (for gap_variance, is_partial, etc.)
    receipt_agg = make_receipt_agg(df_pay_raw)

    # Step 3: Build full invoice history table
    df_inv_full = build_invoice_history(df_inv, receipt_agg)

    # Step 4: Compute 9 new historical features
    df_new_features = make_new_model_features(df_inv_full)

    # Step 5: Standard pre-due features
    df_feat = make_features_pre_due(df_inv, df_pay_raw)
    df_feat = add_party_history_features(
        df_feat, df_inv, df_pay_raw, snapshot_date=snapshot_date
    )

    # Step 6: Merge new features into main feature table
    df_feat = df_feat.merge(df_new_features, on="CUSTOMER_TRX_ID", how="left")

    # Step 7: Add TRX_AMOUNT_log1p
    df_feat["TRX_AMOUNT_log1p"] = np.log1p(
        pd.to_numeric(df_feat["TRX_AMOUNT"], errors="coerce").clip(lower=0).fillna(0)
    )

    return df_feat, df_inv
