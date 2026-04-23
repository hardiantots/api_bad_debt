from __future__ import annotations

from typing import Tuple

import pandas as pd

from bad_debt_app.feature_engineering.io import (
    RawInputFrames,
    _to_dt_naive,
    fix_year_month,
)


def _maybe_filter_negative_unpaid_invoices(
    df_invoice: pd.DataFrame, df_receipt: pd.DataFrame
) -> pd.DataFrame:
    """Drop negative/zero invoices only if there is no payment record at all."""
    if (
        "CUSTOMER_TRX_ID" not in df_invoice.columns
        or "TRX_AMOUNT" not in df_invoice.columns
    ):
        return df_invoice

    df = df_invoice.copy()
    df["TRX_AMOUNT"] = pd.to_numeric(df["TRX_AMOUNT"], errors="coerce")

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
    """Apply Credit Memo (CM) netting to reduce TRX_AMOUNT."""
    if df_invoice.empty:
        df_invoice["TRX_AMOUNT_GROSS"] = df_invoice.get("TRX_AMOUNT", 0.0)
        df_invoice["credit_memo_reduction"] = 0.0
        df_invoice["cm_count"] = 0
        df_invoice["cm_first_date"] = pd.NaT
        return df_invoice

    df = df_invoice.copy()

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

    if prev_col is None:
        df["credit_memo_reduction"] = 0.0
        df["cm_count"] = 0
        df["cm_first_date"] = pd.NaT
        return df

    cm_src = df[[prev_col, "TRX_AMOUNT", "TRX_DATE"]].copy()
    cm_src = cm_src[cm_src[prev_col].notna()].copy()

    if cm_src.empty:
        df["credit_memo_reduction"] = 0.0
        df["cm_count"] = 0
        df["cm_first_date"] = pd.NaT
        return df

    cm_src[prev_col] = pd.to_numeric(cm_src[prev_col], errors="coerce").astype("Int64")
    cm_src["TRX_AMOUNT"] = pd.to_numeric(cm_src["TRX_AMOUNT"], errors="coerce").fillna(
        0.0
    )
    cm_src["TRX_DATE_dt"] = _to_dt_naive(cm_src["TRX_DATE"])
    cm_src = cm_src[
        cm_src["TRX_DATE_dt"].notna() & (cm_src["TRX_DATE_dt"] <= snapshot_date)
    ].copy()

    c = (
        cm_src.groupby(prev_col, as_index=False)
        .agg(
            cm_amount_raw_sum=("TRX_AMOUNT", "sum"),
            cm_count=("TRX_AMOUNT", "size"),
            cm_first_date=("TRX_DATE_dt", "min"),
        )
        .rename(columns={prev_col: "CUSTOMER_TRX_ID"})
    )
    c["CUSTOMER_TRX_ID"] = pd.to_numeric(c["CUSTOMER_TRX_ID"], errors="coerce").astype(
        "Int64"
    )

    c["cm_amount_effective"] = c["cm_amount_raw_sum"].where(
        c["cm_amount_raw_sum"] < 0, 0.0
    )
    c["credit_memo_reduction"] = (-c["cm_amount_effective"]).clip(lower=0.0)

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
    df_invoice = apply_credit_memo_netting(df_invoice, snapshot_date)

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

    if "PARTY_ID" in df_invoice.columns:
        df_invoice["PARTY_ID"] = pd.to_numeric(df_invoice["PARTY_ID"], errors="coerce")

    if "TRX_NUMBER" in df_invoice.columns:
        df_invoice["TRX_NUMBER"] = df_invoice["TRX_NUMBER"].astype(str)

    df_invoice["cust_master_missing"] = 1
    if (
        raw.customer is not None
        and "PARTY_ID" in df_invoice.columns
        and "PARTY_ID" in raw.customer.columns
    ):
        df_cust = raw.customer.copy()
        df_cust["PARTY_ID"] = pd.to_numeric(df_cust["PARTY_ID"], errors="coerce")
        df_cust = df_cust[df_cust["PARTY_ID"].notna()].copy()
        df_cust = df_cust.drop_duplicates(subset=["PARTY_ID"], keep="first")
        df_cust["_in_customer_master_party"] = 1

        cust_keep_party = [
            c
            for c in [
                "PARTY_ID",
                "_in_customer_master_party",
                "ACCOUNT_NUMBER",
                "CUSTOMER_NAME",
                "BU_NAME",
            ]
            if c in df_cust.columns
        ]
        df_invoice = df_invoice.merge(
            df_cust[cust_keep_party], on="PARTY_ID", how="left"
        )

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

        # Rename BU_NAME → SBU for consistent downstream output
        if "BU_NAME" in df_invoice.columns:
            df_invoice.rename(columns={"BU_NAME": "SBU"}, inplace=True)
    else:
        df_invoice["cust_master_missing"] = 1

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
            "SBU",
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
