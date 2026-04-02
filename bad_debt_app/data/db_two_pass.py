from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger("bad_debt_api")


def fetch_raw_inputs_two_pass(
    *,
    time_range: str,
    year: int,
    start_date: str | None,
    end_date: str | None,
    snapshot_date: str | None,
    get_engine,
    fetch_customers,
    fetch_invoices,
    fetch_receipts,
    raw_frames_cls,
):
    engine = get_engine()
    df_customer = fetch_customers(engine=engine)

    df_target_inv = fetch_invoices(
        time_range=time_range,
        year=year,
        engine=engine,
        start_date=start_date,
        end_date=end_date,
        snapshot_date=snapshot_date,
    )

    if df_target_inv.empty:
        return raw_frames_cls(
            invoice=pd.DataFrame(), receipt=pd.DataFrame(), customer=df_customer
        )

    target_trx_ids = df_target_inv["CUSTOMER_TRX_ID"].dropna().unique().tolist()

    if time_range == "all":
        df_receipt = fetch_receipts(
            time_range=time_range,
            year=year,
            engine=engine,
            start_date=start_date,
            end_date=end_date,
            snapshot_date=snapshot_date,
        )
        return raw_frames_cls(
            invoice=df_target_inv,
            receipt=df_receipt,
            customer=df_customer,
            target_trx_ids=target_trx_ids,
        )

    df_t = df_target_inv.copy()
    if "PARTY_ID" not in df_t.columns:
        df_t["PARTY_ID"] = np.nan

    missing_party_mask = df_t["PARTY_ID"].isna() | (df_t["PARTY_ID"] == "")
    if (
        missing_party_mask.any()
        and "TRX_NUMBER" in df_t.columns
        and not df_customer.empty
    ):
        mapping = df_customer.set_index("TRX_NUMBER")["PARTY_ID"].to_dict()
        df_t.loc[missing_party_mask, "PARTY_ID"] = df_t.loc[
            missing_party_mask, "TRX_NUMBER"
        ].map(mapping)

    target_parties = df_t["PARTY_ID"].dropna().unique().tolist()

    if not target_parties:
        df_all_inv = df_target_inv
    else:
        party_list_str = ",".join([f"'{str(p)}'" for p in target_parties])
        snap_cond = ""
        params = {}
        if snapshot_date:
            snap_cond = "AND STR_TO_DATE(TRX_DATE, :fmt) <= :snapshot_date"
            params = {"fmt": "%d-%b-%Y", "snapshot_date": snapshot_date}

        history_query = f"""
            SELECT * FROM ar_invoice_list_2
            WHERE PARTY_ID IN ({party_list_str})
            {snap_cond}
        """
        try:
            df_hist_inv = pd.read_sql(text(history_query), engine, params=params)
        except Exception as exc:
            logger.exception("Failed to fetch historical invoices for target parties")
            raise RuntimeError("Historical invoice query failed.") from exc

        df_all_inv = pd.concat([df_target_inv, df_hist_inv]).drop_duplicates(
            subset=["CUSTOMER_TRX_ID"]
        )

    all_historic_trx_ids = df_all_inv["CUSTOMER_TRX_ID"].dropna().unique().tolist()
    if not all_historic_trx_ids:
        df_all_receipt = pd.DataFrame()
    else:
        all_receipts = []
        chunk_size = 5000
        for i in range(0, len(all_historic_trx_ids), chunk_size):
            chunk = all_historic_trx_ids[i : i + chunk_size]
            trx_list_str = ",".join([f"'{str(t)}'" for t in chunk])
            snap_cond_rcpt = ""
            params_rcpt = {}
            if snapshot_date:
                snap_cond_rcpt = "AND (STR_TO_DATE(LEFT(APPLY_DATE, 10), :fmt) <= :snapshot_date OR STR_TO_DATE(LEFT(RECEIPT_DATE, 10), :fmt) <= :snapshot_date)"
                params_rcpt = {"fmt": "%Y-%m-%d", "snapshot_date": snapshot_date}

            rcpt_query = f"""
                SELECT * FROM ar_receipt_list
                WHERE APPLIED_CUSTOMER_TRX_ID IN ({trx_list_str})
                {snap_cond_rcpt}
            """
            try:
                chunk_df = pd.read_sql(text(rcpt_query), engine, params=params_rcpt)
            except Exception as exc:
                logger.exception("Failed to fetch receipt chunk")
                raise RuntimeError("Historical receipt query failed.") from exc
            all_receipts.append(chunk_df)

        df_all_receipt = (
            pd.concat(all_receipts).drop_duplicates()
            if all_receipts
            else pd.DataFrame()
        )

    return raw_frames_cls(
        invoice=df_all_inv,
        receipt=df_all_receipt,
        customer=df_customer,
        target_trx_ids=target_trx_ids,
    )
