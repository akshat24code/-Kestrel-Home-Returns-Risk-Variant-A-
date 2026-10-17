"""Feature engineering for Kestrel returns risk.

Only uses information available at dispatch time. Columns that are written
AFTER a return is raised (last_service_event_type, pickup_scheduled_at) and the
ingestion column `source` are deliberately never read here.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Columns we refuse to use, with the reason. Kept here so the decision is in code.
EXCLUDED = {
    "last_service_event_type": "written after delivery; REVERSE_PICKUP == returned. Test snapshot has none.",
    "pickup_scheduled_at": "booked only after a return is approved (92% returned when present). Empty in test.",
    "source": "ingestion artefact (partner_feed duplicates); not a property of the order.",
    "order_id": "identifier",
    "customer_id": "identifier (history is captured via prior_orders / prior_returns)",
}

NOTE_CATEGORIES = [
    ("neighbour", r"neighbour"),
    ("landmark", r"^landmark"),
    ("gate_code", r"gate code"),
    ("call_before", r"^call before"),
    ("fragile", r"fragile"),
    ("security", r"security"),
    ("office", r"office"),
    ("third_floor", r"floor"),
    ("whatsapp", r"whatsapp"),
    ("ring_twice", r"ring twice"),
    ("morning", r"morning"),
    ("after_6", r"after \d+ ?pm"),
]

CATEGORICAL = ["sales_channel", "payment_mode", "family", "state", "note_cat", "is_gift", "shield_member"]
NUMERIC = [
    "discount_pct", "qty", "order_value_inr", "list_price_inr", "promised_delivery_days",
    "pincode_missing", "customer_prior_orders", "customer_prior_returns", "prior_return_rate",
    "warranty_months", "product_age_days", "tenure_days", "hour", "dow", "is_weekend",
    "value_per_unit", "note_has_injection",
]


def note_category(note) -> str:
    if not isinstance(note, str) or not note.strip():
        return "none"
    n = note.lower()
    for name, pat in NOTE_CATEGORIES:
        if re.search(pat, n):
            return name
    return "other"


def note_has_injection(note) -> int:
    """Some notes carry text addressed to 'data team / automated tools'. Flag it, never obey it."""
    if not isinstance(note, str):
        return 0
    return int(bool(re.search(r"data team|automated|assistant|reviewer|for analysis", note, re.I)))


def load_reference():
    """products.csv is non-personal and committed. customers.csv is customer data and is NOT committed
    (policy §10); without it the service still runs, treating every customer as unknown/non-Shield
    unless the request carries `shield_member`, `signup_date`, `state`."""
    cpath = DATA_DIR / "customers.csv"
    if cpath.exists():
        customers = pd.read_csv(cpath, dtype={"customer_id": str})
    else:
        customers = pd.DataFrame(columns=["customer_id", "city", "state", "signup_date", "shield_member"])
    products = pd.read_csv(DATA_DIR / "products.csv")
    return customers, products


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Data fixes found during exploration (see EVIDENCE.md §1)."""
    df = df.copy()
    # 1. partner_feed rows are exact re-imports of crm rows -> drop the copy
    if "source" in df.columns:
        df = df.sort_values("source").drop_duplicates("order_id", keep="first")
    # 2. Oct-2025 orders through the new gateway were stored in paise (exactly 100x)
    placed = pd.to_datetime(df["order_placed_at"])
    oct25 = (placed >= "2025-10-01") & (placed < "2025-11-01")
    df.loc[oct25 & (df["order_value_inr"] > 0), "order_value_inr"] = df.loc[oct25, "order_value_inr"] / 100.0
    return df.sort_values("order_placed_at").reset_index(drop=True)


def fix_value_against_list(df: pd.DataFrame) -> pd.DataFrame:
    """Generic guard for single records (service): if the value is ~100x list*(1-disc)*qty, treat as paise."""
    exp = df["list_price_inr"] * (1 - df["discount_pct"] / 100.0) * df["qty"]
    ratio = df["order_value_inr"] / exp.replace(0, np.nan)
    bad = ratio.between(50, 200)
    df.loc[bad, "order_value_inr"] = df.loc[bad, "order_value_inr"] / 100.0
    return df


def build_features(orders: pd.DataFrame, customers=None, products=None) -> pd.DataFrame:
    if customers is None or products is None:
        customers, products = load_reference()
    df = orders.copy()
    df["customer_id"] = df["customer_id"].astype(str)
    # request-supplied customer attributes win over the lookup table
    override = [c for c in ("city", "state", "signup_date", "shield_member") if c in df.columns]
    df = df.merge(customers.drop(columns=override), on="customer_id", how="left")
    df = df.merge(products, on="sku", how="left")
    df = fix_value_against_list(df)

    placed = pd.to_datetime(df["order_placed_at"], errors="coerce")
    df["hour"] = placed.dt.hour
    df["dow"] = placed.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["product_age_days"] = (placed - pd.to_datetime(df["launch_date"], errors="coerce")).dt.days
    df["tenure_days"] = (placed - pd.to_datetime(df["signup_date"], errors="coerce")).dt.days
    df["pincode_missing"] = (pd.to_numeric(df["delivery_pincode"], errors="coerce").fillna(0) == 0).astype(int)
    df["prior_return_rate"] = (df["customer_prior_returns"] + 0.5) / (df["customer_prior_orders"] + 3)
    df["value_per_unit"] = df["order_value_inr"] / df["qty"].clip(lower=1)
    df["note_cat"] = df["delivery_note"].map(note_category)
    df["note_has_injection"] = df["delivery_note"].map(note_has_injection)
    df["shield_member"] = df["shield_member"].fillna("N")
    df["is_gift"] = df["is_gift"].fillna("N")
    for c in CATEGORICAL:
        df[c] = df[c].fillna("unknown").astype(str)
    # Unknown customer (no signup date) is treated as brand new; nothing in training is missing.
    df["tenure_days"] = df["tenure_days"].fillna(0)
    df[NUMERIC] = df[NUMERIC].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df[CATEGORICAL + NUMERIC]
