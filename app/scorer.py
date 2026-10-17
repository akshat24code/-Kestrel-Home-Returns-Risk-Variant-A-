"""Load the model once and score a single order record with plain-English reasons."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import load_reference, build_features
from src.policy import decide, CALL_THRESHOLD

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ["order_placed_at", "customer_id", "sku", "sales_channel", "payment_mode",
            "discount_pct", "qty", "order_value_inr", "promised_delivery_days"]
DEFAULTS = {"delivery_pincode": 0, "is_gift": "N", "customer_prior_orders": 0,
            "customer_prior_returns": 0, "delivery_note": ""}
IGNORED_IF_SENT = ["last_service_event_type", "pickup_scheduled_at", "source", "returned"]

# How each model input reads to a dispatch agent
LABELS = {
    "payment_mode": "Payment: {v}",
    "customer_prior_returns": "Customer has returned {v} earlier order(s)",
    "prior_return_rate": "Customer's past return rate",
    "shield_member": "Kestrel Shield member: {v} (free returns)",
    "list_price_inr": "Product price (Rs {v:,.0f} list)",
    "value_per_unit": "Price paid per unit",
    "order_value_inr": "Order value",
    "promised_delivery_days": "Promised delivery in {v} days",
    "family": "Product family: {v}",
    "sales_channel": "Channel: {v}",
    "is_gift": "Gift order: {v}",
    "discount_pct": "Discount {v}%",
    "warranty_months": "{v}-month warranty",
    "state": "Delivery state: {v}",
    "note_cat": "Delivery note type: {v}",
    "pincode_missing": "No delivery pincode captured",
    "customer_prior_orders": "Customer has {v} earlier order(s)",
    "tenure_days": "Customer since {v} days",
    "product_age_days": "Model on sale for {v} days",
    "qty": "Quantity {v}",
    "hour": "Ordered at {v}:00", "dow": "Day of week", "is_weekend": "Weekend order",
    "note_has_injection": "Delivery note contains unusual system-style text",
}


class Scorer:
    def __init__(self, path: Path = ROOT / "model" / "model.joblib"):
        try:
            bundle = joblib.load(path)
        except Exception as e:  # e.g. different scikit-learn version on this machine
            if not (ROOT / "data" / "train.csv").exists():
                raise SystemExit(f"Could not load {path.name} ({type(e).__name__}). Install the pinned versions "
                                 "from requirements.txt, or put the data pack in data/ and run `python -m src.train`.")
            print(f"Model file unreadable ({type(e).__name__}); retraining from data/ ...")
            from src import train
            train.main()
            bundle = joblib.load(path)
        self.pipe = bundle["pipeline"]
        self.base_rate = bundle["base_rate"]
        self.meta = {k: bundle[k] for k in ("trained_on", "holdout_auc")}
        self.customers, self.products = load_reference()
        self.known_skus = set(self.products.sku)
        pre, clf = self.pipe.named_steps["pre"], self.pipe.named_steps["clf"]
        self.names = pre.get_feature_names_out()
        self.coef = clf.coef_[0]
        self.groups = np.array([self._group(n) for n in self.names])
        # reference point = average training order in transformed space
        self.ref = np.asarray(bundle["ref"])

    @staticmethod
    def _group(name: str) -> str:
        kind, rest = name.split("__", 1)
        if kind == "num":
            return rest
        for col in ["sales_channel", "payment_mode", "family", "state", "note_cat", "is_gift", "shield_member"]:
            if rest.startswith(col + "_") or rest == col:
                return col
        return rest

    def validate(self, rec: dict) -> list[str]:
        errs = [f"missing field '{f}'" for f in REQUIRED if rec.get(f) in (None, "")]
        if rec.get("sku") and rec["sku"] not in self.known_skus:
            errs.append(f"unknown sku '{rec['sku']}'")
        for f in ["discount_pct", "qty", "order_value_inr", "promised_delivery_days"]:
            if rec.get(f) not in (None, ""):
                try:
                    float(rec[f])
                except (TypeError, ValueError):
                    errs.append(f"'{f}' must be a number")
        try:
            pd.to_datetime(rec.get("order_placed_at"))
        except Exception:
            errs.append("'order_placed_at' must be a date-time like 2026-07-01 10:30")
        return errs

    def score(self, rec: dict) -> dict:
        errs = self.validate(rec)
        if errs:
            return {"ok": False, "errors": errs}
        r = {**DEFAULTS, **{k: v for k, v in rec.items() if v not in (None, "")}}
        ignored = [k for k in IGNORED_IF_SENT if k in rec]
        for k in ["discount_pct", "qty", "promised_delivery_days", "customer_prior_orders", "customer_prior_returns", "delivery_pincode"]:
            r[k] = int(float(r[k]))
        r["order_value_inr"] = float(r["order_value_inr"])
        df = pd.DataFrame([r])
        X = build_features(df, self.customers, self.products)
        p = float(self.pipe.predict_proba(X)[0, 1])

        z = self.pipe.named_steps["pre"].transform(X)
        z = z.toarray()[0] if hasattr(z, "toarray") else np.asarray(z)[0]
        contrib = self.coef * (z - self.ref)
        by_group = pd.Series(contrib).groupby(self.groups).sum()
        # merge the two prior-return features for readability
        if "prior_return_rate" in by_group and "customer_prior_returns" in by_group:
            by_group["customer_prior_returns"] += by_group.pop("prior_return_rate")
        top = by_group.reindex(by_group.abs().sort_values(ascending=False).index)
        row = X.iloc[0]
        reasons = []
        for g, w in top.items():
            if abs(w) < 0.08 or len(reasons) >= 5:
                continue
            v = row[g] if g in row.index else ""
            special = {("shield_member", "Y"): "Kestrel Shield member (returns are free for them)",
                       ("shield_member", "N"): "Not a Shield member",
                       ("customer_prior_returns", 0): "No earlier returns by this customer",
                       ("is_gift", "Y"): "Gift order", ("is_gift", "N"): "Not a gift"}
            if (g, v) in special:
                reasons.append({"factor": special[(g, v)], "direction": "raises risk" if w > 0 else "lowers risk",
                                "strength": round(float(w), 2)})
                continue
            try:
                text = LABELS.get(g, g).format(v=v)
            except (ValueError, TypeError):
                text = LABELS.get(g, g).format(v=str(v))
            reasons.append({"factor": text, "direction": "raises risk" if w > 0 else "lowers risk",
                            "strength": round(float(w), 2)})
        d = decide(p, row["shield_member"])
        return {
            "ok": True,
            "order_id": rec.get("order_id"),
            "return_probability": round(p, 4),
            "relative_to_average": round(p / self.base_rate, 2),
            "risk_band": d["band"],
            "recommended_action": d["action"],
            "expected_saving_from_call_inr": d["expected_saving_from_call_inr"],
            "reasons": reasons,
            "note": d["note"],
            "ignored_fields": ignored,
            "explanation": (f"About {round(p*100)} in 100 orders like this are returned "
                            f"(average order: {round(self.base_rate*100)} in 100). "
                            f"A confirmation call pays for itself above {round(CALL_THRESHOLD*100,1)}%."),
            "model": self.meta,
        }
