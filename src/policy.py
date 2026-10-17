"""Turn a return probability into an action, using Kestrel's own policy numbers (ops-policy v4.1).

Decision rule (per order):
  Confirmation call is worth it when   P(return) x 35% x Rs1,150  >  Rs45
  -> break-even P(return) = 45 / (0.35 x 1150) = 11.2%

  A 24h+ hold only "works" through cancellation: 12% of held customers cancel.
  If the order would have been returned we save Rs1,150; if not we lose the sale's margin.
  -> hold worth it only when P x 1150 > (1-P) x margin. With an assumed 20% contribution
     margin on an average ~Rs9,000 order (Rs1,800), break-even P is ~61%, and even then the
     expected gain per held order is tiny (x12%). We do NOT recommend holds.
"""
from __future__ import annotations

import numpy as np

COSTS = {
    "return_cost": 1150,          # §4, Farhan confirms; Ritu's Rs600 is not the policy figure
    "call_cost": 45,              # §4
    "call_prevents": 0.35,        # §7 spring pilot
    "hold_cancel_rate": 0.12,     # §7
    "assumed_margin_pct": 0.20,   # ASSUMPTION - not in the pack; Farhan should replace it
}
CALL_THRESHOLD = COSTS["call_cost"] / (COSTS["call_prevents"] * COSTS["return_cost"])  # 0.1118
HIGH_THRESHOLD = 0.35  # "high risk" band shown to agents; above this a call is clearly worth it


def decide(p: float, shield: str = "N") -> dict:
    ev_call = p * COSTS["call_prevents"] * COSTS["return_cost"] - COSTS["call_cost"]
    if p >= CALL_THRESHOLD:
        action = "call_before_dispatch"
        band = "high" if p >= HIGH_THRESHOLD else "elevated"
    else:
        action = "ship_normally"
        band = "low"
    note = None
    if shield == "Y" and action != "ship_normally":
        note = "Shield member: use the service-courtesy script; never hold."
    return {"action": action, "band": band, "expected_saving_from_call_inr": round(ev_call, 1), "note": note}


def _hold(y, sel, values):
    c = COSTS
    tp, fp = (y[sel] == 1).sum(), (y[sel] == 0).sum()
    saved = c["hold_cancel_rate"] * tp * c["return_cost"]
    lost = c["hold_cancel_rate"] * (values[sel][y[sel] == 0] * c["assumed_margin_pct"]).sum()
    return {"orders_touched": int(sel.sum()), "true_returns_caught": int(tp), "false_alarms": int(fp),
            "good_orders_cancelled": round(c["hold_cancel_rate"] * fp, 1),
            "returns_avoided": round(c["hold_cancel_rate"] * tp, 1),
            "net_inr": round(saved - lost, 0)}


def _call(y, sel):
    c = COSTS
    tp = (y[sel] == 1).sum()
    saved = c["call_prevents"] * tp * c["return_cost"]
    spent = c["call_cost"] * sel.sum()
    return {"orders_touched": int(sel.sum()), "true_returns_caught": int(tp), "false_alarms": int(sel.sum() - tp),
            "returns_avoided": round(c["call_prevents"] * tp, 1), "call_spend_inr": int(spent),
            "net_inr": round(saved - spent, 0)}


def rupee_table(y, p, shield, values) -> dict:
    y, p, shield, values = map(np.asarray, (y, p, shield, values))
    n = len(y)
    top = lambda f: p >= np.quantile(p, 1 - f)
    out = {
        "n_orders": int(n), "actual_returns": int(y.sum()),
        "cost_of_returns_inr": int(y.sum() * COSTS["return_cost"]),
        "call_threshold": round(CALL_THRESHOLD, 4),
        "strategies": {
            "hold_flagged_p>=0.5 (Ritu's plan)": _hold(y, p >= 0.5, values),
            "hold_top10pct": _hold(y, top(0.10), values),
            "call_top10pct": _call(y, top(0.10)),
            "call_top20pct": _call(y, top(0.20)),
            "call_above_breakeven (recommended)": _call(y, p >= CALL_THRESHOLD),
            "call_above_breakeven_nonShield_only": _call(y, (p >= CALL_THRESHOLD) & (shield != "Y")),
            "call_everyone": _call(y, np.ones(n, bool)),
        },
    }
    for k, v in out["strategies"].items():
        v["net_inr_per_1000_orders"] = round(v["net_inr"] / n * 1000, 0)
    return out
