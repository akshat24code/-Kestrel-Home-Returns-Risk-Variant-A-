"""How the recommendation moves if Kestrel's own numbers are off. Run: python -m src.sensitivity"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from src.experiment import SPLIT, make_lr
from src.features import DATA_DIR, build_features, clean_orders

ROOT = Path(__file__).resolve().parent.parent
df = clean_orders(pd.read_csv(DATA_DIR / "train.csv", dtype={"customer_id": str}))
tr, va = df[df.order_placed_at < SPLIT], df[df.order_placed_at >= SPLIT]
p = make_lr().fit(build_features(tr), tr.returned.values).predict_proba(build_features(va))[:, 1]
y = va.returned.values
rows = []
for cost in (600, 1150):
    for prevent in (0.15, 0.25, 0.35, 0.50):
        thr = 45 / (prevent * cost)
        sel = p >= thr
        net = prevent * cost * y[sel].sum() - 45 * sel.sum()
        net_all = prevent * cost * y.sum() - 45 * len(y)
        rows.append({"return_cost": cost, "call_prevents": prevent, "threshold": round(thr, 3),
                     "share_called": round(sel.mean(), 3), "net_per_1000_orders": round(net / len(y) * 1000),
                     "net_if_call_everyone_per_1000": round(net_all / len(y) * 1000)})
t = pd.DataFrame(rows); print(t.to_string(index=False))
(ROOT / "reports" / "sensitivity.json").write_text(json.dumps(rows, indent=2))
