"""Run: python -m pytest -q   (or python tests/test_service.py)"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.scorer import Scorer  # noqa: E402
from src.policy import CALL_THRESHOLD, decide  # noqa: E402

S = Scorer()
EX = json.loads((ROOT / "app" / "static" / "examples.json").read_text())


def test_examples_score_and_order_correctly():
    probs = [S.score(e)["return_probability"] for e in EX]
    assert probs == sorted(probs), probs          # low < elevated < high
    assert all(0 < p < 1 for p in probs)


def test_reasons_present_and_readable():
    out = S.score(EX[2])
    assert out["ok"] and out["reasons"]
    assert all({"factor", "direction"} <= r.keys() for r in out["reasons"])


def test_leak_columns_are_ignored():
    base = S.score(EX[1])["return_probability"]
    leaky = S.score({**EX[1], "last_service_event_type": "REVERSE_PICKUP", "pickup_scheduled_at": "2026-07-10 10:00"})
    assert leaky["return_probability"] == base
    assert "last_service_event_type" in leaky["ignored_fields"]


def test_unknown_customer_still_scores():
    rec = {k: v for k, v in EX[1].items() if k not in ("shield_member", "signup_date", "state")}
    out = S.score({**rec, "customer_id": "NEVER-SEEN"})
    assert out["ok"] and 0 < out["return_probability"] < 1


def test_bad_input_fails_politely():
    out = S.score({"sku": "NOPE"})
    assert out["ok"] is False and any("unknown sku" in e for e in out["errors"])


def test_paise_value_is_corrected():
    rupees = S.score(EX[1])["return_probability"]
    paise = S.score({**EX[1], "order_value_inr": EX[1]["order_value_inr"] * 100})["return_probability"]
    assert abs(rupees - paise) < 1e-9


def test_policy_breakeven():
    assert abs(CALL_THRESHOLD - 45 / (0.35 * 1150)) < 1e-9
    assert decide(0.05)["action"] == "ship_normally"
    assert decide(0.20)["action"] == "call_before_dispatch"
    assert "never hold" in decide(0.9, "Y")["note"]


def test_predictions_file_matches_submission_shape():
    pred = pd.read_csv(ROOT / "predictions.csv")
    assert list(pred.columns) == ["order_id", "score"]
    assert len(pred) == 2096 and pred.order_id.is_unique and pred.score.between(0, 1).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
