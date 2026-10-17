# Kestrel Home — Returns risk at dispatch (Variant A)

Scores an order before dispatch with the probability it will be returned, explains why in plain
English, and recommends an action: **call the customer** when a Rs 45 confirmation call pays for
itself (risk ≥ 11.2%), otherwise ship normally. It never recommends holding an order — see `MEMO.md`.

| File | What |
|---|---|
| `predictions.csv` | Score for each of the 2,096 rows in `test_unlabelled.csv` (calibrated return probability) |
| `MEMO.md` | One page for Ritu |
| `EVIDENCE.md` | Does it work, how often it doesn't, rupees, data problems found |
| `SUBMISSION_FORM.md` | Expected score, decisions, AI use and cost |
| `VIDEO_SCRIPT.md` | Script/shot list for the ≤3 min recording |
| `app/` | The service: `POST /score` endpoint + one screen |
| `src/` | Features, experiments, training, rupee policy |
| `model/model.joblib` | The trained model (committed, so the service starts without the data pack) |
| `reports/` | Raw metrics JSON + a screenshot of the screen |

## Run it (clean machine, no API key needed)

Needs Python 3.11 or newer.

```bash
git clone <this repo> kestrel && cd kestrel
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.server               # → open http://localhost:8000
```

Click one of the three example orders, or fill the form and press **Score this order**.
Use `PORT=9000 python -m app.server` to change the port.

The service makes no network calls and uses no paid API, so there is nothing to configure and
nothing to fail without a key.

### The endpoint

```bash
curl -s -X POST localhost:8000/score -H "Content-Type: application/json" -d '{
  "order_id": "KO2610505", "order_placed_at": "2026-07-01 05:18", "customer_id": "KC106844",
  "sku": "KH-WP-01", "sales_channel": "app", "payment_mode": "prepaid_upi", "discount_pct": 20,
  "qty": 1, "order_value_inr": 9599.2, "promised_delivery_days": 6, "delivery_pincode": "400151",
  "is_gift": "N", "customer_prior_orders": 2, "customer_prior_returns": 0,
  "delivery_note": "Leave with security"}'
```

Returns `return_probability`, `risk_band` (low / elevated / high), `recommended_action`,
`expected_saving_from_call_inr`, up to five `reasons` ("Payment: cod — raises risk"), a Shield
`note` where relevant, and `ignored_fields` (if you send post-dispatch columns such as
`pickup_scheduled_at`, they are ignored and listed). Invalid input gets HTTP 422 with readable
errors. Optional fields `shield_member`, `signup_date`, `state` override the customer lookup.
Also: `GET /health`, `GET /examples`.

### Tests

```bash
python -m pytest -q
```

## Retrain / reproduce the numbers (needs the data pack)

Customer and order data are **not** in this repo (Kestrel ops-policy §10: must not be published).
Only `data/products.csv` is committed. To retrain, copy `train.csv`, `test_unlabelled.csv`,
`customers.csv`, `sample_submission.csv` into `data/`, then:

```bash
python -m src.experiment     # model comparison, leak check  -> reports/experiments.json
python -m src.train          # final model, holdout rupees, predictions.csv -> reports/evaluation.json
python -m src.sensitivity    # what if the cost assumptions are wrong -> reports/sensitivity.json
```

Without `customers.csv` the service still runs; it treats unknown customers as non-Shield unless
the request says otherwise. With the pack in `data/`, Shield status and tenure are looked up.

## Decisions I made where the brief was unclear

1. **Metric.** Not accuracy — 88.5% is free by predicting "no returns". I optimised ranking (AUC)
   and calibration, and judged the result in rupees using the policy's own costs.
2. **Action.** Calls, not holds. A hold only works through the 12% who cancel and also loses good
   orders; a call prevents ~35% of returns for Rs 45 (policy §4, §7). Holds lose money on the holdout.
3. **Return cost Rs 1,150** (policy and finance), not Ritu's Rs 600. Sensitivity to both is in EVIDENCE.
4. **Margin for hold maths = 20% of order value** — an assumption; nothing in the pack gives it.
5. **Dropped `last_service_event_type`, `pickup_scheduled_at`, `source`** — written after the
   outcome, or an ingestion artefact. Also ignored planted text in delivery notes asking for them to be kept.
6. **Fixed the October 2025 values** (÷100) and removed 651 partner-feed duplicates.
7. **Shield:** scored like everyone else (the model needs it — they return twice as often), called
   with a courtesy script, never held.
8. **No LLM inside the product.** Reasons come straight from the model's weights: free per order,
   deterministic, auditable. Farhan asked for no unpriced per-order bills.
9. **`INSTALL_BOOKED`** appears only in the test snapshot; irrelevant because the column is not used.

## Limits

Misses about 1 in 3 returns. Trained on 15 months; Jul–Sep seasonality (fans, heaters) is only
seen once. Re-train monthly. The rupee estimate rests on the spring pilot's 35% — test it (MEMO step 2).
