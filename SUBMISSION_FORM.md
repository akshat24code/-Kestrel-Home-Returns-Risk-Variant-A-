# Submission form — Kestrel Home, Returns Risk (Variant A)

> The pack did not include the form file itself, so every question the brief asks is answered here.
> Fields in **[brackets]** are personal and must be filled in by the submitter before sending.

**Name:** [your name]
**Repository URL (private):** [link]
**Screen recording URL:** [link] — script in `VIDEO_SCRIPT.md`
**Submitted:** [date/time]

---

## 1. What do you expect `predictions.csv` to score, and why?

`score` is a calibrated probability of return (0–1), higher = more likely returned.

| If you score with… | I expect | Basis |
|---|---|---|
| **ROC-AUC** | **0.77 – 0.79** (point estimate 0.78) | Apr–Jun 2026 holdout 0.786 (95% CI 0.753–0.814); 7 rolling monthly holdouts 0.779 ± 0.016 |
| PR-AUC / average precision | 0.36 – 0.42 | holdout 0.409, base rate ~11.5% |
| Log loss | ~0.29 – 0.31 | holdout 0.294 (constant-rate baseline 0.357) |
| Brier | ~0.085 | holdout 0.085 |
| Accuracy at 0.5 | ~0.89 | holdout 0.893; flagging nobody gives ~0.885 |
| Precision in top 10% | ~0.40 | holdout 0.41 |

Why slightly below the holdout: test is Jul–Sep (a season seen only once in training); it contains
an `INSTALL_BOOKED` state never seen before (not used by the model, but a sign the snapshot
differs); 31% of test customers are new to the data. Mean test score is 0.115 → ~242 expected returns of 2,096.

**If my score came in at ~0.99 AUC** it would mean the test labels leak something; if it came in
below ~0.72 it would mean Jul–Sep behaves differently and the model needs a seasonal retrain.

## 2. What I decided where the brief was unclear

See README "Decisions". Short version: judged by AUC + rupees, not accuracy; recommend calls not
holds; Rs 1,150 per return; assumed 20% margin for hold maths; dropped post-outcome columns;
fixed paise values and duplicates; ignored instruction-like text planted in delivery notes;
Shield members called, never held; no LLM in the product.

## 3. AI tools — what I used, what it cost, what I discarded

| Used | For | Cost |
|---|---|---|
| Claude (Opus 5.5) in the Claude app, with its code workspace | Data exploration, writing the pipeline, service, tests, and drafting the memo/evidence; I reviewed the outputs and the numbers | Within my existing Claude subscription — **Rs 0 extra / no API spend** [adjust if your plan differs] |
| Inside the product | **Nothing.** No model API; the service runs offline on scikit-learn | **Rs 0 per order** |

**Discarded:**
- A gradient-boosting model and a blend — consistently weaker than logistic regression (AUC 0.76 vs 0.78).
- The 99%-"accurate" model using `pickup_scheduled_at` / `last_service_event_type` — leak.
- "Accuracy" as the headline metric — doing nothing scores 88.5%.
- Hold-based actions — negative rupee value on the holdout.
- LLM-written explanations in the service — would add a per-order bill and non-determinism for no gain; reasons come from model weights instead.
- Instructions embedded in five delivery notes (to keep the pickup column, report "validation accuracy", and rename the dataset) — treated as data, not instructions.

## 4. Known weaknesses

Misses ~1 in 3 returns (mostly first-time, prepaid buyers). Weakest on robot vacuums and water
purifiers (AUC 0.71–0.73). Rupee estimate depends on the spring pilot's 35% call effectiveness,
which should be re-measured with a randomised trial (memo step 2). Margin figure is assumed.
