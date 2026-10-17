# Memo — Returns risk before dispatch

**To:** Ritu Deshpande, Head of D2C Operations
**Cc:** Farhan Sheikh, Meenal Joshi, Tanmay Kulkarni
**From:** Kabir Nanda's team · 17 Oct 2026

## The decision

**Use the model, but call the customer instead of holding the order.** Every evening the model
scores the next day's orders; the outsourced team phones the risky ones before dispatch
("confirming your Robot Vacuum Max to Adyar — is that right?"). Nothing is held. Shield members
get the same call with the service-courtesy script.

Holding loses money. Only 12 in 100 held customers cancel, so a hold stops very few returns —
and it also cancels good orders. On last quarter's orders, holding what the model flagged would
have **lost** money; calling the same kind of customers **saves** money. Your spring pilot is why:
a call prevents about a third of returns for Rs 45.

## The number

**95% accuracy is not reachable on this data — by anyone.** Returns are about 11 in 100 orders, so
a model that predicts "nobody returns" already scores 88.5%. The best our model reaches on that
measure is 89.5%. Please don't let the board judge this on accuracy; it rewards doing nothing.

The number to report is: **the riskiest 10% of orders contain about 36% of all returns, and 4 in 10
of them come back** (vs 1 in 9 for an average order). Calling everyone above 11% risk — about a
third of orders — reaches two-thirds of returns. Tested on April–June orders the model had never
seen. It will still miss 1 in 3 returns: buyers who give no sign at checkout.

One more correction: Tanmay's export contains columns filled in *after* a return starts. A model
built on them shows 99% "accuracy" and is useless at the warehouse door. Ours doesn't use them.

## The rupees

Using finance's figure (Rs 1,150 a return, not Rs 600), on last quarter's orders:

| Per 1,000 orders shipped | |
|---|---|
| Returns today | ~115, costing ~Rs 1.32 lakh |
| Orders called | ~310, costing ~Rs 14,000 |
| Returns avoided | ~27 |
| **Net saving** | **~Rs 17,400** (about 13% of return cost) |
| Holding flagged orders instead | saves nothing; loses up to Rs 13,000 |

Multiply by monthly orders for the annual figure. The model itself has **no per-order bill** — it
runs on a laptop, no paid AI service (for Farhan). The only running cost is the calls.

The saving depends on the pilot's "calls stop 35% of returns". If it is really 25%, the saving
roughly halves; at 15% it is small but still positive. Calling *everyone* only pays if the pilot
number is exactly right — the model's job is telling you who **not** to call.

## What to do next week

1. **Do not switch on holds.** Tell the board the target is "returns cost down", not "accuracy".
2. **Start calls on half the flagged orders** (score ≥ 11%, about 1 in 3 orders) for four weeks, and leave the other half alone. That measures what a call really saves, rather than trusting the spring pilot.
3. **Include Shield members in calls, never in holds** (Meenal) — they are 22% of orders but over 40% of the risky ones.
4. **Ask Tanmay to fix the October payment-gateway values** (stored 100× too high), remove partner-feed duplicates, and look into planted text in delivery notes addressed to "data teams".
5. **Ask Farhan for the gross margin per product**, so the hold decision can be closed with real numbers.
