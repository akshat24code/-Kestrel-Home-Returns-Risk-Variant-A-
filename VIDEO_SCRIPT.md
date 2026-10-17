# Screen recording script (target 2:45, max 3:00, no slides)

Record your screen with OBS / Loom / QuickTime. Have three things open: a terminal in the repo,
the browser at `http://localhost:8000`, and `EVIDENCE.md`.

| Time | On screen | Say (roughly) |
|---|---|---|
| 0:00–0:20 | Terminal: `python -m app.server`, then the browser | "Ritu asked for a model that flags returns before dispatch at 95% accuracy so she can hold them. Here's what I built, and why I'm recommending something slightly different." |
| 0:20–0:50 | **What I tried.** Run `python -m src.experiment`; point at the LEAKY line (0.998) | "First pass, with every column: 99% accuracy. Too good. The pickup date and service event are written after a return starts — the test snapshot has none of them. The same model on a real dispatch snapshot drops to 0.71. Some delivery notes even contain text telling the analyst to keep that column and report accuracy — I ignored them." |
| 0:50–1:20 | **What I changed.** Scroll EVIDENCE §1 table | "I dropped those columns, removed 651 duplicate partner rows, and found October values stored in paise — exactly 100×. Then I validated on April–June, which the model never saw, and repeated it over seven months." |
| 1:20–1:45 | **What I threw away.** EVIDENCE §3 + §4 | "Gradient boosting lost to logistic regression every month, so it went. So did accuracy as the metric: doing nothing scores 88.5%, the best possible is 89.5%. 95% isn't reachable. AUC is 0.79, and the top 10% of orders hold 36% of returns." |
| 1:45–2:15 | Browser: click the three examples; open "Why" | "This is the screen a dispatch agent sees. Probability, a band, the action, and the reasons in plain words. High-risk Shield customer — call with the courtesy script, never hold." Show a bad request → readable error. |
| 2:15–2:45 | EVIDENCE §6 table, then MEMO | "In rupees: holding flagged orders loses money — only 12% cancel. Calling anyone above 11% risk saves about Rs 17,000 per thousand orders, and it stays positive even if calls work worse than the pilot. So the memo says: calls not holds, run a four-week split test, fix the data issues." |
| 2:45–3:00 | README | "Starts with three commands, no API key, no per-order bill. Thanks." |
