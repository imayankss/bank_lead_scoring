# Strategy: raising AUC and top-10% lift after Checkpoint 2

Date: 2026-10-25. Branch: `feat/propensity-gru-blend`. Status: strategy only. Nothing here is built yet.

Every number below comes from the fit and validation months. The October test month was not touched.

## 1. Where we stand (validation snapshot 2022-07-31)

| Model | AUC | Lift in top 10% |
|---|---:|---:|
| LightGBM, rich features (ships today) | 0.762 | 3.29 |
| GRU, best epoch (3) | 0.741 | 2.99 |
| Target | — | **≥ 3.7** |

- Validation has 100,224 clients and 2,849 positives (base rate 2.84%). The top 10% is 10,022 clients.
- Lift 3.29 puts about 938 positives in the top 10%. Lift 3.7 needs about 1,054, so **about 116 more positives** must move into the top decile.
- The 95% interval on validation lift is about ±0.18. Choose between models by AUC, and report lift with its interval.

## 2. What the data probes found

### F1. The snapshot month's purchases are thrown away

- `data.py` uses the two label months before the snapshot month (`m1`, `m2`) and never the snapshot month itself.
- Clients who bought any product in July 2022 convert at **14.5%** in Aug–Oct, which is 5.1× the base rate. There are 957 of them, and only 375 sit in the GRU's top 10%.
- A crude rule on the GRU's cached validation scores (July buyers ranked first) moves AUC from 0.741 to **0.748** and lift from 2.99 to **3.12**.

Is July's label known on July 31? The evidence says a row stamped `mon = M` records purchases made during month M:

1. The MBD paper shifts the purchase table by one month so that labels run February 2022 to January 2023, while transactions end in December 2022.
2. For first-time product-2 buyers, the first "usage" transaction (event types 12, 23, 26) peaks inside the label month, not the month before.
3. `trans_count` increments line up best with the calendar month *before* `mon` (correlation 0.357 against 0.336 for the same month, a weak signal). This fits rows built as "features up to the end of M−1, purchase during M".

This is the best-supported reading, not a proof. Write the evidence into `data/README.md` before using the flag.

### F2. Purchase-linked event types are missing from LightGBM's vocabulary

- In the purchase month, event types 12, 23 and 26 are **64–93× more common** among product-2 buyers than among non-buyers. One month *before* the purchase, 23 and 26 are still 30–34× more common, which makes them leading signals.
- Event type 26 is 15× more common among product-1 buyers in the purchase month.
- LightGBM's event-type vocabulary is the top 10 by frequency: `[1, 46, 40, 49, 11, 56, 41, 50, 18, 48]`. It misses 12, 23, 26, 52 and 4. The GRU sees them, because its vocabulary has 1,022 values.

### F3. 15% of positives have no transactions at all

| Segment at 2022-07-31 | Clients | 3-month conversion | Positives |
|---|---:|---:|---:|
| No transactions ever before the snapshot | 6,805 | 6.4% (2.2× base) | 434 |
| Dormant (events, but none in the last 365 days) | 812 | 2.0% | 16 |
| Active in the last 365 days | 92,607 | 2.6% | 2,399 |

- These are probably new clients who sign up for a product before transacting.
- The GRU gives every never-transacted client the same padding-token score. On average it ranks inactive clients at the 69th percentile, and never above the 86th, so none reach its top 10%.
- Within the inactive group, GRU AUC is 0.54. No transaction model can rank inside this group. Only product history, dialogs or geo can.

### F4. Repeat purchases are strong but cover few clients

- Buying any product last month gives a 6.5% chance of buying next month, against a 0.93% base rate.
- Only 21% of validation positives had any earlier purchase. History alone can't carry the lift.

### F5. The GRU is too big for MBD-mini

- Validation AUC peaked at epoch 3 while training loss kept falling: the model is memorising the fit months.
- The paper's best RNN (0.819 on full MBD, transactions only) used hidden size 32. Ours uses 128, with 15× fewer clients.
- The GRU never sees product history (F1, F4), which is LightGBM's cleanest signal.

### F6. Dialogs and geo are already on disk

- `data/raw/mbd/archives/detail.tar.gz` holds `dialog/` (0.67 GB) and `geo/` (0.83 GB). Neither is extracted or used.
- In the paper, adding dialogs gives +0.008 to +0.011 AUC across methods, and geo adds about 0. Dialogs cover 46% of clients and include client–manager negotiations, which are sales conversations.

## 3. Strategies, ranked by expected gain per hour

The gains are estimates, not measurements. Each one is confirmed or dropped on validation.

| # | Change | Finding | Expected effect | Cost | Gate |
|---|---|---|---|---|---|
| S1 | Add the snapshot-month purchase flag per product, months since the last purchase, and a purchase count (NaN where the window starts before 2022-02) | F1, F4 | +0.005 to +0.01 AUC, +0.1 to +0.2 lift | 1–2 h plus a feature rebuild | Label-timing evidence in `data/README.md`; leakage rule |
| S2 | Choose each category vocabulary by predictive value on the fit snapshots: for example the 5 most frequent plus the 5 highest-lift codes with at least 200 events. Same 220-column budget | F2 | +0.003 to +0.01 AUC | 1–2 h, same rebuild as S1 | Fit snapshots only |
| S3 | Dialog features: count in the last 30 and 90 days, days since the last dialog, mean embedding reduced to 16 PCA components (PCA fitted on fit-snapshot dialogs only) | F3, F6 | +0.005 to +0.01 AUC; may rank inside the never-transacted group | 3–4 h | Check coverage first; dialog time strictly before the cutoff |
| S4 | GRU rework: feed the S1 history and a "no transactions" flag into the heads; shrink to hidden 32–64; weight decay 0.01; embedding dropout 0.1; LR 3e-4 with cosine decay; at most 5 epochs; each epoch samples one random month-end cutoff per client | F3, F5 | GRU 0.741 → 0.75–0.765 | ~1 h code, ~1 h per config (4 epochs × 13 min) | Epoch chosen on validation, as now |
| S5 | Use the GRU as a LightGBM feature instead of a rank blend: out-of-fold GRU scores over the 5 client folds (train on 4, score the 5th) for the fit snapshots | F5 | +0.003 to +0.01 AUC over the best LightGBM | 5 GRU runs, about 3–4 h overnight on the M2 | Out-of-fold by client fold; the existing snapshot guards |
| S6 | One LightGBM per product, combined as 1 − Π(1 − pₖ) | F2 (product 2 has its own fingerprint) | +0.002 to +0.008 AUC, mostly at the top | 1–2 h | — |
| S7 | Train like you test: give snapshots with full history (Apr–Jul) more weight, or drop Jan–Mar from the final refit if validation agrees | F1 | 0 to +0.005 AUC | < 1 h | Chosen on validation |
| S8 | Top-decile polish: bag 5 seeds; try `lambdarank` grouped by snapshot | — | +0.002 to +0.005 AUC | 1 h | Choose by AUC |
| S9 | Train the GRU on more clients from full MBD (69 GB in total) over the same fit months. This is allowed: different clients, same time windows | F5 | In the paper, the RNN beats aggregates by +0.036 at 1.5M clients | Large download; cloud or very long local runs | Your OK before any spend |
| S10 | CoLES self-supervised pretraining | F5 | In the paper, CoLES alone (0.773) trails aggregates (0.783) | High | Pretrain only on events before each cutoff |

## 4. Edge cases and traps

- **`trans_count` and `diff_trans_date` stay banned.** Their windows are unknown, and their increments don't match any one calendar month well.
- **Anything fitted on all events leaks the future.** PCA, vocabularies, CoLES pretraining and amount caps must only see events before their cutoff.
- **Dates carry noise.** The paper says noise was added to dates, so a purchase fingerprint can land in the month before or after. Use month-level windows, not exact-day features at month boundaries.
- **Ties at the cutoff.** Never-transacted clients share one GRU score. Ties at the top-k cutoff are already counted at their mean rate (commit `2136333`); S1 and S3 give the model ways to split them.
- **Leakage rule still applies.** Any AUC above 0.85, or any single feature above 0.80, means stop and audit. S1 is the change most likely to raise suspicion, so log its single-feature AUC.
- **Drift.** The monthly any-product rate falls from 0.95% (July) to 0.87% (October). Calibration on test will drift more than AUC or lift.
- **The test month stays untouched** until one final scoring run.

## 5. Honest outlook

- If the estimates add up, LightGBM goes from 0.762 to about **0.775–0.795 AUC**, and lift from 3.29 to about **3.5–3.9**.
- 3.7 is reachable but not certain. 15% of positives have no transactions, which caps any transaction-only model; S1 and S3 are the only levers inside that group.
- If S1–S3 together don't reach about 3.5 lift on validation, 3.7 probably needs S9.

## 6. Suggested order

1. S1 and S2 together, in one feature rebuild. **Checkpoint:** validation AUC and lift.
2. S3 (dialogs). **Checkpoint.**
3. S4. If the GRU reaches 0.755 or more, run S5.
4. S6–S8 as polish.
5. S9 only with your OK.

**Checkpoint 1 result (2026-10-26, validation 2022-07-31, best of the 4-setting LightGBM grid):**

| Features | AUC | Top-10% lift |
|---|---:|---:|
| Before S1/S2 | 0.762 | 3.29 |
| S1 + S2 | 0.765 | 3.44 |
| S2 only | 0.759 | 3.31 |
| **S1 only (kept)** | **0.768** | **3.46** |

S2 was reverted: S1 alone beat S1 + S2 on AUC in all four settings. The label-chosen codes (event types 12, 23 and 26) pushed out frequent codes that carried more of the any-product signal. It may still help per-product models (S6). The highest single-feature AUC is 0.645, well under the 0.80 leakage limit.

## 7. Bigger jumps: what 0.85 AUC and 4–5× lift would take

Added 2026-10-26. Measured on the validation snapshot, using the GRU's cached scores.

### How much AUC each lift target needs

This uses a binormal model at the 2.84% base rate, scaled by 0.877 so it matches our two measured points (0.741 → 2.99 and 0.762 → 3.29):

| Lift in top 10% | AUC needed |
|---:|---:|
| 3.7 | ~0.79 |
| 4.0 | ~0.81 |
| 4.5 | ~0.84 |
| 5.0 | ~0.86 |

- The best published result on this dataset is 0.824. It was set on full MBD (15× our clients) with all three data types, on the easier next-month task.
- **On our task (any product, next 3 months, strict time split, MBD-mini), 0.85 AUC or 5× lift at 10% is beyond the published best.** A result like that is more likely a leak than a win, and our own rule stops at 0.85.
- With every lever in section 3, the honest ceiling looks like about 0.79–0.80 AUC and 3.7–4.0× lift.

### The biggest honest jumps on the current task

1. **More clients (S9).** This is the only lever that flipped the ranking in the paper: on full MBD the RNN beats aggregates by +0.036, while on MBD-mini our GRU loses by 0.021. Deep models need data, and we have 10% of it. It is the largest possible jump and also the most expensive: 69 GB, 8 GB of RAM, and likely cloud.
2. **S1–S3 together:** purchase history including the snapshot month, purchase-linked vocabulary, and dialogs.
3. **S5:** GRU scores as LightGBM inputs.

### Reframes that give bigger honest numbers

Each of these is a different metric, not a better model. Each is fine if it matches the business question and the README names it.

| Reframe | Measured now (GRU, validation) | When it's the right metric |
|---|---|---|
| Lift at the real call capacity | Top 1%: **5.13×**; top 5%: 3.57×; top 10%: 2.99× | The bank calls about 1% of clients (1,002 here). The top-1% interval is wide, about ±0.8. |
| Next-month horizon (the paper's protocol) | AUC **0.758**, against 0.741 for 3 months, with no retraining | Monthly campaigns. It is also the protocol the paper's 0.824 uses, so it's the comparable number. |
| One campaign per product | Product 2 with a fingerprint rule: lift **4.56×** at 10% and **21×** at 1% (only 147 buyers) | Product-specific campaigns |
| Score active clients only; send never-transacted ones to an onboarding rule | AUC 0.751 within active clients | Onboarding is handled by another team or channel |

### Traps that would fake 0.85

- Purchase fingerprints (event types 12, 23, 26) from *inside* the label window. They mark the purchase itself.
- `trans_count` and `diff_trans_date`.
- Splitting rows at random across snapshots, so one client's overlapping label windows land in both training and validation.
- Any choice made on the test month.

## Sources

- Mollaev et al., *Multimodal Banking Dataset* ([arXiv 2409.17587](https://arxiv.org/html/2409.17587v2)): task definition, Tables 2–3, RNN settings, modality gains.
- [MBD dataset card](https://huggingface.co/datasets/ai-lab/MBD): full dataset size, field descriptions.
- Babaev et al., *CoLES* ([arXiv 2002.08232](https://arxiv.org/abs/2002.08232)): contrastive pretraining for event sequences.
- Probe scripts (not in the repo): `/private/tmp/claude-501/-Users-mayanksuryavanshi-bank-lead-scoring/39bc1028-13c7-4316-9eb5-54f5b0bcd879/scratchpad/probe1.py` to `probe8.py`.
