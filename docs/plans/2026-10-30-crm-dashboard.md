# CRM Dashboard: Strategy and Plan

**Goal:** Make the main dashboard a tool a bank's CRM / relationship-manager (RM) team would work from: who to call, what to offer them, why, and what the book is worth. The current results page stays, one click away, as the evidence that the scores can be trusted.

## Strategy

**Audience.** An RM or CRM team lead on Monday morning. A recruiter or interviewer who opens the project should see the product first and the evidence second.

**The four questions the main page answers.**

| RM question | What the page shows | Where it comes from |
|---|---|---|
| Who do I call this week? | A ranked call list with Hot / Medium / Cold tiers | Lead score (propensity × predicted value), already in the pipeline |
| What do I offer them? | Next best product and its probability; all 4 product probabilities | **New:** one 3-month LightGBM per product |
| Why this client? | The 3 strongest reasons, in plain words (e.g. "took Product 3 two months ago", "spend up in the last 90 days") | **New:** LightGBM's built-in SHAP contributions (`pred_contrib`), grouped into readable families |
| What is my book worth? | Clients scored, expected takers in the next 3 months, value at stake, demand by product | Sums over the scored book |

**Honesty rules carried over.** Every number is still measured or labelled:
- Next-best-product quality is measured on the Oct 2022 test month: per-product AUC, and, among clients who took a product, how often the top-ranked product was one they took, against a "recommend the most popular product" baseline.
- The CLTV model loses to "last 90 days again", so the page shows the client's actual last-90-day volume as fact and the predicted value labelled as a model estimate.
- Products are unnamed in MBD, so they are "Product 1–4", renamed in one place (`PRODUCT_LABEL`) if a bank maps them.
- Risk flags are rules, labelled as rules.

**The current book.** The pipeline already scores the 31 Dec 2022 snapshot (labels unknown, as in real life). That snapshot is the call list.

**What ships to the browser.** The top 1,000 clients by lead score (an RM team's realistic quarter), plus summary counts over all 100,224. MBD client IDs are already anonymised hashes; the page shows a short prefix. Static JSON, no server, same Vercel project.

## Design

Same visual system as the results page (Plex fonts, tokens, light/dark), but a working layout rather than an editorial one: a wide table is the centre of the page.

1. **Header:** "Lead Desk", snapshot date, link to "How good are these scores? →" (`/results`).
2. **Book summary strip:** clients scored · Hot / Medium / Cold counts · expected product takers next 3 months · predicted value at stake.
3. **Product demand:** 4 cells, one per product: clients for whom it is the next best product, expected takers. Clicking a cell filters the list.
4. **Call list:** search by client ID, filter by tier and next best product, sort by lead score / propensity / value. Columns: rank, client, tier, lead score, P(any product), next best product (probability), predicted 3-month value, last-90-day volume, days since last activity, flags.
5. **Client panel** (opens on row click): the 4 product probabilities as bars, 6-month activity sparkline, product history, top 3 reasons, flags, and a suggested next step ("Offer Product 3; they last bought Product 1 in Oct").
6. **`/results`:** the existing results page, plus one new section measuring the next-best-product model.

## Tasks

1. **Regenerate the lost local outputs.** `scripts/run_pipeline.py` (≈1 hour). Then `git diff reports/` must show the committed result tables unchanged; if any number moves, stop and report.
2. **Next best product** (`src/lead_scoring/next_product.py`): fit one LightGBM per product on `y3_target_k` with the any-product settings in `models/propensity_choice.json` (no new tuning); measure on the test month → `reports/tables/next_product_test.csv`; score the Dec 2022 book → `data/processed/next_product_scores.csv`. Wire into `run_pipeline` after `score_batch`. Test on the small fixtures.
3. **Reasons** (same module): `pred_contrib=True` from the production propensity booster for the exported clients; sum contributions by feature family; keep the top 3 positive families with a plain-English sentence built from the client's own values.
4. **Export** `crm.json` from `web_export.py`: summary, product demand, top-1,000 clients with all fields above. Test: sorted by lead score, probabilities in [0, 1], next best product = arg-max, summary counts = full-book counts.
5. **Web:** move the results page to `app/results/page.tsx`; new `app/page.tsx` (server) + client components `CallList`, `ClientPanel` in `components/crm/`. No new packages.
6. **README:** screenshot of the Lead Desk at the top; the next-best-product result in Results.
7. **Verify:** pytest, lint, build; both themes; 375px (table scrolls inside its own box, panel becomes full-screen); then commit, PR, merge; Vercel redeploys `main`.

## Out of scope

- Writing back to a CRM (call outcomes, notes, assignment): static demo, no backend.
- Retuning the propensity model or re-choosing settings on the test month.
- Named products or real bank customers.
