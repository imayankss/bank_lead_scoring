# Data

Nothing in `data/` is committed except this file. Run `python scripts/fetch_mbd.py` to download and verify the data.

| Item | Detail |
|---|---|
| Dataset | MBD-mini, the 10% client sample of the Multimodal Banking Dataset |
| Publisher | Sber AI Lab (`ai-lab` on Hugging Face), pinned revision `4b0d5b4c` |
| Licence | CC BY 4.0. Cite: Mollaev et al., *Multimodal Banking Dataset: Understanding Client Needs through Event Sequences*, arXiv:2409.17587 |
| Clients | 100,224 business clients of a large bank, split into 5 client folds. These are **not** Bank of India customers. |
| Transactions | 38.8M, Jan 2021 to Dec 2022, anonymised, with noise added to amounts and dates |
| Labels | Monthly flags for 4 unnamed products (`target_1..4`), report months Feb 2022 to Jan 2023 |

## How the project uses it

- Timestamps are stored in UTC and converted to Moscow time (UTC+3) before any day or month boundary is applied.
- Only the `targets` and `detail/trx` tables are used; `geo` and `dialog` are not.
- `trans_count` and `diff_trans_date` in `targets` are never used. No transaction window reproduces them, so their source is unknown.
- The amount cap in `src/lead_scoring/data.py` limits the effect of the noise added to amounts.

## Label timing

A `targets` row stamped `mon = M` is read as "purchases made during month M". At a month-end snapshot, the snapshot month's own label is therefore known history, and the prediction window starts the month after. This is the best-supported reading, not a documented fact:

1. The MBD paper shifts the purchase table by one month so labels run February 2022 to January 2023, while transactions end in December 2022.
2. For first-time product-2 buyers, the first transaction of event type 12, 23 or 26 peaks inside the label month, not the month before. These types are 64–93× more common among buyers than non-buyers in that month.
3. `trans_count` increments line up best with the calendar month before `mon` (correlation 0.36, against 0.34 for the same month), which fits rows built as "features up to the end of M−1, purchase during M".

If this reading were wrong in the unsafe direction (a row stamped M meaning purchases in M+1), the snapshot-month flag `hist_*_m0` would leak. The single-feature check in `scripts/validate_propensity.py` stops the run if any feature alone passes 0.80 validation AUC.
