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
