# Audit of the original version

Audited commit `17a5774` on `main` (2026-10-25), before the move to MBD data. Numbers were measured by re-running that commit's code on its own 500-customer synthetic data, with scripts outside the repo.

| Check | Finding |
|---|---|
| Reported result | ROC AUC 0.913 on 100 random test rows (README, `metrics_summary.csv`) |
| Reproducible? | No. A fresh run gave 0.920 and changed all 39 artifacts; `requirements.txt` was unpinned |
| Split | `train_test_split` on one snapshot (`features.py:74`), described as "historical" |
| Account features | Today's count, balance and status, not values at the snapshot; 66 of 700 accounts opened after it |
| One-column rule | `account_count` alone: AUC 0.912 on the same test rows |
| Without account features | Model AUC 0.833 (95% CI 0.737–0.913), below a recency rule at 0.852 |
| Time split, honest features | Model 0.840 vs recency rule 0.869 |
| Target | "Any transaction next year": 20% for inactive customers vs 89% for active ones |
| Leaderboard | "Baselines" reused model predictions; `business_score` gave the production model +0.05 |
| CLTV | R² −0.37, worse than predicting the mean |
| Precision@10% = 1.0 | Measured on 10 rows |

The rebuild on MBD fixes each of these. See the README's "What I fixed".
