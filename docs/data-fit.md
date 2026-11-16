# Data fit check (Phase 0, 2026-11-10)

Tells needs two things from a dataset. First, each account must have a real history before the fraud happens, because a time series model can only spot a change in rhythm if there is a rhythm to read. Second, the data must span enough time for a train period, a gap, and a later test period. Analogy: you cannot notice that a friend is acting out of character if you met them five minutes ago.

Numbers below come from `scripts/check_data_fit.py` unless marked "reported". "Prior events before first positive" counts how many transactions an account had before its first fraud or laundering label. The threshold of 16 is used because Chronos-2 reads sequences in patches of 16 steps ([Savvidi et al. 2026](https://arxiv.org/html/2610.29667v1), Sec. 3.1).

## Summary

| Dataset | Access | Measured? | Accounts | Span | Positive rate | Prior events before first positive (median) | Verdict |
|---|---|---|---|---|---|---|---|
| IBM TabFormer credit card | Free, no account (GitHub LFS) | Yes | 6,139 cards, 2,000 users | 1991 to 2020 | 0.122% of transactions | 1,493 | **Long-history test.** Fits the idea best, but the bank-style model is near the ceiling here (methodology M4). |
| AMLWorld HI-Small | Free Kaggle account | Yes (Hugging Face mirror, row count matches the paper) | 515,088 | 10 usable days | 0.102% of transactions | 9 | **Main test.** Thin history, but the bank-style model is far from perfect here and it is the benchmark the TSFM paper used (methodology M4). |
| Sparkov (Kaggle "fraud-detection") | Free Kaggle account | No. The only public mirror found was truncated at 1,048,576 rows with rounded card numbers, so it was discarded | about 1,000 cards (reported) | 2019 to 2020 (reported) | not measured | not measured | **Set aside.** Likely fits, but a second card simulator adds little variety next to TabFormer. |
| IEEE-CIS Fraud Detection | Free Kaggle account, accept competition rules | Yes (Hugging Face mirror, 590,540 rows, matching the official count) | 199,070 reconstructed ids | 181 days | 3.5% of transactions | 0 | **Not recommended for Tells.** No usable account histories, and labels are copied forward. |
| PaySim | Free Kaggle account | Yes (Hugging Face mirror, 6,362,620 rows) | 6,353,307 | 30 days | 0.129% | 0 | **Dropped.** 99.9% of accounts appear exactly once. |

## TabFormer (primary)

Source: [IBM/TabFormer](https://github.com/IBM/TabFormer), repository license Apache-2.0 (no separate data license is stated). Synthetic, rule-based generator ([Padhi et al. 2021](https://arxiv.org/abs/2011.01843)). Note that the paper says 20,000 users; the released file has 2,000 users and 6,139 cards.

Measured: 24,386,900 transactions; median 2,602 transactions per card; 98.0% of cards have at least 16. Of the 2,752 cards with any fraud, the median card had 1,493 prior transactions before its first fraud and 98.9% had at least 16. Fraud comes in bursts: median 9 fraud transactions per affected card. Account-day cases: 11,268,396, of which 0.109% are positive.

Traps found:

1. **Fraud rate swings by year.** For example 3,835 fraud transactions in 2010 but 55 in 2011, and 255 in 2017 but 2,491 in 2018. The test period must be fixed in the config before any results are seen, and the report must show results per period, not only pooled.
2. **2020 has zero fraud** (data ends in February 2020). Exclude it from test periods.
3. **Unrealistically high exposure.** 44.8% of cards see fraud at some point. Say so in the README: synthetic data is cleaner and denser than reality.
4. **Bursts.** Once a card is compromised, several fraud transactions follow. Catching the first one is what matters in practice, so add a "caught on first fraud day" metric.

Suggested split (to lock on Day 3, not before): train through 2016, gap of 2017, test 2018 to 2019. Test then holds 4,578 fraud transactions.

Checksums: `transactions.tgz` sha256 `e9f589a0958f40d60f81b1a2e8428db86e00c05755caf44fb055827976c0efa2`; `card_transaction.v1.csv` sha256 `68c438319cf27614d5564b7b520814036f0d7e087b3a17d8c15081848e0f02de`.

## AMLWorld HI-Small (second dataset)

Source: [Altman et al. 2023, NeurIPS Datasets and Benchmarks](https://arxiv.org/abs/2306.16424), Kaggle "IBM Transactions for Anti Money Laundering". Measured here on the Hugging Face mirror `OsamaMIT/IBM-AML-HI-Small`, whose row count (5,078,345) and account count (515,088) match Table 1 of [Savvidi et al. 2026](https://arxiv.org/html/2610.29667v1). Comparing with the official Kaggle file's checksum is deferred by the owner; until then all AMLWorld results come from this mirror. Mirror sha256: `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040`.

Each transaction is counted in both the sender's and receiver's history (10,156,690 account events). Median 6 events per account; 38.1% of accounts have at least 16. Of the 6,357 accounts with any laundering, the median had 9 prior events and 36.7% had at least 16.

Traps found:

1. **End-of-data tail.** After 2022-09-10, only 2,216 account events remain (1,108 transfers, each counted for sender and receiver) and 1,310 of those events are laundering (about 60%, against about 0.1% before). This is a generator artifact. A test period at the end of the data would let a model score well by learning "late timestamp means laundering". **Truncate to 2022-09-01 through 2022-09-10** before splitting.
2. **Only 10 usable days.** The gap must be measured in hours, not weeks, and the test period is 2 to 3 days. Bootstrap by account, not by day.
3. **Weekly pattern.** Weekend days have about 40% of weekday volume. Do not let the test period be all weekend.
4. **Label unit.** Labels are per transaction. The TSFM paper scores transactions; Tells also reports account-day cases (see `docs/decisions.md`).

HI-Medium (16 days, about 32 million transactions, reported) may suffer from the same tail and was not measured.

## Sparkov (optional, not measured)

Kaggle dataset "Credit Card Transactions Fraud Detection Dataset" (kartik2112), generated with the Sparkov tool, about 1,000 customers and 800 merchants, 2019-01-01 to 2020-12-31, CC0 per the [Amazon fraud dataset benchmark](https://github.com/amazon-science/fraud-dataset-benchmark) listing. Not verified. It contains synthetic names and addresses; drop those columns in the loader so nothing resembling personal information enters features.

## IEEE-CIS (measured 2026-11-10, not recommended)

Kaggle competition "IEEE-CIS Fraud Detection" (Vesta). Free with a Kaggle account after accepting the competition rules. Measured on the Hugging Face mirror `aliceczr/ieee-fraud-detection`, `train_transaction.csv`: 590,540 rows (matches the reported count), 20,663 fraud (3.5%), 181 days. Mirror sha256: `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642`.

There is no account id, so one must be rebuilt. Two reconstructions were measured:

| Reconstruction | "Accounts" | Median transactions per account | Median prior events before first fraud | Share of fraud accounts with at least 16 prior | Share of later transactions that are fraud, after an account's first fraud |
|---|---|---|---|---|---|
| Standard: `card1` + `addr1` + implied card start day (`day - D1`); 66,794 rows dropped for missing parts | 199,070 | 1 | 0 | 0.1% | 66.0% |
| Loose: `card1`, `card2`, `card3`, `card5`, `addr1` | 42,946 | 2 | 3 | 25.8% | 6.9% |

What this means:

1. **With the standard id, there is no history.** 58% of ids appear exactly once, and the median fraud id has zero earlier transactions. A model that reads an account's past has nothing to read.
2. **The loose id manufactures history by merging strangers.** `card1` to `card5` describe card attributes such as issuer and card type, not a person, so 590,540 transactions collapse into 42,946 "accounts". Their "history" mixes different cardholders, which is exactly the signal Tells must not fake.
3. **Labels are copied forward.** Under the standard id, 66.0% of transactions after an id's first fraud are also fraud, against a 3.5% base rate. This matches community reports that Vesta labeled later transactions linked to a fraudulent card as fraud (not confirmed from an official source). A model can score well by learning "this id had fraud before", which is a labeling shortcut, not a behavioral tell.
4. **Short and unlabeled test set.** Only about 6 months are labeled; the competition test labels were never released.

IEEE-CIS remains a good dataset for transaction-level tabular fraud models. It is a poor fit for the specific Tells question, which needs real per-account histories.

## Ranking for Tells (best to worst)

| Rank | Dataset | Fit to the idea (real account history) | Realism | Main problem | Cost of handling it |
|---|---|---|---|---|---|
| 1 | TabFormer | Excellent: median 1,493 prior transactions | Synthetic | Fraud rate swings by year | Low: lock the test years in the config, report per year |
| 2 | Sparkov | Likely good (reported 1,000 cards over 2 years), unmeasured | Synthetic | Unverified, one generator | Low: run the fit script after a Kaggle download |
| 3 | AMLWorld HI-Small | Weak: median 9 prior events | Synthetic | End-of-data tail; only 10 days | Low: one filter line; the tail is a good story ("the fit check caught a generator artifact") |
| 4 | IEEE-CIS | None without merging strangers | Real, anonymized | No account id, labels copied forward | High: no fix makes the histories real |
| 5 | PaySim | None | Synthetic | 99.9% of accounts appear once | Not fixable |

## PaySim (dropped)

Measured on the Hugging Face mirror `theman10/paysim` (6,362,620 rows, matching the Kaggle row count): 6,353,307 distinct originating accounts, 99.9% appear once, and every fraudulent account has 0 prior events. There is no account history to read, so it cannot test the Tells idea. The mirror copy was deleted after measuring.
