# data/

Raw data lives in `data/raw/` and is never committed. Only this file and `checksums.sha256` are tracked. Verify downloads with `cd data && sha256sum -c checksums.sha256`.

| Folder | Source | How to get it |
|---|---|---|
| `raw/tabformer/` | [IBM/TabFormer](https://github.com/IBM/TabFormer), Apache-2.0 repo | `curl -L -o data/raw/tabformer/transactions.tgz https://media.githubusercontent.com/media/IBM/TabFormer/main/data/credit_card/transactions.tgz` then `tar -xzf` it. 279 MB compressed, 2.35 GB CSV. |
| `raw/amlworld_hf_mirror/` | Hugging Face mirror `OsamaMIT/IBM-AML-HI-Small` of Kaggle "IBM Transactions for Anti Money Laundering" | Used only for the Phase 0 fit check. Replace with the official Kaggle download (free account) on Day 27 and compare the checksum. |
| `raw/sparkov/` (optional) | Kaggle `kartik2112/fraud-detection` | Free Kaggle account. Run `scripts/check_data_fit.py sparkov` after download. |
| `raw/ieee_hf_mirror/` (not used for experiments) | Hugging Face mirror `aliceczr/ieee-fraud-detection` of Kaggle competition `ieee-fraud-detection` | Used only for the fit check. Not recommended for Tells; see `docs/data-fit.md`. |

Fit results and dataset traps: `docs/data-fit.md`.
