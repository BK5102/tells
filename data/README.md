# data/

Raw data lives in `data/raw/` and is never committed. Only this file and `checksums.sha256` are tracked. Verify downloads with `cd data && sha256sum -c checksums.sha256`.

| Folder | Source | How to get it |
|---|---|---|
| `raw/tabformer/` | [IBM/TabFormer](https://github.com/IBM/TabFormer), Apache-2.0 repo | `curl -L -o data/raw/tabformer/transactions.tgz https://media.githubusercontent.com/media/IBM/TabFormer/main/data/credit_card/transactions.tgz` then `tar -xzf` it. 279 MB compressed, 2.35 GB CSV. |
| `raw/amlworld_hf_mirror/` | Hugging Face mirror `OsamaMIT/IBM-AML-HI-Small` of Kaggle "IBM Transactions for Anti Money Laundering" | Source of all AMLWorld results. Its counts match the published table; comparing its checksum with the official Kaggle file is deferred by the owner. |

Fit results and dataset traps: `docs/data-fit.md`.
