# data/

Raw data lives in `data/raw/` and is never committed. Only this file and `checksums.sha256` are tracked. Verify downloads with `cd data && sha256sum -c checksums.sha256`.

| Folder | Source | How to get it |
|---|---|---|
| `raw/tabformer/` | [IBM/TabFormer](https://github.com/IBM/TabFormer), Apache-2.0 repo | `curl -L -o data/raw/tabformer/transactions.tgz https://media.githubusercontent.com/media/IBM/TabFormer/main/data/credit_card/transactions.tgz` then `tar -xzf` it. 279 MB compressed, 2.35 GB CSV. |
| `raw/amlworld_hf_mirror/` | Hugging Face mirror `OsamaMIT/IBM-AML-HI-Small` of Kaggle "IBM Transactions for Anti Money Laundering" | Source of all AMLWorld results. Its counts match the published table; comparing its checksum with the official Kaggle file is deferred by the owner. |
| `raw/amlworld_li_hf_mirror/` | LI-Small (v0.2 sealed test) from the Hugging Face mirror `LordNR/AMLGraphX-IBM-AML`, file `IBM-AML-LI-Small.zip`, of the same Kaggle dataset | `curl -L -o data/raw/amlworld_li_hf_mirror/IBM-AML-LI-Small.zip https://huggingface.co/datasets/LordNR/AMLGraphX-IBM-AML/resolve/main/IBM-AML-LI-Small.zip` then `unzip` it (skip `__MACOSX/`). 137 MB zip, 650 MB CSV with 6,924,049 rows, matching the published count. The same mirror's `IBM-AML-HI-Small.zip` holds a `HI-Small_Trans.csv` byte-identical to the HI-Small file above, which supports that the mirror is unmodified. Then `uv run tells prepare amlworld_li`. No model may be trained or scored on it before the v0.2 lock. |

Fit results and dataset traps: `docs/data-fit.md`.
