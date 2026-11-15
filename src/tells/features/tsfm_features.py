"""TSFM features: cached embeddings and forecasts, PCA, and surprise scores.

Pipeline for one model (methodology M32 to M34):

1. For every account-day the experiment can touch, cut the account's history
   before midnight (`SequenceIndex`), run the model, and cache embeddings
   (float16) and next-event forecasts on disk in resumable chunks. The cache
   key covers the model version, the data checksum, the sequence code, and the
   context length, so any change recomputes.
2. Fit PCA on account-days in the TRAINING window only (no labels are used)
   and keep the first `k` components. Hundreds of correlated numbers become a
   few columns a tree model can use, and nothing from validation or test
   shapes the projection.
3. Surprise: how far today's transaction is from the model's forecast, for
   amount and for time since the previous event. The forecast uses only the
   history before midnight; the transaction's own amount and gap are known
   when it is scored.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
from cleanroom.provenance import file_sha256

import tells.features.sequences as sequences
from tells.models.tsfm.base import FORECAST_CHANNELS, TSFMAdapter, run_adapter

TSFM_CACHE = Path("data/interim/tsfm")


def feature_names(model: str, k: int) -> list[str]:
    return [
        *(f"{model}_pc{i}" for i in range(k)),
        f"{model}_surprise_amount",
        f"{model}_surprise_gap",
        f"{model}_has_history",
    ]


def _cache_dir(adapter: TSFMAdapter, data_sha: str, context: int, root: Path) -> Path:
    code = file_sha256(sequences.__file__)
    key = f"{adapter.version}|{data_sha}|{code}|{context}"
    return root / f"{adapter.name}_{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def compute_tsfm_cache(
    adapter: TSFMAdapter,
    tx: pl.DataFrame,
    keys: pl.DataFrame,
    data_sha: str,
    context: int = sequences.DEFAULT_CONTEXT,
    chunk: int = 20_000,
    root: Path = TSFM_CACHE,
    log: Callable[[str], None] = print,
) -> Path:
    """Embed every (account_id, day) in `keys`, resuming any chunks already on disk."""
    out = _cache_dir(adapter, data_sha, context, root)
    out.mkdir(parents=True, exist_ok=True)
    keys = keys.select("account_id", "day").unique().sort("account_id", "day")
    meta = {
        "model": adapter.name,
        "version": adapter.version,
        "revision": adapter.revision,
        "data_sha256": data_sha,
        "context": context,
        "n_keys": keys.height,
        "chunk": chunk,
    }
    meta_path = out / "meta.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if previous.get("revision", adapter.revision) != adapter.revision:
            raise ValueError(
                f"cache {out} was built with model revision {previous['revision']}, "
                f"not {adapter.revision}; delete it to rebuild"
            )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    index = None
    n_chunks = (keys.height + chunk - 1) // chunk
    for c in range(n_chunks):
        part = out / f"part_{c:05d}"
        if (part / "done").exists():
            continue
        index = index or sequences.SequenceIndex(tx)
        t0 = time.perf_counter()
        k = keys.slice(c * chunk, chunk)
        seqs = index.build(k, context)
        result = run_adapter(adapter, sequences.to_model_input(seqs))
        part.mkdir(exist_ok=True)
        k.with_columns(has_history=(seqs["length"] > 0).cast(pl.Int8)).write_parquet(
            part / "keys.parquet"
        )
        np.save(part / "emb.npy", result.embeddings.astype(np.float16))
        np.save(part / "fc.npy", result.forecasts.astype(np.float32))
        (part / "done").write_text("ok", encoding="utf-8")
        log(f"{adapter.name} chunk {c + 1}/{n_chunks}: {k.height} account-days "
            f"in {time.perf_counter() - t0:.0f}s")  # fmt: skip
    return out


def _parts(cache: Path) -> list[Path]:
    return sorted(p for p in cache.glob("part_*") if (p / "done").exists())


def fit_pca(
    cache: Path, train_days: Callable[[pl.Series], np.ndarray], k: int, standardize: bool = True
):
    """Mean, scale, and top-k components from training-window account-days with a history.

    With `standardize` (the default since 2026-11-14), every embedding number is
    divided by its training-period standard deviation before PCA, so a handful
    of very large-scale numbers cannot dominate the components. Without it, on
    TTM, one direction held 97% of the variance (methodology M34).
    """
    n, s, ss = 0, None, None
    for part in _parts(cache):
        keys = pl.read_parquet(part / "keys.parquet")
        emb = np.load(part / "emb.npy").astype(np.float64)
        use = train_days(keys["day"]) & (keys["has_history"].to_numpy() == 1)
        x = emb[use]
        if not len(x):
            continue
        s = x.sum(0) if s is None else s + x.sum(0)
        ss = x.T @ x if ss is None else ss + x.T @ x
        n += len(x)
    if n < 2:
        raise ValueError("not enough training account-days with history to fit PCA")
    mean = s / n
    cov = ss / n - np.outer(mean, mean)
    scale = np.sqrt(np.clip(np.diag(cov), 0, None)) if standardize else np.ones_like(mean)
    scale = np.where(scale > 1e-8, scale, 1.0)
    cov = cov / np.outer(scale, scale)
    values, vectors = np.linalg.eigh(cov)
    top = np.argsort(values)[::-1][:k]
    explained = float(values[top].sum() / values.sum())
    return mean, scale, vectors[:, top], explained


def load_tsfm_table(
    cache: Path, model: str, mean: np.ndarray, scale: np.ndarray, components: np.ndarray
) -> pl.DataFrame:
    """Per (account_id, day): PCA components, forecasts, has_history."""
    frames = []
    for part in _parts(cache):
        keys = pl.read_parquet(part / "keys.parquet")
        emb = np.load(part / "emb.npy").astype(np.float64)
        pcs = (((emb - mean) / scale) @ components).astype(np.float32)
        fc = np.load(part / "fc.npy")
        cols = {f"{model}_pc{i}": pcs[:, i] for i in range(components.shape[1])}
        cols.update({f"{model}_fc_{c}": fc[:, j] for j, c in enumerate(FORECAST_CHANNELS)})
        frames.append(
            keys.rename({"has_history": f"{model}_has_history"}).with_columns(
                **{name: pl.Series(v).fill_nan(None) for name, v in cols.items()}
            )
        )
    return pl.concat(frames)


def add_tsfm_features(tx: pl.DataFrame, table: pl.DataFrame, model: str) -> pl.DataFrame:
    """Join model outputs to each transaction by (account_id, day) and add surprise scores.

    Needs the checklist column `secs_since_prev` (time since the previous event).
    """
    gap = (pl.col("secs_since_prev") / 60).log1p()
    out = tx.with_columns(day=pl.col("ts").dt.date()).join(
        table, on=["account_id", "day"], how="left", maintain_order="left"
    )
    return out.with_columns(
        **{
            f"{model}_surprise_amount": pl.col("amount").abs().log1p()
            - pl.col(f"{model}_fc_log_amount"),
            f"{model}_surprise_gap": gap - pl.col(f"{model}_fc_log_gap_minutes"),
            f"{model}_has_history": pl.col(f"{model}_has_history").fill_null(0),
        }
    ).drop("day", f"{model}_fc_log_amount", f"{model}_fc_log_gap_minutes")
