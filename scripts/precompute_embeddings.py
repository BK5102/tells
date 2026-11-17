"""Fill a TSFM embedding cache for a config before running it, without scoring anything.

Embeddings never read labels, so computing them before a plan is sealed does
not touch any result: it only moves the slow part earlier. The account-days
requested are exactly those the harness will request for the same config
(every account-day inside any window), so the later run reads the cache.

Usage:
    uv run python scripts/precompute_embeddings.py configs/<name>.yaml <model>
"""

from __future__ import annotations

import sys

import polars as pl
from cleanroom.provenance import file_sha256

from tells.config import load_config
from tells.features.tsfm_features import compute_tsfm_cache
from tells.models.tsfm import make_adapter


def main(config_path: str, model: str) -> None:
    cfg = load_config(config_path)
    settings = dict(cfg.model.get("tsfm", {}))
    context = int(settings.get("context", 256))
    chunk = int(settings.get("chunk", 20_000))
    tx = pl.read_parquet(cfg.data)
    in_any = pl.lit(False)
    for _, w in cfg.split.windows():
        in_any = in_any | w.mask("ts")
    keys = tx.filter(in_any).select("account_id", day=pl.col("ts").dt.date()).unique()
    adapter = make_adapter(model, **settings.get("adapter", {}).get(model, {}))
    print(f"{model}: {keys.height:,} account-days for {cfg.name}", flush=True)
    path = compute_tsfm_cache(adapter, tx, keys, file_sha256(cfg.data), context, chunk)
    print(f"cache ready: {path}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1], sys.argv[2])
