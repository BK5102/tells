"""Command-line entry point: `uv run tells ...`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cleanroom
import polars as pl

import tells
from tells.data.sampling import sample_accounts
from tells.data.schema import validate_transactions
from tells.data.tabformer import RAW_FILE, load_tabformer

INTERIM = Path("data/interim")


def _prepare_tabformer(raw: Path, out_dir: Path, small_n: int, seed: int) -> None:
    """Parse the raw CSV once and save Parquet: the full set and a small sample."""
    if not raw.exists():
        raise SystemExit(f"{raw} not found. See data/README.md for the download command.")
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    tx = load_tabformer(raw)
    validate_transactions(tx)
    full = out_dir / "tabformer.parquet"
    tx.write_parquet(full)

    picked = sample_accounts(tx["account_id"].unique(), small_n, seed=seed)
    small = tx.filter(pl.col("account_id").is_in(picked))
    small_path = out_dir / f"tabformer_small_n{small_n}_seed{seed}.parquet"
    small.write_parquet(small_path)

    print(f"{full}: {tx.height:,} transactions, {tx['account_id'].n_unique():,} accounts")
    print(
        f"{small_path}: {small.height:,} transactions, {len(picked)} accounts, "
        f"{int(small['label'].sum()):,} positives"
    )
    print(f"done in {time.perf_counter() - start:.0f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tells", description=tells.__doc__)
    parser.add_argument("--version", action="store_true", help="print versions and exit")
    sub = parser.add_subparsers(dest="command")

    prep = sub.add_parser("prepare", help="convert a raw dataset into standard-schema Parquet")
    prep.add_argument("dataset", choices=["tabformer"])
    prep.add_argument("--raw", type=Path, default=RAW_FILE)
    prep.add_argument("--out", type=Path, default=INTERIM)
    prep.add_argument("--small-n", type=int, default=300, help="accounts in the small sample")
    prep.add_argument("--seed", type=int, default=0)

    run = sub.add_parser("run", help="run an experiment config on the validation period")
    run.add_argument("config", type=Path)
    run.add_argument("--part", default="valid", help="valid (test needs the lock, Day 9)")

    args = parser.parse_args(argv)
    if args.command == "run":
        from tells.config import load_config
        from tells.harness import run_experiment

        result = run_experiment(load_config(args.config), part=args.part)
        print(json.dumps(result.summary, indent=2))
        print(f"saved to {result.run_dir}")
        return 0
    if args.version:
        print(f"tells {tells.__version__}, cleanroom {cleanroom.__version__}")
        return 0
    if args.command == "prepare":
        _prepare_tabformer(args.raw, args.out, args.small_n, args.seed)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
