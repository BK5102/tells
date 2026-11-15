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

INTERIM = Path("data/interim")


# Default small-sample sizes: TabFormer has 6,139 long-lived cards, AMLWorld
# has 515,088 mostly short-lived accounts, so it needs more for a similar
# number of positives.
SMALL_N = {"tabformer": 300, "amlworld": 20_000}


def _prepare(dataset: str, raw: Path | None, out_dir: Path, small_n: int | None, seed: int) -> None:
    """Parse a raw dataset once and save Parquet: the full set and a small sample."""
    if dataset == "tabformer":
        from tells.data.tabformer import RAW_FILE
        from tells.data.tabformer import load_tabformer as load
    else:
        from tells.data.amlworld import RAW_FILE
        from tells.data.amlworld import load_amlworld as load
    raw = raw or RAW_FILE
    small_n = small_n or SMALL_N[dataset]
    if not raw.exists():
        raise SystemExit(f"{raw} not found. See data/README.md for the download command.")
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    tx = load(raw)
    validate_transactions(tx)
    full = out_dir / f"{dataset}.parquet"
    tx.write_parquet(full)

    picked = sample_accounts(tx["account_id"].unique(), small_n, seed=seed)
    small = tx.filter(pl.col("account_id").is_in(picked))
    small_path = out_dir / f"{dataset}_small_n{small_n}_seed{seed}.parquet"
    small.write_parquet(small_path)

    print(f"{full}: {tx.height:,} rows, {tx['account_id'].n_unique():,} accounts")
    print(
        f"{small_path}: {small.height:,} rows, {len(picked)} accounts, "
        f"{int(small['label'].sum()):,} positive rows"
    )
    print(f"done in {time.perf_counter() - start:.0f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tells", description=tells.__doc__)
    parser.add_argument("--version", action="store_true", help="print versions and exit")
    sub = parser.add_subparsers(dest="command")

    prep = sub.add_parser("prepare", help="convert a raw dataset into standard-schema Parquet")
    prep.add_argument("dataset", choices=["tabformer", "amlworld"])
    prep.add_argument("--raw", type=Path, default=None)
    prep.add_argument("--out", type=Path, default=INTERIM)
    prep.add_argument("--small-n", type=int, default=None, help="accounts in the small sample")
    prep.add_argument("--seed", type=int, default=0)

    run = sub.add_parser("run", help="run an experiment config on the validation period")
    run.add_argument("config", type=Path)
    run.add_argument("--part", default="valid", help="valid, or test (needs `tells lock` first)")

    lock = sub.add_parser("lock", help="seal a config before scoring its test period")
    lock.add_argument("config", type=Path)
    lock.add_argument("--note", default="", help="why the plan is being sealed now")

    args = parser.parse_args(argv)
    if args.command == "lock":
        from cleanroom.lock import seal

        from tells.config import load_config
        from tells.harness import LOCKS_DIR

        sealed = seal(load_config(args.config).raw, LOCKS_DIR, note=args.note)
        print(f"sealed {args.config} as {sealed.path}. Commit this file before scoring the test.")
        return 0
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
        _prepare(args.dataset, args.raw, args.out, args.small_n, args.seed)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
