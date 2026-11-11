"""Command-line entry point: `uv run tells ...`."""

from __future__ import annotations

import argparse
import sys

import cleanroom

import tells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tells", description=tells.__doc__)
    parser.add_argument("--version", action="store_true", help="print versions and exit")
    args = parser.parse_args(argv)

    if args.version or argv == []:
        print(f"tells {tells.__version__}, cleanroom {cleanroom.__version__}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
