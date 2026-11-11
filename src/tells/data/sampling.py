"""Deterministic account sampling for small, fast experiment subsets."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def _rank(account_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{account_id}".encode()).hexdigest()


def sample_accounts(account_ids: Iterable[str], n: int, seed: int = 0) -> list[str]:
    """Pick `n` accounts by hashing each id with the seed and keeping the lowest hashes.

    The result depends only on the set of ids and the seed (not their order or
    the library version), and a larger `n` always contains a smaller one.
    """
    unique = set(account_ids)
    return sorted(unique, key=lambda a: _rank(a, seed))[:n]
