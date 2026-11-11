from tells.data.sampling import sample_accounts

IDS = [f"acct{i}" for i in range(100)]


def test_size_and_membership() -> None:
    picked = sample_accounts(IDS, 10, seed=0)
    assert len(picked) == 10
    assert set(picked) <= set(IDS)


def test_deterministic_and_order_independent() -> None:
    assert sample_accounts(IDS, 10, seed=0) == sample_accounts(list(reversed(IDS)), 10, seed=0)


def test_seed_changes_sample() -> None:
    assert sample_accounts(IDS, 10, seed=0) != sample_accounts(IDS, 10, seed=1)


def test_growing_n_keeps_earlier_picks() -> None:
    # A bigger sample contains the smaller one, so "small" results stay
    # comparable when the sample grows.
    assert set(sample_accounts(IDS, 10, seed=3)) <= set(sample_accounts(IDS, 20, seed=3))


def test_n_larger_than_population_returns_all() -> None:
    assert sorted(sample_accounts(IDS, 500, seed=0)) == sorted(IDS)
