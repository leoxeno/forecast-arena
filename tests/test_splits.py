import pytest

from forecast_arena.splits import describe_splits, generate_rolling_origins


def test_origins_are_monotonic_with_one_test_length():
    splits = generate_rolling_origins(1000, n_origins=15)
    assert len(splits) == 15
    origins = [o for o, _ in splits]
    assert origins == sorted(origins)
    assert len({t for _, t in splits}) == 1
    assert 500 <= origins[0] and origins[-1] <= 850
    assert origins[-1] + splits[-1][1] == 1000


def test_single_origin_is_a_70_30_split():
    assert generate_rolling_origins(1000, n_origins=1) == [(700, 300)]


def test_short_series_raises():
    with pytest.raises(ValueError):
        generate_rolling_origins(20)


def test_describe_splits_columns():
    splits = generate_rolling_origins(400, n_origins=5)
    df = describe_splits(400, splits)
    assert list(df.columns) == ["split", "n_train", "n_test", "train_frac", "test_end"]
    assert (df["test_end"] <= 400).all()
