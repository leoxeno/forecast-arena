import numpy as np

from forecast_arena.data import synthetic
from forecast_arena.tasks import get_task, list_tasks, load_series, series_sha256


def test_generators_are_deterministic_and_finite():
    for name, gen in synthetic.GENERATORS.items():
        a = gen(n_points=300)
        b = gen(n_points=300)
        assert a.shape[0] == 300, name
        assert np.isfinite(a.to_numpy()).all(), name
        assert np.array_equal(a.to_numpy(), b.to_numpy()), name


def test_chaotic_series_are_not_constant():
    for name, gen in synthetic.GENERATORS.items():
        x = gen(n_points=500).iloc[:, 0].to_numpy()
        assert x.std() > 1e-3, name


def test_offline_tasks_load():
    offline = list_tasks(offline_only=True)
    assert {"lorenz-x", "mackey-glass", "sunspots-yearly", "nile"} <= set(offline)
    y = load_series("lorenz-x")
    assert y.ndim == 1 and len(y) == 2000
    assert len(load_series("sunspots-yearly")) == 309
    assert len(load_series("nile")) == 100


def test_task_metadata_and_hash():
    spec = get_task("sunspots-yearly")
    assert spec.freq == "YE" and spec.domain == "astrophysics" and spec.license
    y = spec.load()
    assert series_sha256(y) == series_sha256(y.copy())
