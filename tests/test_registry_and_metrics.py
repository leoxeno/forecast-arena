import numpy as np
import pandas as pd
import pytest

from forecast_arena import (
    Forecaster, ModelCategory, ModelMetadata, compute_metrics, get_model, list_models,
    per_task_table, register_model, summarise, to_markdown,
)


def test_numpy_baselines_are_registered_and_predict():
    assert {"Naive", "SeasonalNaive", "Drift", "Mean"} <= set(list_models())
    y = np.arange(50, dtype=float)
    np.testing.assert_allclose(get_model("Naive").fit(y).predict(3), [49, 49, 49])
    np.testing.assert_allclose(get_model("Drift").fit(y).predict(2), [50, 51])
    np.testing.assert_allclose(get_model("Mean").fit(y).predict(1), [24.5])
    sn = get_model("SeasonalNaive", m=4).fit(y)
    np.testing.assert_allclose(sn.predict(6), [46, 47, 48, 49, 46, 47])


def test_custom_model_registration():
    @register_model
    class LastTwoMean(Forecaster):
        def fit(self, y, X=None, freq=None, **kw):
            self._train_y = self._validate_y(y)
            self._is_fitted = True
            return self

        def predict(self, horizon, X=None, **kw):
            return np.repeat(self._train_y[-2:].mean(), horizon)

        @staticmethod
        def get_metadata():
            return ModelMetadata(name="LastTwoMean", category=ModelCategory.CLASSICAL,
                                 library="test", year=2026, paper="N/A")

    assert "LastTwoMean" in list_models()
    np.testing.assert_allclose(get_model("LastTwoMean").fit([1.0, 3.0, 5.0]).predict(2), [4, 4])


def test_compute_metrics_values():
    y_train = np.array([1.0, 2.0, 3.0, 4.0])
    y_test = np.array([5.0, 6.0])
    m = compute_metrics(y_train, y_test, np.array([5.0, 8.0]))
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(np.sqrt(2.0))
    assert m["mase"] == pytest.approx(1.0)  # naive in-sample MAE is 1
    assert m["valid_points"] == 2


def test_leaderboard_ranks_and_requires_complete_runs():
    rows = []
    for split in range(3):
        rows.append(dict(task="t", model="A", split=split, mae=1.0, rmse=1.0, smape=1.0, mase=1.0,
                         error=None, fit_time_s=0.1))
        rows.append(dict(task="t", model="B", split=split, mae=2.0, rmse=2.0, smape=2.0, mase=2.0,
                         error=None, fit_time_s=0.1))
        rows.append(dict(task="t", model="C", split=split, mae=0.5 if split < 2 else None,
                         rmse=0.5, smape=0.5, mase=0.5, error=None if split < 2 else "boom",
                         fit_time_s=0.1))
    table = per_task_table(pd.DataFrame(rows))
    ranks = dict(zip(table["model"], table["rank"]))
    assert ranks["A"] == 1 and ranks["B"] == 2 and np.isnan(ranks["C"])
    summary = summarise(table)
    assert summary.iloc[0]["model"] == "A" and summary.iloc[0]["wins"] == 1
    md = to_markdown(table[["model", "mae", "rank"]])
    assert md.splitlines()[0].startswith("| model |")


def test_custom_summary_aligns_scores_by_model_name(monkeypatch):
    table = pd.DataFrame({
        "task": ["t", "t", "t"],
        "model": ["B", "C", "A"],
        "rank": [3.0, 1.0, 2.0],
    })
    scores = pd.Series([30.0, 10.0, 20.0], index=["C", "B", "A"])
    monkeypatch.setattr("forecast_arena.leaderboard.composite_score", lambda table: scores)

    summary = summarise(table, rule="custom")

    assert summary["model"].tolist() == ["C", "A", "B"]
    assert summary["score"].tolist() == [30.0, 20.0, 10.0]
