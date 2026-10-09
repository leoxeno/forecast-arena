<h1 align="center">forecast-arena</h1>

<p align="center">
<b>One rolling-origin protocol for every forecaster.</b><br/>
Statistical, machine-learning, deep and foundation models, scored side by side on chaotic systems, climate, astrophysics, ecology and macroeconomic series.
</p>

<p align="center">
<a href="https://github.com/leoxeno/forecast-arena/actions/workflows/ci.yml"><img src="https://github.com/leoxeno/forecast-arena/actions/workflows/ci.yml/badge.svg" alt="ci"/></a>
<img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="python"/>
<img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="license"/>
<img src="https://img.shields.io/badge/wrappers-170%20models%2C%2034%20libraries-orange" alt="models"/>
<img src="https://img.shields.io/badge/tasks-20%20series%2C%207%20domains-purple" alt="tasks"/>
</p>

---

## Why this exists

Benchmark suites for time-series models are usually maintained by the teams that train the models. forecast-arena is an evaluation harness maintained apart from any model vendor: a naive forecast, a gradient-boosted lag model, a per-origin-trained N-BEATS and a zero-shot foundation model all receive the same expanding-window origins, the same test length and the same four metrics; every baseline runs with the defaults of its own library; and the leaderboard leaves a model unranked on any task where it failed on any origin.

It began as the evaluation harness for AION, a geometric forecaster for nonlinear systems described in a manuscript in preparation, which had to be compared with every available baseline on seventeen tasks under one protocol. The method lives in a separate codebase. This repository holds the harness, the baselines and the task atlas, so that any forecaster can be put through the same test.

## What you get

| Piece | What it does |
|---|---|
| **Protocol** | 15 expanding-window origins spaced evenly between 50% and 85% of the series, one constant test length, MAE, RMSE, sMAPE and MASE per origin, mean over origins, rank within task. |
| **Model zoo** | 170 wrappers over 34 libraries behind one `Forecaster` base class, from four pure-NumPy baselines to fifteen time-series foundation models. |
| **Task atlas** | 20 series: six chaotic systems generated at load time, three bundled with statsmodels, eleven fetched from public providers with the source URL and licence stored on the task. |
| **Records** | One JSONL line per (task, model, origin) with the data hash, the protocol, timings, parameters and the forecast itself. Runs resume from disk and merge across machines. |
| **Statistics** | Diebold-Mariano tests, block-bootstrap confidence intervals and Friedman-Nemenyi comparisons on the saved records. |
| **M-competitions** | M3 and M4 loaders, the Naive2 reference and OWA aggregation in the order the competitions specify. |

## Quickstart

```bash
git clone https://github.com/leoxeno/forecast-arena && cd forecast-arena
pip install -e ".[stats,boosting]"        # core + statsforecast + xgboost/lightgbm
python examples/01_quickstart.py --origins 5
```

Twelve CPU models, two tasks, five rolling origins, about thirty seconds on a laptop. The output below is the actual run.

**lorenz-x** (Lorenz 1963 attractor, 2,000 points, 300-step horizon, mean over 5 origins)

| rank | model | mae | rmse | mase | n_errors | fit_time_s |
|---|---|---|---|---|---|---|
| 1 | KNN | 4.732 | 6.491 | 14.12 | 0 | 0.0031 |
| 2 | ExtraTrees | 4.824 | 6.521 | 14.37 | 0 | 0.128 |
| 3 | XGBoost | 6.335 | 8.034 | 18.82 | 0 | 0.107 |
| 4 | Ridge | 6.93 | 8.4 | 20.62 | 0 | 0.0079 |
| 5 | Mean | 7.608 | 8.92 | 22.61 | 0 | 0 |
| 6 | SeasonalNaive | 10.96 | 12.49 | 32.65 | 0 | 0 |
| 7 | Naive | 11.13 | 12.63 | 33.16 | 0 | 0 |
| 8 | StatsForecastAutoTheta | 11.47 | 12.96 | 34.16 | 0 | 0.124 |
| 9 | ARIMA | 11.69 | 13.13 | 34.82 | 0 | 1.66 |
| 10 | Drift | 12.03 | 13.51 | 35.85 | 0 | 0 |
| 11 | StatsForecastAutoETS | 15.56 | 17.06 | 46.24 | 0 | 0.063 |
| 12 | ExponentialSmoothing | 70.23 | 81.8 | 208.6 | 0 | 0.164 |

**sunspots-yearly** (1700 to 2008, 309 points, 47-year horizon, mean over 5 origins)

| rank | model | mae | rmse | mase | n_errors | fit_time_s |
|---|---|---|---|---|---|---|
| 1 | XGBoost | 22.78 | 30.88 | 1.381 | 0 | 0.041 |
| 2 | ExtraTrees | 31 | 40.51 | 1.877 | 0 | 0.040 |
| 3 | ARIMA | 31.04 | 38.33 | 1.892 | 0 | 1.35 |
| 4 | Ridge | 31.08 | 39.85 | 1.891 | 0 | 0.0006 |
| 5 | KNN | 32.06 | 40.69 | 1.941 | 0 | 0.0005 |
| 6 | Mean | 34.38 | 42.42 | 2.09 | 0 | 0 |
| 7 | Naive | 39 | 47.07 | 2.379 | 0 | 0 |
| 7 | SeasonalNaive | 39 | 47.07 | 2.379 | 0 | 0 |
| 9 | Drift | 39.26 | 47.13 | 2.396 | 0 | 0 |
| 10 | ExponentialSmoothing | 468.5 | 537.5 | 27.55 | 0 | 0.144 |
|  | StatsForecastAutoETS |  |  |  | 2 | 0.005 |
|  | StatsForecastAutoTheta |  |  |  | 2 | 0.009 |

On the Lorenz attractor over a 300-step horizon the nearest-neighbour lag model leads, which is Lorenz's own method of analogues, and the training mean beats every extrapolating classical model, because a bounded chaotic signal punishes any trend. On the sunspots, two automatic statsforecast models returned all-NaN forecasts on the last two origins; the record carries the error, the model is unranked for that task, and the leaderboard says so in the `n_errors` column.

## Chaos versus the model zoo

`examples/02_chaos_vs_foundation.py` runs a cross-section of the registry on three generated chaotic series and the monthly sunspot number (3,326 points from SILSO): the NumPy floor, lag-window learners, the Legendre-polynomial state-space reconstruction with local prediction, two automatic classical models, N-BEATS trained from scratch at every origin (darts, default epochs) and Chronos-Bolt used zero-shot (the `tiny` checkpoint). Five origins, CPU only, about twenty minutes, most of it the Legendre cross-validation and the N-BEATS training. The tables are the actual run.

**mackey-glass** (tau = 17, 2,000 points, 300-step horizon)

| rank | model | mae | mase | n_errors | fit_time_s |
|---|---|---|---|---|---|
| 1 | GibsonLegendre | 0.04726 | 1.693 | 0 | 49.72 |
| 2 | KNN | 0.0526 | 1.886 | 0 | 0.0033 |
| 3 | ExtraTrees | 0.05418 | 1.942 | 0 | 0.118 |
| 4 | NBEATS | 0.07718 | 2.766 | 0 | 26.89 |
| 5 | Chronos2 | 0.1319 | 4.725 | 0 | 0 |
| 6 | Ridge | 0.1523 | 5.459 | 0 | 0.0211 |
| 7 | StatsForecastAutoARIMA | 0.2023 | 7.253 | 0 | 0.414 |
| 8 | Naive | 0.2445 | 8.77 | 0 | 0 |
| 9 | StatsForecastAutoTheta | 0.2455 | 8.805 | 0 | 0.113 |
| 10 | Drift | 0.2648 | 9.499 | 0 | 0 |

**thomas-x** (Thomas attractor, 2,000 points, 300-step horizon)

| rank | model | mae | mase | n_errors | fit_time_s |
|---|---|---|---|---|---|
| 1 | GibsonLegendre | 0.5566 | 7.369 | 0 | 48.52 |
| 2 | ExtraTrees | 0.6127 | 8.122 | 0 | 0.128 |
| 3 | KNN | 0.6205 | 8.233 | 0 | 0.0034 |
| 4 | Ridge | 0.9069 | 12.04 | 0 | 0.0274 |
| 5 | Chronos2 | 0.9849 | 13.06 | 0 | 0 |
| 6 | NBEATS | 1.075 | 14.27 | 0 | 25.8 |
| 7 | Naive | 1.719 | 22.72 | 0 | 0 |
| 8 | StatsForecastAutoTheta | 1.731 | 22.88 | 0 | 0.110 |
| 9 | Drift | 1.836 | 24.26 | 0 | 0 |
| 10 | StatsForecastAutoARIMA | 6.334 | 85.41 | 0 | 0.519 |

**henon-x** (Henon map, 2,000 points, 300-step horizon)

| rank | model | mae | mase | n_errors | fit_time_s |
|---|---|---|---|---|---|
| 1 | Chronos2 | 0.5769 | 0.5881 | 0 | 0 |
| 2 | StatsForecastAutoTheta | 0.5868 | 0.5982 | 0 | 0.259 |
| 3 | StatsForecastAutoARIMA | 0.5935 | 0.6049 | 0 | 0.927 |
| 4 | Ridge | 0.5943 | 0.6057 | 0 | 0.0203 |
| 5 | ExtraTrees | 0.6147 | 0.6265 | 0 | 0.117 |
| 6 | GibsonLegendre | 0.617 | 0.6288 | 0 | 49.67 |
| 7 | KNN | 0.6304 | 0.6425 | 0 | 0.004 |
| 8 | NBEATS | 0.6887 | 0.702 | 0 | 28.54 |
| 9 | Naive | 0.7731 | 0.7876 | 0 | 0 |
| 10 | Drift | 0.8554 | 0.8712 | 0 | 0 |

**sunspots-monthly** (1749 to 2026, 3,326 points, 499-month horizon)

| rank | model | mae | mase | n_errors | fit_time_s |
|---|---|---|---|---|---|
| 1 | Ridge | 49.6 | 2.542 | 0 | 0.037 |
| 2 | ExtraTrees | 52.06 | 2.668 | 0 | 0.321 |
| 3 | GibsonLegendre | 52.32 | 2.678 | 0 | 67.24 |
| 4 | KNN | 53.87 | 2.76 | 0 | 0.0055 |
| 5 | NBEATS | 59.24 | 3.036 | 0 | 47.39 |
| 6 | Chronos2 | 64.19 | 3.289 | 0 | 0 |
| 7 | Naive | 72.34 | 3.703 | 0 | 0 |
| 8 | Drift | 78.84 | 4.035 | 0 | 0 |
|  | StatsForecastAutoARIMA |  |  | 1 | 0.716 |
|  | StatsForecastAutoTheta |  |  | 1 | 1.182 |

**Across the four tasks**

| model | tasks | mean_rank | median_rank | wins | podiums | worst_rank |
|---|---|---|---|---|---|---|
| GibsonLegendre | 4 | 2.75 | 2 | 2 | 3 | 6 |
| ExtraTrees | 4 | 3 | 2.5 | 0 | 3 | 5 |
| Ridge | 4 | 3.75 | 4 | 1 | 1 | 6 |
| KNN | 4 | 4 | 3.5 | 0 | 2 | 7 |
| Chronos2 | 4 | 4.25 | 5 | 1 | 1 | 6 |
| NBEATS | 4 | 5.75 | 5.5 | 0 | 0 | 8 |
| StatsForecastAutoTheta | 3 | 6.33 | 8 | 0 | 1 | 9 |
| StatsForecastAutoARIMA | 3 | 6.67 | 7 | 0 | 1 | 10 |
| Naive | 4 | 7.75 | 7.5 | 0 | 0 | 9 |
| Drift | 4 | 9.25 | 9.5 | 0 | 0 | 10 |

On the two continuous chaotic flows the state-space reconstruction leads and the lag-window learners follow within a few percent. On the Henon map, whose one-step rule is a quadratic, the zero-shot foundation model takes first place, one percent ahead of an automatic Theta model, with eight models inside a 7% band. On the sunspots a ridge regression on lagged values beats every deep and foundation entrant, and the two automatic statsforecast models are unranked after one non-finite origin each. A model's fit time is part of the record, so the cost of each rank is visible beside it.

## The protocol

```
series  |------------------------------------------------------------------>| time
origin  1   train ######################|  test ..........
origin  2   train ########################|  test ..........
origin  3   train ##########################|  test ..........
  ...
origin 15   train ######################################|  test ..........
                 50% <----- origins spaced evenly -----> 85%   test = last 15%
```

* Origins are expanding windows: training always starts at the first observation.
* The test length is fixed at the remainder after the largest training set, so every origin forecasts the same number of steps and errors are comparable across origins.
* Every model refits from scratch at every origin. Zero-shot models receive the training window as context.
* Metrics per origin: MAE, RMSE, sMAPE and MASE (scaled by the in-sample seasonal naive error). The task score is the mean over origins; the rank is the position of that mean within the task.
* A model that errors, returns a short forecast or returns non-finite values on any origin is unranked on that task. The error string is in the record.
* The same seed is set before every evaluation (Python, NumPy and, when present, PyTorch).

With one origin the split is the familiar 70/30. `Protocol(n_origins=30)` or any other design is one argument away, and every record stores the protocol it was produced under.

## Python API

```python
from forecast_arena import Protocol, run_arena, per_task_table, summarise, to_markdown

df = run_arena(
    tasks=["mackey-glass", "sunspots-monthly"],
    models=["Naive", "Ridge", "StatsForecastAutoARIMA", "NBEATS", "Chronos2"],
    protocol=Protocol(n_origins=15),
    output_dir="results/demo",
)
table = per_task_table(df)                 # mean metrics and rank per (task, model)
print(to_markdown(table[table.task == "mackey-glass"]))
print(to_markdown(summarise(table)))       # mean rank, wins, podiums across tasks
```

Lower-level pieces are importable on their own: `generate_rolling_origins` for the splits, `run_single_model` for one fit, `compute_metrics` for the four errors, `load_series` for any task.

## Command line

```bash
forecast-arena tasks --offline                      # the atlas, no downloads needed
forecast-arena models --library statsforecast       # what is registered in this environment
forecast-arena run --tasks lorenz-x,nile --models Naive,Ridge,XGBoost --origins 15 --out results/run1
forecast-arena run --tasks sunspots-monthly --library darts --origins 5
forecast-arena leaderboard results/run1              # per-task tables
forecast-arena leaderboard results/run1 --summary    # one row per model across tasks
```

## Add a model

A forecaster needs `fit`, `predict` and `get_metadata`. Register it and it is addressable by name from the API and the CLI. The core of `examples/03_add_your_model.py` is this:

```python
import numpy as np
from forecast_arena import Forecaster, ModelCategory, ModelMetadata, register_model

@register_model
class DampedDrift(Forecaster):
    def fit(self, y, X=None, freq=None, **kwargs):
        self._train_y = self._validate_y(y)
        self._is_fitted = True
        return self

    def predict(self, horizon, X=None, **kwargs):
        y = self._train_y
        phi = float(self.params.get("phi", 0.9))
        slope = (y[-1] - y[-20]) / 19
        return y[-1] + slope * np.cumsum(phi ** np.arange(1, horizon + 1))

    @staticmethod
    def get_metadata():
        return ModelMetadata(name="DampedDrift", category=ModelCategory.CLASSICAL,
                             library="numpy", year=2026, paper="N/A")
```

Optional hooks: `get_param_space()` declares tunable parameters for `forecast_arena.forecasters.tuning` (random, grid and Bayesian search over the same `ParamSpace` objects); `predict_quantiles()` exposes probabilistic output; `get_diagnostics()` returns anything the fitted model knows about itself (AIC, coefficients, convergence) into the record.

## Add a task

```python
import numpy as np
from forecast_arena import register_csv_task, register_task, TaskSpec

# Your own file, never committed: the loader reads it at run time.
register_csv_task("my-series", "data/raw/my_series.csv", column="value",
                  description="Daily widget demand", domain="retail", freq="D",
                  source="internal ERP export", license="private", seasonality=7)

# Or any callable that returns a 1-D array.
register_task(TaskSpec(id="noisy-sine", description="Sine plus noise", domain="synthetic",
                       freq="D", loader=lambda: np.sin(np.arange(2000) / 20) + 0.1 * np.random.default_rng(0).normal(size=2000),
                       source="generated", license="Apache-2.0"))
```

## Model zoo

Wrappers import their libraries at fit time, so the registry always lists every wrapper; `forecast-arena models` marks which ones have their library installed, and a model whose library is missing is recorded as an error on each origin. Install extras by family: `pip install -e ".[stats,boosting,deep,foundation,extras]"` or `".[all]"`.

| Family | Libraries (wrappers) | Examples |
|---|---|---|
| Reference baselines | numpy (4) | Naive, SeasonalNaive, Drift, Mean |
| Classical and automatic | statsmodels (4), statsforecast (25), prophet (1), orbit (3), sktime (10) | ARIMA, ETS, AutoARIMA, AutoETS, AutoTheta, MSTL, TBATS, Croston, Prophet |
| Lag-window machine learning | sklearn (14), xgboost, lightgbm, catboost, mlforecast, ngboost, mapie, tsfresh, tslearn, fforma, flaml, autogluon, autots | Ridge, KNN, ExtraTrees, HistGradientBoosting, XGBoost, LightGBM, CatBoost |
| Deep learning, trained per origin | darts (24), neuralforecast (33), gluonts (6), pytorch-forecasting (4), tsai (12) | N-BEATS, N-HiTS, TiDE, TFT, DeepAR, PatchTST, TimesNet, iTransformer, xLSTM |
| Foundation models, zero-shot | chronos-forecasting, timesfm, uni2ts, granite-tsfm, momentfm, lag-llama, transformers, toto-ts (15) | Chronos (T5), Chronos2 (the Chronos-Bolt checkpoints), TimesFM 2.5, Moirai, Moirai-MoE, TTM, MOMENT, Lag-Llama, Timer, Time-MoE, Toto |
| State-space reconstruction | pyEDM (2), built in (1) | Simplex projection, S-map, Legendre-polynomial embedding with local prediction |
| Behind a flag | pypots (18) | set `FORECAST_ARENA_LOAD_PYPOTS=1`; excluded from the default sweep by the speed audit |

`forecast_arena/benchmarks/exclusions.py` records which registered models the default sweep leaves out and why (slow duplicates of a faster model, libraries whose forecasters failed a sanity audit, multivariate-only models), so that "every model" has a written definition; `get_active_models()` returns the resulting roster for the current environment.

## Task atlas

| task | domain | freq | source | licence |
|---|---|---|---|---|
| lorenz-x, rossler-x, thomas-x, mackey-glass, henon-x, logistic | chaos | D (index) | generated at load time from fixed initial conditions | Apache-2.0 |
| sunspots-yearly | astrophysics | YE | statsmodels (bundled) | public domain |
| sunspots-monthly | astrophysics | ME | WDC-SILSO, Royal Observatory of Belgium | CC BY-NC 4.0 |
| nile | hydrology | YE | statsmodels (bundled) | public domain |
| co2-monthly-1958-2001 | climate | ME | statsmodels (bundled) | public domain |
| co2-mauna-loa | climate | ME | NOAA Global Monitoring Laboratory | public domain |
| enso-nino34 | climate | ME | NOAA PSL | public domain |
| nao, pna, ao, aao | climate | ME | NOAA Climate Prediction Center | public domain |
| global-temp | climate | YE | NASA GISTEMP v4 | public domain |
| fed-funds | macro | ME | Federal Reserve Board via FRED | public domain |
| wti | finance | D | US EIA via FRED | public domain |
| lynx | ecology | YE | Brockwell and Davis (1991) via Rdatasets | public data |

Fetched series are downloaded on first use into `~/.cache/forecast-arena` (or `FORECAST_ARENA_CACHE`) and are never redistributed by this package. `forecast-arena tasks` prints the full table with descriptions.

## Reproducibility

* Every record stores `data_sha256`, the full `protocol`, the package version, the model parameters, fit and predict times, and the forecast vector.
* Records are appended line by line, so a crash loses at most one evaluation; the next run resumes from what is on disk.
* Several machines can run disjoint `--models` lists into the same `--out`; the leaderboard reads all files and keeps the latest record per (task, model, origin).
* `FORECAST_ARENA_FAST_MODE=1` cuts training to a few epochs in the neuralforecast, gluonts and pytorch-forecasting wrappers, which CI uses; darts models keep their own defaults, and the full sweep runs without the flag.

## Statistics on the records

`forecast_arena.benchmarks.robustness` works on the per-origin errors that the arena saves: `diebold_mariano_test` for a pair of models, `bootstrap_ci` for a confidence interval on a model's mean error, and `friedman_nemenyi_test` for a joint comparison of many models across origins. `forecast_arena.forecasters.metrics` holds the metric library (MAE, MSE, RMSE, MAPE, sMAPE, MASE, RMSSE, OWA, a Diebold-Mariano test and a block bootstrap).

## Scaling to the full sweep

The full study behind this harness ran the whole sweep roster over 17 tasks with 15 rolling origins per task, on an HPC cluster with A100 GPUs for the deep and foundation models and on CPU for everything else. The same code path runs on a laptop: pick fewer origins, fewer tasks or a library at a time, and let the JSONL files accumulate.

## Roadmap

* Multivariate tasks with exogenous channels (the wrappers already accept `X`).
* Probabilistic scoring (CRPS, pinball) for the models that expose quantiles.
* Per-model wall-clock budgets and a cost column on the leaderboard.
* A hosted leaderboard built from community-submitted record files.

## Citation

```bibtex
@software{stavroglou2026forecastarena,
  author  = {Stavroglou, Stavros K.},
  title   = {forecast-arena: one rolling-origin protocol for every forecaster},
  year    = {2026},
  url     = {https://github.com/leoxeno/forecast-arena},
  version = {0.1.0}
}
```

## Licence

Apache-2.0. Third-party models and datasets keep their own licences; the task atlas lists each one.
