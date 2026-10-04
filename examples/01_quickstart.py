"""Quickstart: a dozen CPU baselines on a chaotic series and a real one.

Run:  python examples/01_quickstart.py --origins 5
Every (task, model, origin) record lands in results/quickstart/<task>.jsonl and
the leaderboards print at the end. Re-running resumes from the files.
"""

import argparse

from forecast_arena import Protocol, per_task_table, run_arena, summarise, to_markdown
from forecast_arena.runner import available_models

CANDIDATES = [
    "Naive", "SeasonalNaive", "Drift", "Mean",                    # numpy floor
    "ExponentialSmoothing", "ARIMA",                              # statsmodels
    "StatsForecastAutoETS", "StatsForecastAutoTheta",             # statsforecast
    "Ridge", "KNN", "ExtraTrees", "XGBoost",                      # sklearn family on lagged windows
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="lorenz-x,sunspots-yearly")
    p.add_argument("--origins", type=int, default=5)
    p.add_argument("--out", default="results/quickstart")
    args = p.parse_args()

    available = set(available_models())
    models = [m for m in CANDIDATES if m in available]
    missing = [m for m in CANDIDATES if m not in available]
    if missing:
        print(f"Skipping models whose library is not installed: {missing}")

    tasks = [t.strip() for t in args.tasks.split(",")]
    df = run_arena(tasks, models, protocol=Protocol(n_origins=args.origins), output_dir=args.out)

    table = per_task_table(df)
    for task in tasks:
        sub = table[table["task"] == task][["rank", "model", "mae", "rmse", "mase", "n_errors", "fit_time_s"]]
        print(f"\n### {task} (mean over {args.origins} rolling origins)\n")
        print(to_markdown(sub))

    print("\n### Across tasks\n")
    print(to_markdown(summarise(table)))


if __name__ == "__main__":
    main()
