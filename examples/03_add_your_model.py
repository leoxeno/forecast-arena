"""Enter your own forecaster in the arena in about twenty lines.

A model needs fit(), predict() and get_metadata(). Register it with the
decorator and it is addressable by name like every other entrant.

Run:  python examples/03_add_your_model.py
"""

from typing import List

import numpy as np

from forecast_arena import (
    Forecaster, ModelCategory, ModelMetadata, ParamSpace, Protocol, per_task_table,
    register_model, run_arena, to_markdown,
)


@register_model
class DampedDrift(Forecaster):
    """Drift whose slope decays geometrically: y_T + slope * (phi + phi^2 + ... + phi^h)."""

    def fit(self, y, X=None, freq=None, **kwargs) -> "DampedDrift":
        self._train_y = self._validate_y(y)
        self._is_fitted = True
        return self

    def predict(self, horizon: int, X=None, **kwargs) -> np.ndarray:
        self._check_fitted()
        y = self._train_y
        phi = float(self.params.get("phi", 0.9))
        window = int(self.params.get("window", 20))
        recent = y[-window:]
        slope = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        steps = np.cumsum(phi ** np.arange(1, horizon + 1))
        return y[-1] + slope * steps

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(name="DampedDrift", category=ModelCategory.CLASSICAL, library="numpy",
                             year=2026, paper="Example model from the forecast-arena README")

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [ParamSpace("phi", "float", low=0.5, high=0.99, default=0.9),
                ParamSpace("window", "int", low=5, high=60, default=20)]


def main() -> None:
    df = run_arena(["nile", "co2-monthly-1958-2001"], ["Naive", "Drift", "DampedDrift", "Ridge"],
                   protocol=Protocol(n_origins=5), output_dir="results/add-your-model")
    table = per_task_table(df)
    for task in ["nile", "co2-monthly-1958-2001"]:
        print(f"\n### {task}\n")
        print(to_markdown(table[table["task"] == task][["rank", "model", "mae", "mase"]]))


if __name__ == "__main__":
    main()
