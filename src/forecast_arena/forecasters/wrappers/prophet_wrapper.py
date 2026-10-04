"""
forecasters/wrappers/prophet_wrapper.py - Facebook Prophet.

Wraps Prophet for additive/multiplicative decomposition forecasting.
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from ..date_utils import safe_date_range
from .. import register_model


@register_model
class Prophet(Forecaster):
    """
    Facebook Prophet forecaster.

    Decomposition-based model with trend, seasonality, and holidays.
    Good for daily/weekly data with strong seasonal patterns.

    Args:
        growth: 'linear' or 'logistic'
        seasonality_mode: 'additive' or 'multiplicative'
        yearly_seasonality: Enable yearly seasonality
        weekly_seasonality: Enable weekly seasonality
        daily_seasonality: Enable daily seasonality
    """

    def __init__(
        self,
        growth: str = 'linear',
        seasonality_mode: str = 'additive',
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        **kwargs
    ):
        super().__init__(
            growth=growth,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'Prophet':
        try:
            from prophet import Prophet as FBProphet
        except ImportError:
            raise ImportError("Prophet not installed. Install with: pip install prophet")

        y = self._validate_y(y)
        self._train_y = y
        self._freq = freq or 'D'

        # Prophet requires ds (date) and y columns
        df = pd.DataFrame({
            'ds': safe_date_range(start='2000-01-01', periods=len(y), freq=self._freq),
            'y': y
        })

        # Track exogenous column names
        self._exog_cols = []
        exog = self._validate_X(X, len(y)) if X is not None else None
        if exog is not None:
            self._train_X = exog  # Store for predict() — needed to fill historical rows
            for j in range(exog.shape[1]):
                col_name = f'exog_{j}'
                df[col_name] = exog[:, j]
                self._exog_cols.append(col_name)

        self._model = FBProphet(
            growth=self.params.get('growth', 'linear'),
            seasonality_mode=self.params.get('seasonality_mode', 'additive'),
            yearly_seasonality=self.params.get('yearly_seasonality', True),
            weekly_seasonality=self.params.get('weekly_seasonality', True),
            daily_seasonality=self.params.get('daily_seasonality', False),
        )

        # Register exogenous regressors before fitting
        for col in self._exog_cols:
            self._model.add_regressor(col)

        self._model.fit(df)
        self._last_date = df['ds'].iloc[-1]

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        future = self._model.make_future_dataframe(periods=horizon, freq=self._freq)

        # Add exogenous columns to future dataframe
        if self._exog_cols:
            exog_future = self._validate_X(X, horizon) if X is not None else None
            if exog_future is not None:
                # future df includes historical + future rows, fill exog for all
                for j, col in enumerate(self._exog_cols):
                    # Historical rows: use training values, future rows: use X_future
                    hist_vals = list(self._train_X[:, j]) if self._train_X is not None else []
                    future_vals = list(exog_future[:, j])
                    future[col] = hist_vals + future_vals
            else:
                # No future X provided but model expects it — fill with zeros
                for col in self._exog_cols:
                    future[col] = 0.0

        forecast = self._model.predict(future)

        return forecast['yhat'].values[-horizon:]

    def _compute_fitted_values(self):
        df = pd.DataFrame({
            'ds': safe_date_range(start='2000-01-01', periods=len(self._train_y), freq=self._freq),
        })
        forecast = self._model.predict(df)
        return forecast['yhat'].values

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        future = self._model.make_future_dataframe(periods=horizon, freq=self._freq)

        # Add exogenous columns to future dataframe
        if self._exog_cols:
            exog_future = self._validate_X(X, horizon) if X is not None else None
            if exog_future is not None:
                for j, col in enumerate(self._exog_cols):
                    hist_vals = list(self._train_X[:, j]) if self._train_X is not None else []
                    future_vals = list(exog_future[:, j])
                    future[col] = hist_vals + future_vals
            else:
                for col in self._exog_cols:
                    future[col] = 0.0

        forecast = self._model.predict(future)

        # Prophet provides yhat_lower and yhat_upper at ~80% interval
        result = np.zeros((horizon, len(quantiles)))
        for i, q in enumerate(quantiles):
            if q == 0.5:
                result[:, i] = forecast['yhat'].values[-horizon:]
            elif q < 0.5:
                result[:, i] = forecast['yhat_lower'].values[-horizon:]
            else:
                result[:, i] = forecast['yhat_upper'].values[-horizon:]

        return result

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="Prophet",
            category=ModelCategory.CLASSICAL,
            library="prophet",
            year=2017,
            paper="Taylor & Letham (2017)",
            probabilistic=True,
            exogenous=True,
            github_url="https://github.com/facebook/prophet",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("growth", "categorical", choices=['linear', 'logistic']),
            ParamSpace("seasonality_mode", "categorical", choices=['additive', 'multiplicative']),
        ]

    def get_diagnostics(self):
        if not self._is_fitted or self._model is None:
            return {}
        d = {}
        # n_changepoints derivable from len(changepoint_dates)
        if hasattr(self._model, "changepoints") and self._model.changepoints is not None:
            d["changepoint_dates"] = [str(c) for c in self._model.changepoints]
        if hasattr(self._model, "seasonalities"):
            d["seasonalities"] = {
                k: {"period": v["period"], "fourier_order": v["fourier_order"]}
                for k, v in self._model.seasonalities.items()
            }
        # Bayesian posterior point estimates — NOT derivable post-fit
        if hasattr(self._model, "params") and self._model.params:
            fp = {}
            for k, v in self._model.params.items():
                if hasattr(v, "tolist"):
                    fp[k] = v.tolist()
                elif v is None:
                    fp[k] = None
                else:
                    try:
                        fp[k] = float(v)
                    except (TypeError, ValueError):
                        fp[k] = str(v)
            d["fitted_params"] = fp
        return d
