"""
forecasters/wrappers/statsmodels_wrapper.py - Statsmodels forecasters.

Wraps:
- ARIMA (auto and manual)
- ExponentialSmoothing (Holt-Winters)
- VAR (Vector Autoregression)
- SARIMAX
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from .. import register_model


@register_model
class ARIMA(Forecaster):
    """
    ARIMA (AutoRegressive Integrated Moving Average) forecaster.

    Uses pmdarima's auto_arima for automatic order selection or
    statsmodels ARIMA for manual specification.

    Args:
        order: (p, d, q) tuple for manual ARIMA. If None, uses auto_arima.
        seasonal_order: (P, D, Q, m) for seasonal ARIMA
        auto: Use auto_arima for order selection
        max_p, max_q, max_d: Limits for auto_arima search
    """

    def __init__(
        self,
        order: Optional[tuple] = None,
        seasonal_order: Optional[tuple] = None,
        auto: bool = True,
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 2,
        **kwargs
    ):
        super().__init__(
            order=order,
            seasonal_order=seasonal_order,
            auto=auto,
            max_p=max_p,
            max_q=max_q,
            max_d=max_d,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'ARIMA':
        y = self._validate_y(y)
        X = self._validate_X(X, len(y)) if X is not None else None
        self._train_y = y
        self._train_X = X

        if self.params.get('auto', True):
            import pmdarima as pm
            self._is_pmdarima = True
            self._model = pm.auto_arima(
                y,
                X=X,
                max_p=self.params.get('max_p', 5),
                max_q=self.params.get('max_q', 5),
                max_d=self.params.get('max_d', 2),
                seasonal=self.params.get('seasonal_order') is not None,
                m=self.params.get('seasonal_order', (0, 0, 0, 1))[-1] if self.params.get('seasonal_order') else 1,
                suppress_warnings=True,
                error_action='ignore',
                stepwise=True,
            )
        else:
            from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
            self._is_pmdarima = False
            order = self.params.get('order', (1, 1, 1))
            self._model = SM_ARIMA(y, order=order, exog=X).fit()

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()
        X = self._validate_X(X, horizon) if X is not None else None

        if self._is_pmdarima:
            return self._model.predict(n_periods=horizon, X=X)
        else:  # statsmodels
            return self._model.forecast(steps=horizon, exog=X)

    def _compute_fitted_values(self):
        if self._is_pmdarima:
            return self._model.predict_in_sample()
        return np.asarray(self._model.fittedvalues)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="ARIMA",
            category=ModelCategory.CLASSICAL,
            library="statsmodels",
            year=1970,
            paper="Box & Jenkins (1970)",
            probabilistic=True,
            exogenous=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("max_p", "int", low=1, high=10, default=5),
            ParamSpace("max_q", "int", low=1, high=10, default=5),
            ParamSpace("max_d", "int", low=0, high=2, default=2),
        ]

    def get_diagnostics(self):
        if not self._is_fitted or self._model is None:
            return {}
        d = {}
        if self._is_pmdarima:
            inner = self._model.arima_res_
            d["selected_order"] = list(self._model.order)
            d["seasonal_order"] = list(self._model.seasonal_order)
            for attr in ("aic", "bic", "llf"):
                if hasattr(inner, attr):
                    d[attr] = float(getattr(inner, attr))
            # nobs derivable from len(y_train), tvalues derivable from params/bse
            for attr in ("pvalues", "bse"):
                if hasattr(inner, attr):
                    pv = getattr(inner, attr)
                    if hasattr(pv, "items"):
                        d[attr] = {str(k): float(v) for k, v in pv.items()}
                    else:
                        d[attr] = [float(v) for v in pv]
        else:
            for attr in ("aic", "bic", "llf"):
                if hasattr(self._model, attr):
                    d[attr] = float(getattr(self._model, attr))
            for attr in ("pvalues", "bse"):
                if hasattr(self._model, attr):
                    pv = getattr(self._model, attr)
                    if hasattr(pv, "items"):
                        d[attr] = {str(k): float(v) for k, v in pv.items()}
                    else:
                        d[attr] = [float(v) for v in pv]
        return d


@register_model
class ExponentialSmoothing(Forecaster):
    """
    Holt-Winters Exponential Smoothing.

    Args:
        trend: 'add', 'mul', or None
        seasonal: 'add', 'mul', or None
        seasonal_periods: Number of periods in a season
        damped_trend: Use damped trend
    """

    def __init__(
        self,
        trend: Optional[str] = 'add',
        seasonal: Optional[str] = 'add',
        seasonal_periods: Optional[int] = None,
        damped_trend: bool = False,
        **kwargs
    ):
        super().__init__(
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            damped_trend=damped_trend,
            **kwargs
        )

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'ExponentialSmoothing':
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as ETS

        y = self._validate_y(y)
        self._train_y = y

        # Get seasonal_periods: explicit param > infer from data > default
        seasonal_periods = self.params.get('seasonal_periods')
        seasonal = self.params.get('seasonal', 'add')

        # If seasonal model but no periods specified, use reasonable default
        if seasonal is not None and seasonal_periods is None:
            # Default to 12 (monthly) if data is long enough, else 4 (quarterly)
            seasonal_periods = 12 if len(y) >= 24 else 4

        self._model = ETS(
            y,
            trend=self.params.get('trend', 'add'),
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            damped_trend=self.params.get('damped_trend', False),
        ).fit()

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()
        return self._model.forecast(horizon)

    def _compute_fitted_values(self):
        return np.asarray(self._model.fittedvalues)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="ExponentialSmoothing",
            category=ModelCategory.CLASSICAL,
            library="statsmodels",
            year=1960,
            paper="Holt (1957), Winters (1960)",
            probabilistic=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("trend", "categorical", choices=['add', 'mul', None]),
            ParamSpace("seasonal", "categorical", choices=['add', 'mul', None]),
            ParamSpace("damped_trend", "categorical", choices=[True, False]),
        ]

    def get_diagnostics(self):
        if not self._is_fitted or self._model is None:
            return {}
        d = {}
        # sse derivable from y_train - fitted_values
        if hasattr(self._model, "params"):
            sp = {}
            for k, v in self._model.params.items():
                if v is None:
                    sp[k] = None
                elif hasattr(v, "tolist"):
                    sp[k] = v.tolist()
                else:
                    try:
                        sp[k] = float(v)
                    except (TypeError, ValueError):
                        sp[k] = str(v)
            d["smoothing_params"] = sp
        return d


@register_model
class SARIMAX(Forecaster):
    """
    Seasonal ARIMA with eXogenous variables.

    Full-featured seasonal ARIMA model with exogenous support.

    Args:
        order: (p, d, q) tuple
        seasonal_order: (P, D, Q, m) tuple
    """

    def __init__(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (0, 0, 0, 0),
        **kwargs
    ):
        super().__init__(order=order, seasonal_order=seasonal_order, **kwargs)

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'SARIMAX':
        from statsmodels.tsa.statespace.sarimax import SARIMAX as SM_SARIMAX

        y = self._validate_y(y)
        X = self._validate_X(X, len(y)) if X is not None else None
        self._train_y = y
        self._train_X = X

        self._model = SM_SARIMAX(
            y,
            exog=X,
            order=self.params.get('order', (1, 1, 1)),
            seasonal_order=self.params.get('seasonal_order', (0, 0, 0, 0)),
        ).fit(disp=False)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()
        X = self._validate_X(X, horizon) if X is not None else None
        return self._model.forecast(steps=horizon, exog=X)

    def _compute_fitted_values(self):
        return np.asarray(self._model.fittedvalues)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="SARIMAX",
            category=ModelCategory.CLASSICAL,
            library="statsmodels",
            year=1970,
            paper="Box & Jenkins (1970)",
            probabilistic=True,
            exogenous=True,
        )

    def get_diagnostics(self):
        if not self._is_fitted or self._model is None:
            return {}
        d = {}
        for attr in ("aic", "bic", "llf"):
            if hasattr(self._model, attr):
                d[attr] = float(getattr(self._model, attr))
        # nobs derivable from len(y_train), tvalues derivable from params/bse
        for attr in ("pvalues", "bse"):
            if hasattr(self._model, attr):
                pv = getattr(self._model, attr)
                if hasattr(pv, "items"):
                    d[attr] = {str(k): float(v) for k, v in pv.items()}
                else:
                    d[attr] = [float(v) for v in pv]
        if hasattr(self._model, "specification"):
            spec = self._model.specification
            d["order"] = list(spec.get("order", []))
            d["seasonal_order"] = list(spec.get("seasonal_order", []))
        return d


@register_model
class VAR(Forecaster):
    """
    Vector Autoregression for multivariate time series.

    Forecasts multiple related time series jointly.

    Args:
        maxlags: Maximum lag order (None for automatic selection)
    """

    def __init__(self, maxlags: Optional[int] = None, **kwargs):
        super().__init__(maxlags=maxlags, **kwargs)

    def fit(
        self,
        y: Union[np.ndarray, pd.Series, pd.DataFrame],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'VAR':
        from statsmodels.tsa.api import VAR as SM_VAR

        if isinstance(y, pd.DataFrame):
            y = y.values
        elif isinstance(y, pd.Series):
            y = y.values.reshape(-1, 1)
        elif isinstance(y, np.ndarray):
            y = np.atleast_2d(y)
            if y.shape[0] == 1:
                y = y.T

        # VAR requires at least 2 variables
        if y.shape[1] < 2:
            raise ValueError(
                f"VAR requires multivariate input (at least 2 columns), got shape {y.shape}. "
                "Use ARIMA or SARIMAX for univariate time series."
            )

        self._train_y = y
        self._model = SM_VAR(y).fit(maxlags=self.params.get('maxlags'))
        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()
        return self._model.forecast(self._train_y[-self._model.k_ar:], steps=horizon)

    def _compute_fitted_values(self):
        if self._model is None:
            return None
        fv = self._model.fittedvalues
        if fv is not None:
            return np.asarray(fv).flatten()
        return None

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="VAR",
            category=ModelCategory.CLASSICAL,
            library="statsmodels",
            year=1980,
            paper="Sims (1980)",
            multivariate=True,
        )

    def get_diagnostics(self):
        if not self._is_fitted or self._model is None:
            return {}
        d = {"k_ar": int(self._model.k_ar)}
        if hasattr(self._model, "aic"):
            d["aic"] = float(self._model.aic)
        if hasattr(self._model, "bic"):
            d["bic"] = float(self._model.bic)
        if hasattr(self._model, "det_coef"):
            d["det_coef"] = float(self._model.det_coef)
        return d
