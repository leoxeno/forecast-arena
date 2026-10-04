"""
forecasters/wrappers/sktime_wrapper.py - sktime unified interface.

Wraps sktime models:
- ThetaForecaster
- AutoETS
- BATS/TBATS
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from ..date_utils import safe_date_range
from .. import register_model


def _to_period_freq(freq: str) -> str:
    """Translate new pandas freq aliases to old ones for pd.period_range compatibility.

    PeriodIndex requires 'M', 'Y', 'Q' — rejects 'ME', 'YE', 'QE'.
    """
    _MAP = {
        'YE': 'Y', 'ME': 'M', 'QE': 'Q',
        'BYE': 'BY', 'BME': 'BM', 'BQE': 'BQ',
        'h': 'H', 'min': 'T', 's': 'S',
    }
    for new, old in _MAP.items():
        if freq == new or freq.startswith(new + '-'):
            return freq.replace(new, old, 1)
    return freq


def _register_if_tbats_available(cls):
    """Only register TBATS model if tbats package is installed."""
    try:
        import tbats
        return register_model(cls)
    except ImportError:
        return cls


class _BaseSktimeForecaster(Forecaster):
    """Base class for sktime forecasters."""

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> '_BaseSktimeForecaster':
        y = self._validate_y(y)
        self._train_y = y

        # sktime uses pd.Series with DatetimeIndex
        self._y_series = pd.Series(
            y,
            index=safe_date_range(start='2000-01-01', periods=len(y), freq=freq or 'D')
        )

        self._model = self._create_model()
        self._model.fit(self._y_series)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()
        fh = list(range(1, horizon + 1))
        return self._model.predict(fh=fh).values

    def _compute_fitted_values(self):
        n = len(self._train_y)
        fh = list(range(-n + 1, 1))  # in-sample forecast horizons
        return self._model.predict(fh=fh).values

    def get_diagnostics(self):
        if not self._is_fitted or self._model is None:
            return {}
        d = {}
        try:
            fitted_params = self._model.get_fitted_params()
            fp = {}
            for k, v in fitted_params.items():
                if hasattr(v, "tolist"):
                    fp[k] = v.tolist()
                elif hasattr(v, "__float__"):
                    try:
                        fp[k] = float(v)
                    except (TypeError, ValueError):
                        fp[k] = str(v)
                else:
                    fp[k] = str(v)
            d["fitted_params"] = fp
        except Exception:
            pass
        return d

    def _create_model(self):
        raise NotImplementedError


@register_model
class SktimeTheta(_BaseSktimeForecaster):
    """
    sktime Theta forecaster.

    Classical Theta method with decomposition.
    """

    def __init__(self, sp: int = 1, **kwargs):
        super().__init__(sp=sp, **kwargs)

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'SktimeTheta':
        y = self._validate_y(y)
        self._train_y = y

        # ThetaForecaster requires PeriodIndex (not DatetimeIndex)
        period_freq = _to_period_freq(freq or 'D')
        self._y_series = pd.Series(
            y,
            index=pd.period_range(start='2000', periods=len(y), freq=period_freq)
        )

        self._model = self._create_model()
        self._model.fit(self._y_series)

        self._is_fitted = True
        return self

    def _create_model(self):
        from sktime.forecasting.theta import ThetaForecaster
        return ThetaForecaster(sp=self.params.get('sp', 1))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="SktimeTheta",
            category=ModelCategory.CLASSICAL,
            library="sktime",
            year=2020,
            paper="Assimakopoulos & Nikolopoulos (2000)",
        )


@register_model
class SktimeAutoETS(_BaseSktimeForecaster):
    """
    sktime AutoETS forecaster.

    Automatic exponential smoothing selection.
    """

    def __init__(self, sp: int = 1, **kwargs):
        super().__init__(sp=sp, **kwargs)

    def _create_model(self):
        from sktime.forecasting.ets import AutoETS
        return AutoETS(sp=self.params.get('sp', 1), auto=True)

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="SktimeAutoETS",
            category=ModelCategory.CLASSICAL,
            library="sktime",
            year=2020,
            paper="Hyndman et al. (2002)",
            probabilistic=True,
        )


@_register_if_tbats_available
class SktimeTBATS(Forecaster):
    """
    TBATS forecaster using tbats package directly.

    Trigonometric Box-Cox ARMA Trend Seasonal.
    Uses tbats package directly to avoid sktime's numpy version constraint.

    Requires: pip install tbats
    """

    def __init__(self, sp: int = 1, use_box_cox: bool = True, **kwargs):
        super().__init__(sp=sp, use_box_cox=use_box_cox, **kwargs)

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> 'SktimeTBATS':
        import warnings
        from tbats import TBATS as TBATSModel

        y = self._validate_y(y)
        self._train_y = y

        sp = self.params.get('sp', 1)
        seasonal_periods = [sp] if sp > 1 else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = TBATSModel(
                seasonal_periods=seasonal_periods,
                use_box_cox=self.params.get('use_box_cox', True),
            )
            self._fitted_model = model.fit(y)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()
        return self._fitted_model.forecast(steps=horizon)

    def _compute_fitted_values(self):
        return np.asarray(self._fitted_model.y_hat)

    def get_diagnostics(self):
        if not self._is_fitted or not hasattr(self, "_fitted_model"):
            return {}
        d = {}
        try:
            fm = self._fitted_model
            # Box-Cox lambda — NOT derivable (estimated during fit)
            if hasattr(fm, "params") and hasattr(fm.params, "box_cox_lambda"):
                lam = fm.params.box_cox_lambda
                d["box_cox_lambda"] = float(lam) if lam is not None else None
            # AIC for model selection — NOT derivable
            if hasattr(fm, "aic"):
                d["aic"] = float(fm.aic)
            # ARMA errors order — NOT derivable (selected during fit)
            if hasattr(fm, "params") and hasattr(fm.params, "alpha"):
                d["alpha"] = float(fm.params.alpha)
            if hasattr(fm, "params") and hasattr(fm.params, "beta"):
                d["beta"] = float(fm.params.beta) if fm.params.beta is not None else None
        except Exception:
            pass
        return d

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="SktimeTBATS",
            category=ModelCategory.CLASSICAL,
            library="sktime",
            year=2020,
            paper="De Livera et al. (2011)",
            probabilistic=True,
        )


# ============================================================================
# Phase 3: Additional sktime Models
# ============================================================================


@register_model
class SktimeAutoARIMA(_BaseSktimeForecaster):
    """
    sktime AutoARIMA forecaster.

    Automatic ARIMA order selection using pmdarima.

    Args:
        sp: Seasonal period
        max_p: Maximum AR order
        max_q: Maximum MA order
    """

    def __init__(self, sp: int = 1, max_p: int = 5, max_q: int = 5, **kwargs):
        super().__init__(sp=sp, max_p=max_p, max_q=max_q, **kwargs)

    def _create_model(self):
        from sktime.forecasting.arima import AutoARIMA
        return AutoARIMA(
            sp=self.params.get('sp', 1),
            max_p=self.params.get('max_p', 5),
            max_q=self.params.get('max_q', 5),
            suppress_warnings=True,
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="SktimeAutoARIMA",
            category=ModelCategory.CLASSICAL,
            library="sktime",
            year=2020,
            paper="Box & Jenkins (1970) / pmdarima",
            probabilistic=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("sp", "int", low=1, high=52, default=1),
            ParamSpace("max_p", "int", low=1, high=10, default=5),
            ParamSpace("max_q", "int", low=1, high=10, default=5),
        ]


@register_model
class SktimeARIMA(_BaseSktimeForecaster):
    """
    sktime ARIMA forecaster.

    Fixed-order ARIMA model.

    Args:
        order: (p, d, q) ARIMA order tuple
        seasonal_order: (P, D, Q, m) seasonal order tuple
    """

    def __init__(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (0, 0, 0, 0),
        **kwargs
    ):
        super().__init__(order=order, seasonal_order=seasonal_order, **kwargs)

    def _create_model(self):
        from sktime.forecasting.arima import ARIMA
        return ARIMA(
            order=self.params.get('order', (1, 1, 1)),
            seasonal_order=self.params.get('seasonal_order', (0, 0, 0, 0)),
            suppress_warnings=True,
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="SktimeARIMA",
            category=ModelCategory.CLASSICAL,
            library="sktime",
            year=2020,
            paper="Box & Jenkins (1970)",
            probabilistic=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("p", "int", low=0, high=5, default=1),
            ParamSpace("d", "int", low=0, high=2, default=1),
            ParamSpace("q", "int", low=0, high=5, default=1),
        ]


@register_model
class SktimeExponentialSmoothing(_BaseSktimeForecaster):
    """
    sktime ExponentialSmoothing forecaster.

    Holt-Winters exponential smoothing.

    Args:
        sp: Seasonal period
        trend: 'add', 'mul', or None
        seasonal: 'add', 'mul', or None
    """

    def __init__(
        self,
        sp: int = 1,
        trend: Optional[str] = 'add',
        seasonal: Optional[str] = None,
        **kwargs
    ):
        super().__init__(sp=sp, trend=trend, seasonal=seasonal, **kwargs)

    def _create_model(self):
        from sktime.forecasting.exp_smoothing import ExponentialSmoothing
        return ExponentialSmoothing(
            sp=self.params.get('sp', 1),
            trend=self.params.get('trend', 'add'),
            seasonal=self.params.get('seasonal'),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="SktimeExponentialSmoothing",
            category=ModelCategory.CLASSICAL,
            library="sktime",
            year=2020,
            paper="Hyndman et al. (2008)",
            probabilistic=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("sp", "int", low=1, high=52, default=1),
            ParamSpace("trend", "categorical", choices=['add', 'mul', None], default='add'),
            ParamSpace("seasonal", "categorical", choices=['add', 'mul', None], default=None),
        ]


@register_model
class SktimeProphet(_BaseSktimeForecaster):
    """
    sktime Prophet forecaster (Meta/Facebook Prophet).

    Additive model with trend, seasonality, and holidays.

    Args:
        growth: 'linear' or 'logistic'
        yearly_seasonality: Enable yearly seasonality
        weekly_seasonality: Enable weekly seasonality
    """

    def __init__(
        self,
        growth: str = 'linear',
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        **kwargs
    ):
        super().__init__(
            growth=growth,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            **kwargs
        )

    def _create_model(self):
        from sktime.forecasting.fbprophet import Prophet
        return Prophet(
            growth=self.params.get('growth', 'linear'),
            yearly_seasonality=self.params.get('yearly_seasonality', True),
            weekly_seasonality=self.params.get('weekly_seasonality', True),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="SktimeProphet",
            category=ModelCategory.CLASSICAL,
            library="sktime",
            year=2018,
            paper="Taylor & Letham (2018)",
            probabilistic=True,
            notes="Meta/Facebook Prophet via sktime",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("growth", "categorical", choices=['linear', 'logistic'], default='linear'),
        ]


@register_model
class SktimeNaiveForecaster(_BaseSktimeForecaster):
    """
    sktime NaiveForecaster - simple baseline.

    Various naive strategies: last, mean, drift.

    Args:
        strategy: 'last', 'mean', or 'drift'
        sp: Seasonal period (for 'last' with seasonality)
    """

    def __init__(self, strategy: str = 'last', sp: int = 1, **kwargs):
        super().__init__(strategy=strategy, sp=sp, **kwargs)

    def _create_model(self):
        from sktime.forecasting.naive import NaiveForecaster
        return NaiveForecaster(
            strategy=self.params.get('strategy', 'last'),
            sp=self.params.get('sp', 1),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="SktimeNaiveForecaster",
            category=ModelCategory.CLASSICAL,
            library="sktime",
            year=2020,
            paper="sktime documentation",
            notes="Baseline with multiple naive strategies",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("strategy", "categorical", choices=['last', 'mean', 'drift'], default='last'),
            ParamSpace("sp", "int", low=1, high=52, default=1),
        ]


@register_model
class SktimePolynomialTrend(_BaseSktimeForecaster):
    """
    sktime PolynomialTrendForecaster.

    Fits polynomial trend to time series.

    Args:
        degree: Polynomial degree
        with_intercept: Include intercept
    """

    def __init__(self, degree: int = 1, with_intercept: bool = True, **kwargs):
        super().__init__(degree=degree, with_intercept=with_intercept, **kwargs)

    def _create_model(self):
        from sktime.forecasting.trend import PolynomialTrendForecaster
        return PolynomialTrendForecaster(
            degree=self.params.get('degree', 1),
            with_intercept=self.params.get('with_intercept', True),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="SktimePolynomialTrend",
            category=ModelCategory.CLASSICAL,
            library="sktime",
            year=2020,
            paper="sktime documentation",
            notes="Polynomial trend extrapolation",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("degree", "int", low=1, high=5, default=1),
        ]


@register_model
class SktimeSTLForecaster(_BaseSktimeForecaster):
    """
    sktime STLForecaster - STL decomposition with forecaster.

    Seasonal-Trend decomposition using LOESS.

    Args:
        sp: Seasonal period
    """

    def __init__(self, sp: int = 7, **kwargs):
        super().__init__(sp=sp, **kwargs)

    def _create_model(self):
        from sktime.forecasting.trend import STLForecaster
        return STLForecaster(sp=self.params.get('sp', 7))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="SktimeSTLForecaster",
            category=ModelCategory.CLASSICAL,
            library="sktime",
            year=2020,
            paper="Cleveland et al. (1990)",
            notes="STL decomposition with trend forecaster",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("sp", "int", low=2, high=52, default=7),
        ]
