"""
forecasters/wrappers/statsforecast_wrapper.py - Nixtla StatsForecast.

Fast implementations of statistical models:
- AutoARIMA
- AutoETS
- AutoTheta
- AutoCES
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from ..date_utils import safe_date_range
from .. import register_model


class _BaseStatsForecast(Forecaster):
    """Base class for StatsForecast models."""

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> '_BaseStatsForecast':
        from statsforecast import StatsForecast

        y = self._validate_y(y)
        self._train_y = y

        # StatsForecast requires specific DataFrame format
        dates = safe_date_range(start='2000-01-01', periods=len(y), freq=freq or 'D')
        df = pd.DataFrame({
            'unique_id': ['series'] * len(y),
            'ds': dates,
            'y': y
        })

        # Add exogenous columns if provided
        self._exog_cols = []
        if X is not None:
            exog = self._validate_X(X, len(y))
            for j in range(exog.shape[1]):
                col_name = f'exog_{j}'
                df[col_name] = exog[:, j]
                self._exog_cols.append(col_name)

        self._model = StatsForecast(
            models=[self._create_model()],
            freq=freq or 'D',
        )
        # Call fit() explicitly so fitted_ is populated for get_diagnostics()
        self._model.fit(df=df)
        self._df = df
        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        # Build X_df for exogenous if model was fit with exog
        X_df = None
        if self._exog_cols and X is not None:
            exog_future = self._validate_X(X, horizon)
            dates = pd.date_range(
                start=self._df['ds'].iloc[-1] + pd.tseries.frequencies.to_offset(
                    self._df['ds'].iloc[-1] - self._df['ds'].iloc[-2]
                ),
                periods=horizon,
                freq=self._df['ds'].iloc[-1] - self._df['ds'].iloc[-2],
            )
            X_df = pd.DataFrame({'unique_id': ['series'] * horizon, 'ds': dates})
            for j, col in enumerate(self._exog_cols):
                X_df[col] = exog_future[:, j]

        try:
            forecast = self._model.predict(h=horizon, X_df=X_df)
        except (OverflowError, pd.errors.OutOfBoundsDatetime, ValueError):
            # Datetime overflow when StatsForecast tries to build forecast indices
            # beyond numpy/pandas datetime64 limits (annual data spanning >260 years).
            return np.full(horizon, np.nan)
        # StatsForecast returns [unique_id, ds, model_name] - get the last column (forecast values)
        return forecast.iloc[:, -1].values.astype(float)

    def get_diagnostics(self):
        """Extract fitted model internals from StatsForecast.

        After forecast(), the underlying fitted model is stored internally.
        We extract non-derivable info: selected model type/order for Auto* models.
        """
        if not self._is_fitted or self._model is None:
            return {}
        d = {}
        try:
            # StatsForecast stores fitted models after forecast()
            # Access varies by version; guard with try/except
            if hasattr(self._model, "fitted_"):
                fitted = self._model.fitted_
                if fitted and len(fitted) > 0 and len(fitted[0]) > 0:
                    inner = fitted[0][0]
                    # AutoARIMA: extract selected order
                    if hasattr(inner, "model_"):
                        model_dict = inner.model_
                        if "arma" in model_dict:
                            d["arma_order"] = list(model_dict["arma"])
                        if "aic" in model_dict:
                            d["aic"] = float(model_dict["aic"])
                        if "aicc" in model_dict:
                            d["aicc"] = float(model_dict["aicc"])
                        if "bic" in model_dict:
                            d["bic"] = float(model_dict["bic"])
                        if "coef" in model_dict:
                            d["coef"] = [float(c) for c in model_dict["coef"]]
                        # AutoETS: extract selected model components
                        if "components" in model_dict:
                            d["components"] = list(model_dict["components"])
        except Exception:
            pass
        return d

    def _compute_fitted_values(self):
        if self._model is None:
            return None
        # StatsForecast 2.x: must call forecast(df=..., fitted=True) first,
        # then forecast_fitted_values() to retrieve in-sample predictions
        try:
            self._model.forecast(df=self._df, h=1, fitted=True)
            insample_df = self._model.forecast_fitted_values()
        except Exception:
            # Fallback: try predict_insample (older API)
            try:
                insample_df = self._model.predict_insample()
            except Exception:
                return None
        if insample_df is None or len(insample_df) == 0:
            return None
        # Select point forecast column (skip metadata and confidence intervals)
        model_cols = [c for c in insample_df.columns
                      if c not in ('unique_id', 'ds')
                      and '-lo-' not in c and '-hi-' not in c]
        if not model_cols:
            return None
        return insample_df[model_cols[0]].values.astype(float)

    def _create_model(self):
        raise NotImplementedError


@register_model
class StatsForecastAutoARIMA(_BaseStatsForecast):
    """
    StatsForecast AutoARIMA - fast ARIMA with automatic order selection.

    Uses Hyndman-Khandakar algorithm for order selection.
    """

    def __init__(self, season_length: int = 1, **kwargs):
        super().__init__(season_length=season_length, **kwargs)

    def _create_model(self):
        from statsforecast.models import AutoARIMA
        return AutoARIMA(season_length=self.params.get('season_length', 1))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastAutoARIMA",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Garza et al. (2022)",
            probabilistic=True,
            exogenous=True,
        )


@register_model
class StatsForecastAutoETS(_BaseStatsForecast):
    """
    StatsForecast AutoETS - automatic Exponential Smoothing.

    Selects best ETS model from all combinations.
    """

    def __init__(self, season_length: int = 1, **kwargs):
        super().__init__(season_length=season_length, **kwargs)

    def _create_model(self):
        from statsforecast.models import AutoETS
        return AutoETS(season_length=self.params.get('season_length', 1))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastAutoETS",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Garza et al. (2022)",
            probabilistic=True,
        )


@register_model
class StatsForecastAutoTheta(_BaseStatsForecast):
    """
    StatsForecast AutoTheta - automatic Theta method.

    Simple yet competitive forecasting method.
    """

    def __init__(self, season_length: int = 1, **kwargs):
        super().__init__(season_length=season_length, **kwargs)

    def _create_model(self):
        from statsforecast.models import AutoTheta
        return AutoTheta(season_length=self.params.get('season_length', 1))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastAutoTheta",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Garza et al. (2022)",
        )


@register_model
class StatsForecastAutoCES(_BaseStatsForecast):
    """
    StatsForecast AutoCES - automatic Complex Exponential Smoothing.

    CES handles complex seasonal patterns with Fourier terms.

    Args:
        season_length: Seasonal period
    """

    def __init__(self, season_length: int = 1, **kwargs):
        super().__init__(season_length=season_length, **kwargs)

    def _create_model(self):
        from statsforecast.models import AutoCES
        return AutoCES(season_length=self.params.get('season_length', 1))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastAutoCES",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Garza et al. (2022)",
            probabilistic=True,
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("season_length", "int", low=1, high=52, default=1),
        ]


@register_model
class StatsForecastMSTL(_BaseStatsForecast):
    """
    StatsForecast MSTL - Multiple Seasonal-Trend Decomposition using LOESS.

    Decomposes series with multiple seasonal patterns.

    Args:
        season_length: Seasonal period
    """

    def __init__(self, season_length: int = 7, **kwargs):
        super().__init__(season_length=season_length, **kwargs)

    def _create_model(self):
        from statsforecast.models import MSTL
        from statsforecast.models import AutoARIMA
        # MSTL wraps a forecaster for the trend residual
        return MSTL(season_length=self.params.get('season_length', 7))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastMSTL",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Garza et al. (2022) / Bandara et al. (2021)",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("season_length", "int", low=2, high=52, default=7),
        ]


@register_model
class StatsForecastCroston(_BaseStatsForecast):
    """
    StatsForecast Croston - intermittent demand forecasting.

    Separates demand size and inter-arrival time for sparse time series.
    Ideal for inventory/spare parts forecasting.

    WARNING: Designed for intermittent demand (many zeros). Will produce
    flat constant forecasts. Not suitable for continuous or seasonal data.

    Args:
        None (simple model)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_model(self):
        from statsforecast.models import CrostonOptimized
        return CrostonOptimized()

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastCroston",
            category=ModelCategory.INTERMITTENT,
            library="statsforecast",
            year=2022,
            paper="Croston (1972) / Garza et al. (2022)",
            notes="For sparse demand (many zeros). Outputs flat forecasts - not for continuous data.",
        )


@register_model
class StatsForecastSeasonalNaive(_BaseStatsForecast):
    """
    StatsForecast SeasonalNaive - naive seasonal forecast.

    Forecasts the value from the same period in the last season.
    Strong baseline for seasonal data.

    Args:
        season_length: Seasonal period
    """

    def __init__(self, season_length: int = 7, **kwargs):
        super().__init__(season_length=season_length, **kwargs)

    def _create_model(self):
        from statsforecast.models import SeasonalNaive
        return SeasonalNaive(season_length=self.params.get('season_length', 7))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastSeasonalNaive",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Garza et al. (2022)",
            notes="Baseline model - repeats last season",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("season_length", "int", low=2, high=52, default=7),
        ]


@register_model
class StatsForecastTheta(_BaseStatsForecast):
    """
    StatsForecast Theta - standard Theta method.

    Decomposes series into two theta lines and extrapolates.
    Won M3 forecasting competition.

    Args:
        season_length: Seasonal period
    """

    def __init__(self, season_length: int = 1, **kwargs):
        super().__init__(season_length=season_length, **kwargs)

    def _create_model(self):
        from statsforecast.models import Theta
        return Theta(season_length=self.params.get('season_length', 1))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastTheta",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Assimakopoulos & Nikolopoulos (2000) / Garza et al. (2022)",
            notes="M3 competition winner",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("season_length", "int", low=1, high=52, default=1),
        ]


# @register_model  # ARCHIVED: volatility model, not point forecast
class StatsForecastGARCH(_BaseStatsForecast):
    """
    StatsForecast GARCH - volatility forecasting.

    Generalized Autoregressive Conditional Heteroskedasticity.
    Models time-varying variance for financial time series.

    Args:
        p: GARCH lag order
        q: ARCH lag order
    """

    def __init__(self, p: int = 1, q: int = 1, **kwargs):
        super().__init__(p=p, q=q, **kwargs)

    def _create_model(self):
        from statsforecast.models import GARCH
        return GARCH(
            p=self.params.get('p', 1),
            q=self.params.get('q', 1),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastGARCH",
            category=ModelCategory.VOLATILITY,
            library="statsforecast",
            year=2022,
            paper="Bollerslev (1986) / Garza et al. (2022)",
            notes="Forecasts VOLATILITY (σ), not level. Exclude from point forecast benchmarks.",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("p", "int", low=1, high=5, default=1),
            ParamSpace("q", "int", low=1, high=5, default=1),
        ]


@register_model
class StatsForecastTBATS(_BaseStatsForecast):
    """
    StatsForecast TBATS - complex seasonality handling.

    Trigonometric seasonality, Box-Cox transformation, ARMA errors,
    Trend and Seasonal components.

    Args:
        season_length: Primary seasonal period
    """

    def __init__(self, season_length: int = 7, **kwargs):
        super().__init__(season_length=season_length, **kwargs)

    def _create_model(self):
        from statsforecast.models import TBATS
        return TBATS(season_length=self.params.get('season_length', 7))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastTBATS",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="De Livera et al. (2011) / Garza et al. (2022)",
            notes="Handles multiple/complex seasonalities",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("season_length", "int", low=2, high=52, default=7),
        ]


@register_model
class StatsForecastOptimizedTheta(_BaseStatsForecast):
    """
    StatsForecast OptimizedTheta - improved Theta method.

    Optimizes the theta parameter using information criteria.
    Often outperforms standard Theta.

    Args:
        season_length: Seasonal period
    """

    def __init__(self, season_length: int = 1, **kwargs):
        super().__init__(season_length=season_length, **kwargs)

    def _create_model(self):
        from statsforecast.models import OptimizedTheta
        return OptimizedTheta(season_length=self.params.get('season_length', 1))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastOptimizedTheta",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Fiorucci et al. (2016) / Garza et al. (2022)",
            notes="Improved Theta with parameter optimization",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("season_length", "int", low=1, high=52, default=1),
        ]


@register_model
class StatsForecastDynamicOptimizedTheta(_BaseStatsForecast):
    """
    StatsForecast DynamicOptimizedTheta - dynamic version of Optimized Theta.

    Updates theta parameter dynamically as data evolves.

    Args:
        season_length: Seasonal period
    """

    def __init__(self, season_length: int = 1, **kwargs):
        super().__init__(season_length=season_length, **kwargs)

    def _create_model(self):
        from statsforecast.models import DynamicOptimizedTheta
        return DynamicOptimizedTheta(season_length=self.params.get('season_length', 1))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastDynamicOptimizedTheta",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Fiorucci et al. (2016) / Garza et al. (2022)",
            notes="Dynamic theta parameter estimation",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("season_length", "int", low=1, high=52, default=1),
        ]


@register_model
class StatsForecastHistoricAverage(_BaseStatsForecast):
    """
    StatsForecast HistoricAverage - simple average baseline.

    Forecasts the mean of all historical values.
    Simplest possible baseline for comparison.

    Args:
        None (no parameters)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_model(self):
        from statsforecast.models import HistoricAverage
        return HistoricAverage()

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastHistoricAverage",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Garza et al. (2022)",
            notes="Baseline model - returns historical mean",
        )


# ============================================================================
# Phase 3: Additional StatsForecast Models
# ============================================================================


@register_model
class StatsForecastCrostonSBA(_BaseStatsForecast):
    """
    StatsForecast CrostonSBA - Syntetos-Boylan Approximation.

    Bias-corrected version of Croston's method for intermittent demand.
    Often outperforms standard Croston for sparse time series.

    WARNING: Designed for intermittent demand (many zeros). Will produce
    flat constant forecasts. Not suitable for continuous or seasonal data.

    Args:
        None (simple model)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_model(self):
        from statsforecast.models import CrostonSBA
        return CrostonSBA()

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastCrostonSBA",
            category=ModelCategory.INTERMITTENT,
            library="statsforecast",
            year=2022,
            paper="Syntetos & Boylan (2005) / Garza et al. (2022)",
            notes="For sparse demand (many zeros). Outputs flat forecasts - not for continuous data.",
        )


@register_model
class StatsForecastCrostonClassic(_BaseStatsForecast):
    """
    StatsForecast CrostonClassic - Original Croston method.

    Original method for intermittent demand forecasting.
    Separates demand size and inter-arrival time.

    WARNING: Designed for intermittent demand (many zeros). Will produce
    flat constant forecasts. Not suitable for continuous or seasonal data.

    Args:
        None (simple model)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_model(self):
        from statsforecast.models import CrostonClassic
        return CrostonClassic()

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastCrostonClassic",
            category=ModelCategory.INTERMITTENT,
            library="statsforecast",
            year=2022,
            paper="Croston (1972) / Garza et al. (2022)",
            notes="For sparse demand (many zeros). Outputs flat forecasts - not for continuous data.",
        )


@register_model
class StatsForecastTSB(_BaseStatsForecast):
    """
    StatsForecast TSB - Teunter-Syntetos-Babai method.

    Alternative intermittent demand method that directly
    estimates demand rate rather than interval.

    WARNING: Designed for intermittent demand (many zeros). Will produce
    flat constant forecasts. Not suitable for continuous or seasonal data.

    Args:
        alpha_d: Smoothing for demand (0-1)
        alpha_p: Smoothing for probability (0-1)
    """

    def __init__(self, alpha_d: float = 0.1, alpha_p: float = 0.1, **kwargs):
        super().__init__(alpha_d=alpha_d, alpha_p=alpha_p, **kwargs)

    def _create_model(self):
        from statsforecast.models import TSB
        return TSB(
            alpha_d=self.params.get('alpha_d', 0.1),
            alpha_p=self.params.get('alpha_p', 0.1),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastTSB",
            category=ModelCategory.INTERMITTENT,
            library="statsforecast",
            year=2022,
            paper="Teunter et al. (2011) / Garza et al. (2022)",
            notes="For sparse demand (many zeros). Outputs flat forecasts - not for continuous data.",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("alpha_d", "float", low=0.01, high=0.5, default=0.1),
            ParamSpace("alpha_p", "float", low=0.01, high=0.5, default=0.1),
        ]


@register_model
class StatsForecastADIDA(_BaseStatsForecast):
    """
    StatsForecast ADIDA - Aggregate-Disaggregate Intermittent Demand Approach.

    Aggregates sparse data to reduce zeros, then disaggregates forecast.

    WARNING: Designed for intermittent demand (many zeros). Will produce
    flat constant forecasts. Not suitable for continuous or seasonal data.

    Args:
        None (simple model)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_model(self):
        from statsforecast.models import ADIDA
        return ADIDA()

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastADIDA",
            category=ModelCategory.INTERMITTENT,
            library="statsforecast",
            year=2022,
            paper="Nikolopoulos et al. (2011) / Garza et al. (2022)",
            notes="For sparse demand (many zeros). Outputs flat forecasts - not for continuous data.",
        )


@register_model
class StatsForecastIMAPA(_BaseStatsForecast):
    """
    StatsForecast IMAPA - Intermittent Multiple Aggregation Prediction Algorithm.

    Multiple temporal aggregation for intermittent demand.

    WARNING: Designed for intermittent demand (many zeros). Will produce
    flat constant forecasts. Not suitable for continuous or seasonal data.

    Args:
        None (simple model)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_model(self):
        from statsforecast.models import IMAPA
        return IMAPA()

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastIMAPA",
            category=ModelCategory.INTERMITTENT,
            library="statsforecast",
            year=2022,
            paper="Petropoulos & Kourentzes (2015) / Garza et al. (2022)",
            notes="For sparse demand (many zeros). Outputs flat forecasts - not for continuous data.",
        )


@register_model
class StatsForecastNaive(_BaseStatsForecast):
    """
    StatsForecast Naive - simplest baseline.

    Forecasts the last observed value. Essential baseline.

    Args:
        None (no parameters)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_model(self):
        from statsforecast.models import Naive
        return Naive()

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastNaive",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Garza et al. (2022)",
            notes="Baseline - repeats last value",
        )


@register_model
class StatsForecastWindowAverage(_BaseStatsForecast):
    """
    StatsForecast WindowAverage - moving average baseline.

    Forecasts the mean of last window_size values.

    Args:
        window_size: Number of past values to average
    """

    def __init__(self, window_size: int = 7, **kwargs):
        super().__init__(window_size=window_size, **kwargs)

    def _create_model(self):
        from statsforecast.models import WindowAverage
        return WindowAverage(window_size=self.params.get('window_size', 7))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastWindowAverage",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Garza et al. (2022)",
            notes="Moving average baseline",
        )

    def _compute_fitted_values(self):
        """WindowAverage fitted = rolling mean of last window_size values."""
        if self._train_y is None:
            return None
        w = self.params.get('window_size', 7)
        y = self._train_y
        fitted = np.full(len(y), np.nan)
        for t in range(w, len(y)):
            fitted[t] = np.mean(y[t - w:t])
        return fitted

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("window_size", "int", low=2, high=30, default=7),
        ]


@register_model
class StatsForecastSeasonalWindowAverage(_BaseStatsForecast):
    """
    StatsForecast SeasonalWindowAverage - seasonal moving average.

    Averages values from same seasonal period across windows.

    Args:
        season_length: Seasonal period
        window_size: Number of seasons to average
    """

    def __init__(self, season_length: int = 7, window_size: int = 4, **kwargs):
        super().__init__(season_length=season_length, window_size=window_size, **kwargs)

    def _create_model(self):
        from statsforecast.models import SeasonalWindowAverage
        return SeasonalWindowAverage(
            season_length=self.params.get('season_length', 7),
            window_size=self.params.get('window_size', 4),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastSeasonalWindowAverage",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Garza et al. (2022)",
            notes="Seasonal moving average baseline",
        )

    def _compute_fitted_values(self):
        """SeasonalWindowAverage fitted = mean of same-season values across windows."""
        if self._train_y is None:
            return None
        m = self.params.get('season_length', 7)
        w = self.params.get('window_size', 4)
        y = self._train_y
        fitted = np.full(len(y), np.nan)
        for t in range(m * w, len(y)):
            vals = [y[t - m * k] for k in range(1, w + 1) if t - m * k >= 0]
            if vals:
                fitted[t] = np.mean(vals)
        return fitted

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("season_length", "int", low=2, high=52, default=7),
            ParamSpace("window_size", "int", low=1, high=10, default=4),
        ]


@register_model
class StatsForecastRandomWalkWithDrift(_BaseStatsForecast):
    """
    StatsForecast RandomWalkWithDrift - random walk with trend.

    Naive + average historical change (drift).

    Args:
        None (simple model)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_model(self):
        from statsforecast.models import RandomWalkWithDrift
        return RandomWalkWithDrift()

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastRandomWalkWithDrift",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Garza et al. (2022)",
            notes="Random walk plus drift term",
        )


@register_model
class StatsForecastSES(_BaseStatsForecast):
    """
    StatsForecast SES - Simple Exponential Smoothing.

    Basic exponential smoothing without trend or seasonality.

    Args:
        alpha: Smoothing parameter (0-1)
    """

    def __init__(self, alpha: float = 0.1, **kwargs):
        super().__init__(alpha=alpha, **kwargs)

    def _create_model(self):
        from statsforecast.models import SimpleExponentialSmoothing
        return SimpleExponentialSmoothing(alpha=self.params.get('alpha', 0.1))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastSES",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Brown (1959) / Garza et al. (2022)",
            notes="Basic exponential smoothing",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("alpha", "float", low=0.01, high=0.99, default=0.1),
        ]


@register_model
class StatsForecastHolt(_BaseStatsForecast):
    """
    StatsForecast Holt - Double Exponential Smoothing.

    Exponential smoothing with additive trend.

    Args:
        season_length: Seasonal period (default 1 for non-seasonal)
        error_type: Error type ('A' additive or 'M' multiplicative)
    """

    def __init__(self, season_length: int = 1, error_type: str = 'A', **kwargs):
        super().__init__(season_length=season_length, error_type=error_type, **kwargs)

    def _create_model(self):
        from statsforecast.models import Holt
        return Holt(
            season_length=self.params.get('season_length', 1),
            error_type=self.params.get('error_type', 'A'),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastHolt",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Holt (1957) / Garza et al. (2022)",
            notes="Double exponential smoothing with trend",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("season_length", "int", low=1, high=52, default=1),
            ParamSpace("error_type", "categorical", choices=['A', 'M'], default='A'),
        ]


@register_model
class StatsForecastHoltWinters(_BaseStatsForecast):
    """
    StatsForecast HoltWinters - Triple Exponential Smoothing.

    Exponential smoothing with trend and seasonality.

    Args:
        season_length: Seasonal period
        error_type: Error type ('A' additive or 'M' multiplicative)
    """

    def __init__(
        self,
        season_length: int = 7,
        error_type: str = 'A',
        **kwargs
    ):
        super().__init__(
            season_length=season_length,
            error_type=error_type,
            **kwargs
        )

    def _create_model(self):
        from statsforecast.models import HoltWinters
        return HoltWinters(
            season_length=self.params.get('season_length', 7),
            error_type=self.params.get('error_type', 'A'),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastHoltWinters",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2022,
            paper="Winters (1960) / Garza et al. (2022)",
            notes="Triple exponential smoothing with trend and seasonality",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("season_length", "int", low=2, high=52, default=7),
            ParamSpace("error_type", "categorical", choices=['A', 'M'], default='A'),
        ]


# @register_model  # ARCHIVED: volatility model, not point forecast
class StatsForecastARCH(_BaseStatsForecast):
    """
    StatsForecast ARCH - Autoregressive Conditional Heteroskedasticity.

    Models time-varying variance. Simpler than GARCH.

    Args:
        p: ARCH lag order
    """

    def __init__(self, p: int = 1, **kwargs):
        super().__init__(p=p, **kwargs)

    def _create_model(self):
        from statsforecast.models import ARCH
        return ARCH(p=self.params.get('p', 1))

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastARCH",
            category=ModelCategory.VOLATILITY,
            library="statsforecast",
            year=2022,
            paper="Engle (1982) / Garza et al. (2022)",
            notes="Forecasts VOLATILITY (σ), not level. Exclude from point forecast benchmarks.",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("p", "int", low=1, high=5, default=1),
        ]


# NOTE: StatsForecastCES removed - CES was deprecated and removed from statsforecast library


# ============================================================================
# ============================================================================


@register_model
class StatsForecastMFLES(_BaseStatsForecast):
    """
    StatsForecast MFLES - Median Fourier Linear Exponential Smoothing.

    New in StatsForecast 1.7.5+. Combines gradient-boosted time series
    decomposition with multiple seasonalities and exogenous features.
    Contributed by Tyler Blume.

    Args:
        season_length: Primary seasonal period
        n_harmonics: Number of Fourier terms for seasonality (default: 4)
        trend_alpha: Smoothing for trend (0-1)
        seasonal_alpha: Smoothing for seasonal (0-1)
    """

    def __init__(
        self,
        season_length: int = 7,
        n_harmonics: int = 4,
        trend_alpha: float = 0.1,
        seasonal_alpha: float = 0.1,
        **kwargs
    ):
        super().__init__(
            season_length=season_length,
            n_harmonics=n_harmonics,
            trend_alpha=trend_alpha,
            seasonal_alpha=seasonal_alpha,
            **kwargs
        )

    def _create_model(self):
        from statsforecast.models import MFLES
        return MFLES(
            season_length=self.params.get('season_length', 7),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="StatsForecastMFLES",
            category=ModelCategory.CLASSICAL,
            library="statsforecast",
            year=2024,
            paper="Blume (2024) - Median Fourier Linear Exponential Smoothing",
            notes="New in v1.7.5+. Handles multiple seasonalities with Fourier terms.",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("season_length", "int", low=2, high=365, default=7),
        ]
