# Modified for forecast-arena: extracted and adapted from AION benchmark utilities, October 2026.
"""
forecasters/wrappers/arch_wrapper.py - ARCH/GARCH volatility models.

ARCHIVED: These models forecast VOLATILITY (σ), not series level.
They always produce sMAPE ≈ 200 on point forecast benchmarks (M3, M4, etc.)
because their output is conditional variance, not the conditional mean.
Removed from model registry to avoid wasting compute on benchmarks.
Re-enable by uncommenting @register_model decorators if needed for
volatility-specific evaluation.

Wraps the `arch` library for advanced volatility forecasting:
- GARCH: Generalized ARCH
- GJR-GARCH: Asymmetric GARCH (leverage effect)
- EGARCH: Exponential GARCH
- FIGARCH: Fractionally integrated GARCH

Install: pip install arch
"""

import numpy as np
from typing import List, Optional, Union
import pandas as pd

from ..base import Forecaster, ModelMetadata, ModelCategory, ParamSpace
from .. import register_model


class _BaseArchForecaster(Forecaster):
    """Base class for arch library volatility models.

    Args:
        forecast_type: What to forecast - 'volatility' (default) returns σ,
                       'mean' returns the conditional mean from the mean model.
    """

    def __init__(self, forecast_type: str = 'volatility', **kwargs):
        super().__init__(forecast_type=forecast_type, **kwargs)
        if forecast_type not in ('volatility', 'mean'):
            raise ValueError(f"forecast_type must be 'volatility' or 'mean', got {forecast_type}")

    def fit(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        freq: Optional[str] = None,
        **kwargs
    ) -> '_BaseArchForecaster':
        from arch import arch_model  # noqa: F401

        y = self._validate_y(y)
        self._train_y = y

        # GARCH models work on returns, not levels
        # User should pre-process if needed
        model = self._create_arch_model(y)
        self._model_result = model.fit(disp='off')
        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ) -> np.ndarray:
        self._check_fitted()

        forecast_type = self.params.get('forecast_type', 'volatility')

        # Use simulation for multi-step forecasts (EGARCH requires this for horizon > 1)
        method = 'simulation' if horizon > 1 else 'analytic'
        try:
            forecasts = self._model_result.forecast(
                horizon=horizon,
                method=method,
                simulations=1000 if method == 'simulation' else None
            )
        except ValueError:
            # Fall back to simulation if analytic fails
            forecasts = self._model_result.forecast(
                horizon=horizon,
                method='simulation',
                simulations=1000
            )

        if forecast_type == 'mean':
            # Return conditional mean forecast
            return forecasts.mean.values[-1, :]
        else:
            # Return volatility forecast (sqrt of variance)
            variance_forecast = forecasts.variance.values[-1, :]
            return np.sqrt(variance_forecast)

    def _compute_fitted_values(self):
        if not hasattr(self, "_model_result"):
            return None
        forecast_type = self.params.get('forecast_type', 'volatility')
        if forecast_type == 'volatility':
            return np.asarray(self._model_result.conditional_volatility)
        # conditional mean: y - resid
        return self._train_y - np.asarray(self._model_result.resid)

    def get_diagnostics(self):
        if not self._is_fitted or not hasattr(self, "_model_result"):
            return {}
        r = self._model_result
        d = {}
        if hasattr(r, "aic"):
            d["aic"] = float(r.aic)
        if hasattr(r, "bic"):
            d["bic"] = float(r.bic)
        if hasattr(r, "params"):
            d["params"] = {k: float(v) for k, v in r.params.items()}
        if hasattr(r, "pvalues"):
            d["pvalues"] = {k: float(v) for k, v in r.pvalues.items()}
        if hasattr(r, "loglikelihood"):
            d["loglikelihood"] = float(r.loglikelihood)
        if hasattr(r, "convergence_flag"):
            d["convergence_flag"] = int(r.convergence_flag)
        # num_params derivable from len(params)
        if hasattr(r, "std_resid") and r.std_resid is not None:
            d["std_resid"] = np.asarray(r.std_resid).tolist()
        # nobs derivable from len(y_train)
        return d

    def _create_arch_model(self, y: np.ndarray):
        raise NotImplementedError


# ============================================================================
# ============================================================================


# @register_model  # ARCHIVED: volatility model, not point forecast
class GJRGARCHForecaster(_BaseArchForecaster):
    """
    GJR-GARCH: Asymmetric GARCH model for volatility forecasting.

    Captures the leverage effect where negative shocks increase volatility
    more than positive shocks of the same magnitude. Essential for
    financial time series modeling.

    Args:
        p: GARCH lag order (default: 1)
        o: Asymmetry lag order (default: 1)
        q: ARCH lag order (default: 1)
        mean: Mean model ('Constant', 'Zero', 'AR') (default: 'Constant')
        dist: Error distribution ('normal', 't', 'skewt') (default: 'normal')
        forecast_type: 'volatility' (default) or 'mean' - what to forecast
    """

    def __init__(
        self,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        mean: str = 'Constant',
        dist: str = 'normal',
        forecast_type: str = 'volatility',
        **kwargs
    ):
        super().__init__(forecast_type=forecast_type, p=p, o=o, q=q, mean=mean, dist=dist, **kwargs)

    def _create_arch_model(self, y: np.ndarray):
        from arch import arch_model
        return arch_model(
            y,
            vol='GARCH',
            p=self.params.get('p', 1),
            o=self.params.get('o', 1),  # GJR asymmetry term
            q=self.params.get('q', 1),
            mean=self.params.get('mean', 'Constant'),
            dist=self.params.get('dist', 'normal'),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="GJRGARCHForecaster",
            category=ModelCategory.VOLATILITY,
            library="arch",
            year=1993,
            paper="Glosten, Jagannathan & Runkle (1993) - On the Relation between Expected Value and Volatility",
            notes="Volatility model. Use forecast_type='mean' for point forecasts.",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("p", "int", low=1, high=5, default=1),
            ParamSpace("o", "int", low=1, high=3, default=1),
            ParamSpace("q", "int", low=1, high=5, default=1),
            ParamSpace("mean", "categorical", choices=['Constant', 'Zero', 'AR'], default='Constant'),
            ParamSpace("dist", "categorical", choices=['normal', 't', 'skewt'], default='normal'),
            ParamSpace("forecast_type", "categorical", choices=['volatility', 'mean'], default='volatility'),
        ]


# ============================================================================
# ============================================================================


# @register_model  # ARCHIVED: volatility model, not point forecast
class EGARCHForecaster(_BaseArchForecaster):
    """
    EGARCH: Exponential GARCH model for volatility forecasting.

    Models log variance, ensuring positive variance without constraints.
    Naturally handles asymmetric effects (leverage).

    Args:
        p: GARCH lag order (default: 1)
        o: Asymmetry lag order (default: 1)
        q: ARCH lag order (default: 1)
        mean: Mean model ('Constant', 'Zero', 'AR') (default: 'Constant')
        dist: Error distribution ('normal', 't', 'skewt') (default: 'normal')
        forecast_type: 'volatility' (default) or 'mean' - what to forecast
    """

    def __init__(
        self,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        mean: str = 'Constant',
        dist: str = 'normal',
        forecast_type: str = 'volatility',
        **kwargs
    ):
        super().__init__(forecast_type=forecast_type, p=p, o=o, q=q, mean=mean, dist=dist, **kwargs)

    def _create_arch_model(self, y: np.ndarray):
        from arch import arch_model
        return arch_model(
            y,
            vol='EGARCH',
            p=self.params.get('p', 1),
            o=self.params.get('o', 1),
            q=self.params.get('q', 1),
            mean=self.params.get('mean', 'Constant'),
            dist=self.params.get('dist', 'normal'),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="EGARCHForecaster",
            category=ModelCategory.VOLATILITY,
            library="arch",
            year=1991,
            paper="Nelson (1991) - Conditional Heteroskedasticity in Asset Returns",
            notes="Volatility model. Use forecast_type='mean' for point forecasts.",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("p", "int", low=1, high=5, default=1),
            ParamSpace("o", "int", low=1, high=3, default=1),
            ParamSpace("q", "int", low=1, high=5, default=1),
            ParamSpace("mean", "categorical", choices=['Constant', 'Zero', 'AR'], default='Constant'),
            ParamSpace("dist", "categorical", choices=['normal', 't', 'skewt'], default='normal'),
            ParamSpace("forecast_type", "categorical", choices=['volatility', 'mean'], default='volatility'),
        ]


# ============================================================================
# Bonus: FIGARCH (Fractionally Integrated)
# ============================================================================


# @register_model  # ARCHIVED: volatility model, not point forecast
class FIGARCHForecaster(_BaseArchForecaster):
    """
    FIGARCH: Fractionally Integrated GARCH model.

    Captures long memory in volatility through fractional integration.
    Useful when volatility shocks have persistent effects.

    Args:
        p: GARCH lag order (default: 1)
        q: ARCH lag order (default: 1)
        mean: Mean model ('Constant', 'Zero', 'AR') (default: 'Constant')
        dist: Error distribution ('normal', 't', 'skewt') (default: 'normal')
        forecast_type: 'volatility' (default) or 'mean' - what to forecast
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        mean: str = 'Constant',
        dist: str = 'normal',
        forecast_type: str = 'volatility',
        **kwargs
    ):
        super().__init__(forecast_type=forecast_type, p=p, q=q, mean=mean, dist=dist, **kwargs)

    def _create_arch_model(self, y: np.ndarray):
        from arch import arch_model
        return arch_model(
            y,
            vol='FIGARCH',
            p=self.params.get('p', 1),
            q=self.params.get('q', 1),
            mean=self.params.get('mean', 'Constant'),
            dist=self.params.get('dist', 'normal'),
        )

    @staticmethod
    def get_metadata() -> ModelMetadata:
        return ModelMetadata(
            name="FIGARCHForecaster",
            category=ModelCategory.VOLATILITY,
            library="arch",
            year=1996,
            paper="Baillie, Bollerslev & Mikkelsen (1996) - Fractionally Integrated GARCH",
            notes="Volatility model. Use forecast_type='mean' for point forecasts.",
        )

    @staticmethod
    def get_param_space() -> List[ParamSpace]:
        return [
            ParamSpace("p", "int", low=1, high=5, default=1),
            ParamSpace("q", "int", low=1, high=5, default=1),
            ParamSpace("mean", "categorical", choices=['Constant', 'Zero', 'AR'], default='Constant'),
            ParamSpace("dist", "categorical", choices=['normal', 't', 'skewt'], default='normal'),
            ParamSpace("forecast_type", "categorical", choices=['volatility', 'mean'], default='volatility'),
        ]
