"""ArimaRegressor: time-series ARIMA regressor with exogenous inputs.

Fits a univariate ARIMA (or SARIMAX with seasonal terms and exogenous
regressors) by directly minimizing the mean-equation residual using
``scipy.optimize.least_squares``, and reduces the result to the standard
:class:`~flexparameterize.regression.base.FitResult` / ``SurrogateSpec``
shape, so every downstream consumer (provenance logging,
``emit_model_config``, ``apply_to_model``) works without change.

``scipy`` is a core runtime dependency, so an explicit ``order`` needs no
extra. ``statsforecast`` ships in the ``[parameterize]`` extra and is used
only when ``auto=True``, for order selection; the final parameter fit
always uses the direct OLS / Levenberg-Marquardt backend that minimizes
the exact residual the Pyomo surrogate implements.

**Restriction**: ``d`` may be ``0`` or ``1``; seasonal differencing
(``D``) must be ``0``, and seasonal AR/MA terms (``P>0`` or ``Q>0``) are not
supported by this backend at all.  The Pyomo surrogate implements the mean
ARIMA equation in closed form, which supports at most a single order of
differencing.  Models with ``d>1``, ``D>0``, or a non-trivial seasonal AR/MA
order will raise ``FlexConfigError``.

Typical usage::

    import pandas as pd
    from flexparameterize.regression.arima import ArimaRegressor

    df = pd.read_csv(
        "imputed_bio_gas_generation.csv", parse_dates=["timestamp"]
    ).set_index("timestamp")
    X = df[["feed_volume_kg", "TS_pct"]]
    y = df[["biogas_m3_hour"]]

    regressor = ArimaRegressor(order=(1, 0, 1)).fit(
        X, y,
        input_units={"feed_volume_kg": "kg", "TS_pct": "%"},
        output_units="m^3/hr",
    )
    result = regressor.to_fit_result()
    spec  = regressor.to_surrogate_spec()

    # Or let statsforecast pick the best order (no differencing), then
    # refit the winning order with scipy for Pyomo compatibility:
    auto_regressor = ArimaRegressor(auto=True, max_p=3, max_q=3).fit(
        X, y,
        input_units={"feed_volume_kg": "kg", "TS_pct": "%"},
        output_units="m^3/hr",
    )

:class:`ArimaRegressor` attributes:
    model: The fitted :class:`_DirectResults` (or ``None`` before
        :meth:`fit`).
    n_samples: Number of rows the fit used, after dropping nulls.
    metrics: ``{"aic": ..., "rmse": ...}`` of the fitted model against ``y``.
    data_window: ``(first, last)`` index value of the rows used.
    exogenous_variables: Column names of the fitted exogenous inputs.
    output_variable: Column name of the fitted output.
    input_units: Units of every fitted exogenous column, keyed by column
        name.  Set by :meth:`fit`.
    output_units: Units of the fitted output column.  Set by :meth:`fit`.
    order: ``(p, d, q)`` order used (or ``None`` when ``auto`` was used).
    seasonal_order: ``(P, D, Q, m)`` seasonal order, or ``None``.
"""

from __future__ import annotations

import math
import warnings
from types import MethodType

import numpy as np
import pandas as pd

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexConfigError, FlexDataError
from flexcore.logger import get_logger
from flexparameterize.regression.base import FitResult

_log = get_logger(__name__)

_FIT_OBJECTIVES = ("equation_error", "output_error")
_FIT_SOLVERS = ("scipy", "ipopt")

# Invertibility bound applied to the MA block in the ipopt backend. Without
# it the solver trades the bounded AR block against MA and drives the MA
# roots outside the unit circle, which makes the innovation recursion those
# coefficients imply explosive.
_IPOPT_MA_BOUND = 0.99


# Upper bound on the automatically chosen free-run window. The best horizon
# is the one the caller actually forecasts over, which this module cannot
# know, so "auto" spans the data up to this cap. The cap matters: measured on
# 5757 rows, an uncapped full-series window left the objective so flat for
# d=0 orders that least_squares ground to its evaluation limit -- 120-140 s
# to remove under 2% of the objective -- while the same fits at 192 steps
# took under 8 s and scored no worse.
_AUTO_MAX_HORIZON = 192


def _auto_forecast_horizon(n_rows: int, p: int, d: int, q: int) -> int:
    """Return the default free-run window length for ``n_rows`` of data.

    Prefer passing ``forecast_horizon`` explicitly: the best value is the
    horizon you intend to forecast over, and no rule here can infer it.

    Intermediate horizons carry a specific hazard worth knowing about. They
    leave the MA block *weakly* identified rather than unidentified -- the MA
    gradient becomes large enough to drag those coefficients around but too
    small to pin them, and in testing they drifted outside the unit circle.
    A fit that does so is reported through the non-invertibility warning in
    :meth:`ArimaRegressor.fit` and the ``ma_max_root`` diagnostic.

    Args:
        n_rows: Number of rows the fit will use.
        p: Autoregressive order.
        d: Differencing order.
        q: Moving-average order.

    Returns:
        The usable row count capped at ``_AUTO_MAX_HORIZON``, raised if
        needed so a window outlasts the ``max(p + d, q)`` rows seeding it.
    """
    seed = max(p + d, q)
    usable = max(n_rows - seed, 1)
    return max(min(usable, _AUTO_MAX_HORIZON), seed + 1)


def _ma_max_root(ma_coefs) -> float:
    """Return the largest MA root magnitude, 0.0 when there is no MA block.

    The MA polynomial ``1 + t1*B + ... + tq*B**q`` is invertible when every
    root of ``z**q + t1*z**(q-1) + ... + tq`` lies inside the unit circle,
    so this value is below 1.0 exactly when the fitted MA is invertible.

    Args:
        ma_coefs: Fitted MA coefficients in lag order.

    Returns:
        The largest root magnitude.
    """
    if len(ma_coefs) == 0:
        return 0.0
    return float(np.abs(np.roots([1.0, *ma_coefs])).max())


class ArimaRegressor:
    """Fit a univariate ARIMA / SARIMAX model using direct OLS/NLS and expose
    the shared fit protocol.

    Either supply an explicit ``order`` (and optional ``seasonal_order``), or
    set ``auto=True`` to delegate order selection to statsforecast
    ``AutoARIMA``, then refit the winning order with the direct scipy
    backend.

    **Restriction**: ``d`` may be ``0`` or ``1`` (the Pyomo surrogate's mean
    ARIMA equation supports at most a single order of differencing).
    Seasonal differencing (``D``) must be ``0``, and seasonal AR/MA terms
    (``P>0`` or ``Q>0``) are not supported by the direct-fit backend at all —
    only a plain, non-seasonal ARIMA (``seasonal_order=None`` or the trivial
    ``(0, 0, 0, m)``) can be fit.

    **Fitting backend**: All final parameter estimation uses
    ``scipy.optimize.least_squares`` (Levenberg-Marquardt) to minimise the
    mean-equation residual directly.  This is the exact objective the Pyomo
    surrogate implements, so fitted parameters reproduce the surrogate
    one-to-one.  Pure AR(p) models use closed-form OLS.

    **AR persistence bound**: AR coefficients are constrained to
    ``[-max_ar_persistence, max_ar_persistence]`` (default 0.85) during the
    fit itself, not merely checked afterward: the Pyomo surrogate implements
    the mean ARIMA equation as a difference equation, and coefficients near
    the unit root (\\|ar\\| >= 1) make the resulting optimisation problem
    numerically unstable regardless of the backend used to estimate them.
    Set ``max_ar_persistence=None`` to fit unconstrained.

    Args:
        order: ``(p, d, q)`` ARIMA order; required when ``auto`` is ``False``.
            Must satisfy ``d in (0, 1)``.
        seasonal_order: ``(P, D, Q, m)`` seasonal order; pass ``None`` (the
            default) for a plain (non-seasonal) ARIMA.  If provided, must
            satisfy ``D == 0`` and ``P == Q == 0`` — the direct-fit backend
            does not support seasonal AR/MA terms, so only the trivial
            ``(0, 0, 0, m)`` (equivalent to ``None``) is accepted.
        include_mean: Whether to include the model's deterministic term.
            This is a level intercept for ``d=0`` and constant drift for
            ``d=1``. Default ``True``.
        include_drift: Whether to include a drift term (constant in the
            differenced series). Default ``False``. For ``d=1`` this is an
            explicit alias for the default ``include_mean`` deterministic
            term; setting it with ``d=0`` raises ``FlexConfigError``.
        auto: If ``True``, run ``statsforecast.models.AutoARIMA`` to
            discover the best ``(p, d, q)`` and ``(P, D, Q, m)``, then
            refit that order with the direct scipy backend for Pyomo
            compatibility.  AutoARIMA may select ``d=0`` or ``d=1``;
            ``D`` is always forced to 0.  Use ``auto_kwargs`` to restrict
            the search (e.g. ``max_d=1``).
        auto_kwargs: Extra keyword arguments forwarded to ``AutoARIMA``
            (e.g. ``max_p``, ``max_q``, ``max_P``, ``season_length``).
            ``D`` defaults to 0; pass ``max_d=1`` to allow differencing.
        max_ar_persistence: Bounds every AR coefficient to
            ``[-max_ar_persistence, max_ar_persistence]`` during the fit
            itself (via bounded least squares). Default ``0.85``. Set to
            ``None`` to fit unconstrained.
        stationary: If ``True``, force ``stationary=True`` in
            ``statsforecast.models.AutoARIMA``, which restricts the
            search to models with stationary AR coefficients.  Default
            ``False``.  This is an additional safety net for the Pyomo
            surrogate; it does not replace the ``max_ar_persistence`` check.

    Raises:
        FlexConfigError: If ``d > 1``, ``D > 0``, or a seasonal AR/MA order
            (``P > 0`` or ``Q > 0``) is requested; if ``include_drift=True``
            is combined with anything other than an explicit ``d=1`` order
            (including ``auto=True``, where no order is known yet); if
            ``max_ar_persistence`` is not a number in ``(0, 1]`` or ``None``;
            or if any fitted AR coefficient exceeds ``max_ar_persistence``.
    """

    def __init__(
        self,
        order: tuple[int, int, int] | None = None,
        seasonal_order: tuple[int, int, int, int] | None = None,
        include_mean: bool = True,
        include_drift: bool = False,
        auto: bool = False,
        max_ar_persistence: float | None = 0.85,
        stationary: bool = False,
        fit_objective: str = "equation_error",
        forecast_horizon: int | str = "auto",
        fit_solver: str = "scipy",
        **auto_kwargs: object,
    ) -> None:
        if include_drift and (order is None or order[1] != 1):
            raise FlexConfigError(
                "ArimaRegressor only supports include_drift=True when d=1. "
                f"Got include_drift=True with order={order}. "
                "Set include_drift=False or use order=(p, 1, q).",
                field="include_drift",
                value=True,
            )
        if order is not None and order[1] != 0:
            if order[1] != 1:
                raise FlexConfigError(
                    f"ArimaRegressor only supports d=0 or d=1. "
                    f"Got order={order} with d={order[1]}. "
                    f"Use order=(p, 0, q) or order=(p, 1, q).",
                    field="order",
                    value=order,
                )
        if seasonal_order is not None and seasonal_order[1] != 0:
            raise FlexConfigError(
                f"ArimaRegressor only supports non-differenced seasonal models "
                f"(D=0). Got seasonal_order={seasonal_order} with "
                f"D={seasonal_order[1]}. Use seasonal_order=(P, 0, Q, m) instead.",
                field="seasonal_order",
                value=seasonal_order,
            )
        if seasonal_order is not None and (
            seasonal_order[0] != 0 or seasonal_order[2] != 0
        ):
            raise FlexConfigError(
                f"ArimaRegressor's direct-fit backend does not support seasonal "
                f"AR/MA terms (P>0 or Q>0). Got seasonal_order={seasonal_order} "
                f"with P={seasonal_order[0]}, Q={seasonal_order[2]}. Use "
                f"seasonal_order=(0, 0, 0, m) or None instead.",
                field="seasonal_order",
                value=seasonal_order,
            )
        if max_ar_persistence is not None and (
            not isinstance(max_ar_persistence, (int, float))
            or max_ar_persistence <= 0
            or max_ar_persistence > 1
        ):
            raise FlexConfigError(
                f"ArimaRegressor max_ar_persistence must be a number in "
                f"(0, 1] or None, got {max_ar_persistence!r}.",
                field="max_ar_persistence",
                value=max_ar_persistence,
            )
        if fit_objective not in _FIT_OBJECTIVES:
            raise FlexConfigError(
                f"ArimaRegressor fit_objective must be one of "
                f"{list(_FIT_OBJECTIVES)}, got {fit_objective!r}.",
                field="fit_objective",
                value=fit_objective,
            )
        if forecast_horizon != "auto" and (
            isinstance(forecast_horizon, bool)
            or not isinstance(forecast_horizon, int)
            or forecast_horizon < 1
        ):
            raise FlexConfigError(
                f"ArimaRegressor forecast_horizon must be a positive int or "
                f'"auto", got {forecast_horizon!r}.',
                field="forecast_horizon",
                value=forecast_horizon,
            )
        if fit_objective == "equation_error" and forecast_horizon != "auto":
            raise FlexConfigError(
                "ArimaRegressor forecast_horizon only affects an "
                'output_error fit; with fit_objective="equation_error" it '
                "would be silently ignored. Drop forecast_horizon or set "
                'fit_objective="output_error".',
                field="forecast_horizon",
                value=forecast_horizon,
            )
        if fit_solver not in _FIT_SOLVERS:
            raise FlexConfigError(
                f"ArimaRegressor fit_solver must be one of "
                f"{list(_FIT_SOLVERS)}, got {fit_solver!r}.",
                field="fit_solver",
                value=fit_solver,
            )
        if fit_solver == "ipopt" and fit_objective != "output_error":
            raise FlexConfigError(
                'ArimaRegressor fit_solver="ipopt" only applies to an '
                "output_error fit; the equation-error fit is a direct "
                "OLS/least-squares solve with no NLP to hand to a solver. "
                'Set fit_objective="output_error" or fit_solver="scipy".',
                field="fit_solver",
                value=fit_solver,
            )
        if fit_solver == "ipopt" and forecast_horizon != "auto":
            raise FlexConfigError(
                'ArimaRegressor fit_solver="ipopt" builds one Pyomo model '
                "over the whole training series, so it cannot window the "
                "free run and forecast_horizon does not apply. Drop "
                'forecast_horizon or use fit_solver="scipy".',
                field="forecast_horizon",
                value=forecast_horizon,
            )

        self._fit_solver = fit_solver
        self._fit_objective = fit_objective
        self._forecast_horizon = (
            None if forecast_horizon == "auto" else int(forecast_horizon)
        )
        self._order = order
        self._seasonal_order = seasonal_order
        self._include_mean = include_mean
        self._include_drift = include_drift
        self._auto = auto
        self._auto_kwargs: dict[str, object] = auto_kwargs
        self._max_ar_persistence = max_ar_persistence
        self._stationary = stationary

        self.model = None
        self.n_samples: int = 0
        self.metrics: dict[str, float] = {}
        self.data_window: tuple = ()
        self.exogenous_variables: list[str] = []
        self.output_variable: str = ""
        self.input_units: dict[str, str] = {}
        self.output_units: str = ""
        self._fitted: bool = False

    def _resolve_horizon(self, n_rows: int, p: int, d: int, q: int) -> int:
        """Resolve and remember the free-run window length for this fit.

        An explicit ``forecast_horizon`` is used verbatim; ``"auto"`` is
        resolved through :func:`_auto_forecast_horizon` once the order is
        known, so that an auto-selected order still gets a matching horizon.

        Args:
            n_rows: Number of rows the fit is using.
            p: Autoregressive order.
            d: Differencing order.
            q: Moving-average order.

        Returns:
            The resolved window length in steps.
        """
        if self._forecast_horizon is None:
            self._forecast_horizon = _auto_forecast_horizon(n_rows, p, d, q)
        return self._forecast_horizon

    def _theta_refiner(self, y_values, x_values, order):
        """Return the output-error refinement callback, or ``None``.

        ``None`` means ``_fit_direct`` uses its built-in scipy refinement.
        The ipopt backend is injected as a callback so that ``_fit_direct``
        stays free of any Pyomo dependency.

        Args:
            y_values: Endogenous series the fit is using.
            x_values: Exogenous regressor matrix, or ``None``.
            order: ``(p, d, q)``.

        Returns:
            A ``(theta, has_const, has_drift, n_exog) -> theta`` callable, or
            ``None``.
        """
        if self._fit_objective != "output_error" or self._fit_solver != "ipopt":
            return None

        def refine(theta, has_const, has_drift, n_exog):
            return _fit_output_error_ipopt(
                theta,
                y_values,
                x_values,
                order=order,
                has_const=has_const,
                has_drift=has_drift,
                n_exog=n_exog,
                exog_names=list(self.exogenous_variables),
                input_units=dict(self.input_units),
                output_name=self.output_variable,
                output_units=self.output_units,
                training_index=self._training_index,
                max_ar_persistence=self._max_ar_persistence,
            )

        return refine

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        *,
        input_units: dict[str, str] | None = None,
        output_units: str | None = None,
    ) -> ArimaRegressor:
        """Fit an ARIMA model to ``y`` with optional exogenous regressors ``X``.

        Uses ``scipy.optimize.least_squares`` to minimize the mean-equation
        residual directly, so the fitted parameters exactly match the Pyomo
        surrogate.  When ``auto=True``, statsforecast ``AutoARIMA`` is used
        for order selection only; the winning order is then refitted with
        scipy.  Pure AR(p) models use closed-form OLS.

        Rows with any null value across ``X``/``y`` are dropped before fitting.

        **Restriction**: ``d`` may be ``0`` or ``1``; ``D`` must be ``0``, and
        seasonal AR/MA terms (``P>0`` or ``Q>0``) are not supported.

        Args:
            X: Zero or more exogenous-input columns. Pass an empty
                ``DataFrame`` for a pure ARIMA fit.
            y: One output column (a one-column ``DataFrame`` or a
                ``Series``).
            input_units: Units of every fitted exogenous column, keyed by its
                column name.  Defaults to ``{}``.  Recorded for
                :meth:`to_surrogate_spec`.
            output_units: Units of the fitted output column.  Defaults to
                ``""``.  Recorded for :meth:`to_surrogate_spec`.

        Returns:
            ``self``, fitted.

        Raises:
            FlexConfigError: If scipy is not installed, ``auto``
                is ``False`` and no ``order`` was supplied, or
                ``input_units`` is missing an entry for one of ``X``'s
                columns.
            FlexDataError: If ``y`` does not hold exactly one column, if no
                rows survive dropping nulls, or if fewer than
                ``max(p, q) + d + k + 1`` rows survive (where ``k`` is the
                number of fitted parameters), which would leave the fit
                rank-deficient. With ``auto=True`` only one row is required,
                because the order is not yet known.
        """
        if not self._auto and self._order is None:
            raise FlexConfigError(
                "ArimaRegressor.fit requires either `order=(p,d,q)` or "
                "`auto=True`. Pass both or set `auto=True`."
            )

        try:
            import scipy.optimize  # noqa: F401
        except ImportError as exc:
            raise FlexConfigError(
                "ArimaRegressor requires scipy. Install it with "
                "`pip install 'flex-pse[parameterize]'`."
            ) from exc

        if input_units is None:
            input_units = {}
        if output_units is None:
            output_units = ""

        missing = [name for name in X.columns if name not in input_units]
        if missing:
            raise FlexConfigError(
                f"fit is missing input_units for {missing}; every input "
                f"column ({list(X.columns)}) needs an entry.",
                field="input_units",
                value=missing,
            )

        self.input_units = dict(input_units)
        self.output_units = output_units

        output = _single_column(y, "y")
        exogenous = _exog_columns(X)
        paired = pd.concat([output.rename("__y__"), exogenous], axis=1).dropna()

        if paired.empty:
            raise FlexDataError(
                "ArimaRegressor has no usable rows after dropping nulls "
                f"(of {len(output)} rows). Supply non-null data.",
                field="y",
            )

        if self._auto:
            min_rows = 1
            detail = ""
        else:
            p, d, q = self._order or (0, 0, 0)
            n_exog = exogenous.shape[1]
            k_params = (1 if self._include_mean else 0) + p + q + n_exog
            # Rows consumed by lags/differencing (max(p, q) + d), plus
            # enough remaining equations to identify every fitted parameter
            # with at least one degree of freedom left over. Without this,
            # e.g. order=(4, 0, 0) on 5 rows "fits" via a rank-deficient,
            # minimum-norm `lstsq` solve that silently returns meaningless
            # coefficients instead of failing loudly.
            min_rows = max(p, q) + d + k_params + 1
            detail = f" to fit order={self._order} ({k_params} parameter(s))"
        if len(paired) < min_rows:
            raise FlexDataError(
                f"ArimaRegressor needs at least {min_rows} row(s){detail}; "
                f"only {len(paired)} survived dropping nulls.",
                field="y",
            )

        self.n_samples = len(paired)
        self.data_window = (paired.index.min(), paired.index.max())
        self.exogenous_variables = list(exogenous.columns)
        self.output_variable = str(output.name)

        y_values = paired["__y__"].values
        x_df = exogenous if not exogenous.empty else None
        self._y_values = y_values
        self._training_index = paired["__y__"].index

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self._auto:
                order, seasonal_order = _auto_select_order(
                    y_values,
                    x_df.values if x_df is not None else None,
                    seasonal=self._seasonal_order is not None,
                    stationary=self._stationary,
                    **self._auto_kwargs,
                )
                P, D, Q, m = seasonal_order or (0, 0, 0, 0)
                if P != 0 or Q != 0:
                    raise FlexConfigError(
                        f"ArimaRegressor auto=True selected a seasonal order "
                        f"with P>0 or Q>0 ({seasonal_order}), which the "
                        f"direct-fit backend does not support. Retry with "
                        f"seasonal_order=None (the default, which also skips "
                        f"the seasonal search) or restrict the search via "
                        f"auto_kwargs (e.g. max_P=0, max_Q=0).",
                        field="seasonal_order",
                        value=seasonal_order,
                    )
                self._order = order
                self._seasonal_order = seasonal_order
                p, d, q = order
                if d > 1:
                    raise FlexConfigError(
                        f"ArimaRegressor auto=True selected d={d}, but the "
                        f"direct-fit backend and Pyomo surrogate only support "
                        f"d=0 or d=1. Retry with auto_kwargs (e.g. max_d=1).",
                        field="order",
                        value=order,
                    )
                fitted_model = _fit_direct(
                    y_values,
                    x_df,
                    order=order,
                    seasonal_order=(P, D, Q, m),
                    include_mean=self._include_mean,
                    include_drift=self._include_drift,
                    max_ar_persistence=self._max_ar_persistence,
                    fit_objective=self._fit_objective,
                    forecast_horizon=self._resolve_horizon(len(paired), p, d, q),
                    refine_theta=self._theta_refiner(
                        y_values, x_df.values if x_df is not None else None, order
                    ),
                )
            else:
                p, d, q = self._order
                P, D, Q, m = self._seasonal_order or (0, 0, 0, 0)
                fitted_model = _fit_direct(
                    y_values,
                    x_df,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, m),
                    include_mean=self._include_mean,
                    include_drift=self._include_drift,
                    max_ar_persistence=self._max_ar_persistence,
                    fit_objective=self._fit_objective,
                    forecast_horizon=self._resolve_horizon(len(paired), p, d, q),
                    refine_theta=self._theta_refiner(
                        y_values,
                        x_df.values if x_df is not None else None,
                        (p, d, q),
                    ),
                )

        self.model = fitted_model
        self._fitted = True
        self._order = _extract_order(fitted_model)
        self._seasonal_order = _extract_seasonal_order(fitted_model)

        if self._max_ar_persistence is not None:
            coef = self.model_.get("coef", {})
            ar_keys = [k for k in coef.keys() if str(k).startswith("ar")]
            for key in ar_keys:
                val = float(coef[key])
                # The fit is bounded to +/-max_ar_persistence (see
                # _fit_ar_ols/_fit_arma_nls), so this can only fire from
                # floating-point slop right at the boundary.
                if abs(val) > self._max_ar_persistence * (1 + 1e-9):
                    raise FlexConfigError(
                        f"ArimaRegressor rejected fitted AR coefficient "
                        f"{key}={val:.4f} because it exceeds "
                        f"max_ar_persistence={self._max_ar_persistence}. "
                        f"High-persistence AR models are unstable in the "
                        f"Pyomo mean-equation surrogate. Use a lower AR order "
                        f"or an MA-only model.",
                        field="ar_coefs",
                        value=val,
                    )

        ma_max_root = _ma_max_root(_collect_lags(self.coefficients or {}, "ma", q))
        if ma_max_root >= 1.0:
            _log.warning(
                "ArimaRegressor fitted a non-invertible MA block for "
                "order=%s: largest MA root magnitude is %.4f, which is not "
                "inside the unit circle. The innovation recursion such "
                "coefficients imply is explosive, so the reported one-step "
                "statistics (rmse, aic, bic, sigma2) and any use of this fit "
                "outside a zero-innovation forward simulation are unreliable. "
                "This is most likely with an intermediate forecast_horizon, "
                "which identifies the MA block weakly rather than not at all; "
                'try the default forecast_horizon="auto", a lower q, or '
                'fit_objective="equation_error".',
                self._order,
                ma_max_root,
            )

        fitted_level = fitted_model.fittedvalues_level
        valid = ~np.isnan(fitted_level)
        residual_ss = float(np.nansum((y_values[valid] - fitted_level[valid]) ** 2))
        rmse = math.sqrt(residual_ss / max(valid.sum(), 1))
        aic = float(fitted_model.aic)

        self.metrics = {
            "aic": aic,
            "rmse": rmse,
            "free_run_rmse": fitted_model.free_run_rmse,
        }
        return self

    @property
    def model_(self) -> dict[str, object]:
        """A dict-like view of the fitted model's key attributes.

        Provides ``"coef"``, ``"residuals"``, ``"aic"``, ``"bic"``,
        ``"aicc"``, ``"loglik"``, and ``"sigma2"`` keys.

        The direct backend names parameters ``ar1``, ``ma1``, ``const``,
        ``drift``, and exogenous column names directly, so no name
        normalisation is needed.

        ``"residuals"`` is the full-length array, whose leading
        ``max(p, q)`` entries are zero-padded because the mean equation
        defines no residual there. Every derived statistic (``"aic"``,
        ``"bic"``, ``"aicc"``, ``"loglik"``, ``"sigma2"``) excludes that
        padding. ``"aicc"`` is ``None`` when too few effective observations
        remain for its correction term to be defined.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        if not self._fitted or self.model is None:
            raise FlexDataError(
                "ArimaRegressor has no fit yet; call fit(X, y) before "
                "accessing model attributes."
            )
        m = self.model
        coef = {
            str(k): float(v) for k, v in zip(m.model.param_names, m.params, strict=True)
        }
        sigma2 = coef.pop("sigma2", float(m.sigma2))

        return {
            "coef": coef,
            "residuals": np.asarray(m.resid),
            "aic": float(m.aic),
            "bic": float(m.bic),
            "aicc": _aicc_from_direct(m),
            "loglik": float(m.llf),
            "sigma2": float(sigma2),
        }

    @property
    def coefficients(self) -> dict[str, float] | None:
        """The fitted model's parameters as a name -> value mapping.

        Returns:
            The parameter map, or ``None`` before :meth:`fit` is called.
        """
        if not self._fitted or self.model is None:
            return None
        return {
            str(k): float(v)
            for k, v in zip(
                self.model.model.param_names, self.model.params, strict=True
            )
        }

    @property
    def order(self) -> tuple[int, int, int] | None:
        """The fitted ``(p, d, q)`` order, or ``None`` before :meth:`fit`."""
        return self._order

    @property
    def seasonal_order(self) -> tuple[int, int, int, int] | None:
        """The fitted seasonal order, or ``None`` before :meth:`fit`."""
        return self._seasonal_order

    @property
    def fit_objective(self) -> str:
        """The residual this fit minimizes.

        ``"output_error"`` (the default) minimizes windowed free-run error,
        the criterion the Pyomo surrogate exercises when it simulates
        forward with its innovations fixed at zero. ``"equation_error"``
        minimizes one-step-ahead residuals using actual lagged values.
        """
        return self._fit_objective

    @property
    def fit_solver(self) -> str:
        """The optimizer behind an ``output_error`` fit.

        ``"scipy"`` (the default) uses ``scipy.optimize.least_squares`` on
        the windowed free-run residual. ``"ipopt"`` instead builds the real
        :class:`~flexops.surrogates.arima.ArimaSurrogate` over the training
        series and minimizes the squared output error through it, so the
        fitted coefficients are optimal for the exact Pyomo equation rather
        than for a Python transcription of it. Irrelevant to an
        ``equation_error`` fit, which is a direct least-squares solve.
        """
        return self._fit_solver

    @property
    def forecast_horizon(self) -> int | None:
        """The free-run window length in steps.

        An explicit horizon is readable immediately; ``"auto"`` resolves
        during :meth:`fit`, so this is ``None`` until then.
        """
        return self._forecast_horizon

    @property
    def fitted(self) -> bool:
        """``True`` once :meth:`fit` has succeeded."""
        return self._fitted

    def _params(self) -> dict[str, float]:
        """Return the fitted model's named parameters as a flat dict.

        Returns:
            Mapping of parameter name to its fitted value.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        if not self._fitted or self.model is None:
            raise FlexDataError(
                "ArimaRegressor has no fit yet; call fit(X, y) before "
                "accessing fitted parameters."
            )
        return dict(self.coefficients)

    def to_fit_result(self) -> FitResult:
        """Return this fit as the shared :class:`~.base.FitResult` shape.

        The coefficient map contains:

        - ``"const"`` (``d=0``) or ``"drift"`` (``d=1``) when the model has
          a deterministic term, and neither when it does not,
        - ``"ar.L{j}"`` for each AR lag ``j``,
        - ``"ma.L{j}"`` for each MA lag ``j``,
        - exogenous coefficients keyed by their column name.

        Returns:
            A :class:`~flexparameterize.regression.base.FitResult` carrying
            the model's coefficients, ``aic``, ``rmse``, sample count, and
            data window.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        params = self._params()
        p, d, q = self._order  # type: ignore[misc]

        ar_coefs = _collect_lags(params, "ar", p)
        ma_coefs = _collect_lags(params, "ma", q)
        exog_coefs = [params.get(col, 0.0) for col in self.exogenous_variables]

        coefficients = {
            **{f"ar.L{j}": v for j, v in enumerate(ar_coefs, 1)},
            **{f"ma.L{j}": v for j, v in enumerate(ma_coefs, 1)},
            **dict(zip(self.exogenous_variables, exog_coefs, strict=True)),
        }
        deterministic_name = "const" if d == 0 else "drift"
        if deterministic_name in params:
            coefficients[deterministic_name] = params[deterministic_name]

        return FitResult(
            coefficients=coefficients,
            metrics=dict(self.metrics),
            n_samples=self.n_samples,
            data_window=self.data_window,
        )

    def to_surrogate_spec(self) -> SurrogateSpec:
        """Return the fit as a persistable ``arima`` ``SurrogateSpec``.

        Uses the ``input_units``/``output_units`` recorded by :meth:`fit`.

        The ``data`` field of the returned spec matches the contract expected
        by :class:`~flexops.surrogates.arima.ArimaSurrogate`:

        - ``input_variables``: all fitted exogenous variable names and units.
        - ``output_variables``: the output variable name and its units.
        - ``order``: the fitted ``(p, d, q)`` tuple.
        - ``intercept``: the fitted level intercept, when ``d=0`` and the
          fit included a deterministic term. Omitted entirely when
          ``include_mean=False``, which the surrogate reads as "no
          intercept" rather than "intercept of zero".
        - ``drift``: the fitted differenced-equation constant, when ``d=1``
          and the fit included a deterministic term.
        - ``ar_coefs``: list of AR coefficients in lag order.
        - ``ma_coefs``: list of MA coefficients in lag order.
        - ``exog_coefs``: list of exogenous coefficients, one per column in
          fitted order.

        The ``history`` field carries ``start_date``, ``time_step_seconds``,
        and the disturbance/innovation series the surrogate replays to seed
        its pre-horizon lags. Because no true pre-sample data exists, the
        first ``max(p + d, q)`` entries repeat the start of the series.

        Returns:
            A :class:`~flexcore.config.schema.SurrogateSpec` of type
            ``SurrogateType.ARIMA``.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
            FlexConfigError: If ``input_units`` is missing an entry for a
                fitted exogenous column.
        """
        if not self._fitted:
            raise FlexDataError(
                "ArimaRegressor has no fit yet; call fit(X, y) before "
                "to_surrogate_spec()."
            )

        missing = [
            name for name in self.exogenous_variables if name not in self.input_units
        ]
        if missing:
            raise FlexConfigError(
                f"to_surrogate_spec is missing input_units for {missing}; "
                f"every fitted exogenous column ({self.exogenous_variables}) "
                f"needs an entry.",
                field="input_units",
                value=missing,
            )

        p, d, q = self._order  # type: ignore[misc]
        params = self._params()

        ar_coefs = _collect_lags(params, "ar", p)
        ma_coefs = _collect_lags(params, "ma", q)
        exog_coefs = [params.get(col, 0.0) for col in self.exogenous_variables]

        coefficients: dict[str, object] = {"order": [int(p), int(d), int(q)]}
        if d == 0 and "const" in params:
            coefficients["intercept"] = float(params["const"])
        elif d == 1 and "drift" in params:
            coefficients["drift"] = float(params["drift"])
        if p > 0:
            coefficients["ar_coefs"] = [float(v) for v in ar_coefs]
        if q > 0:
            coefficients["ma_coefs"] = [float(v) for v in ma_coefs]
        if len(self.exogenous_variables) > 0:
            coefficients["exog_coefs"] = [float(v) for v in exog_coefs]

        if len(self.exogenous_variables) > 0 and self.model._x is not None:
            beta_vec = np.array(
                [params.get(col, 0.0) for col in self.exogenous_variables], dtype=float
            )
            eta_series = self._y_values - self.model._x @ beta_vec
        else:
            eta_series = self._y_values

        history = _history_payload(
            eta_series,
            np.asarray(self.model_["residuals"]),
            p=p,
            d=d,
            q=q,
            start_date=self._training_index[0].isoformat(),
            time_step_seconds=_step_seconds_from_index(self._training_index),
        )

        return SurrogateSpec(
            surrogate_type=SurrogateType.ARIMA,
            data={
                "input_variables": {
                    name: self.input_units[name] for name in self.exogenous_variables
                },
                "output_variables": {self.output_variable: self.output_units},
                "coefficients": coefficients,
                "history": history,
            },
        )

    def fit_diagnostics(self) -> dict[str, float | None]:
        """Return extended fit statistics from the underlying direct fit result.

        Includes AIC, BIC, AICc, log-likelihood, and residual standard
        deviation alongside the standard ``aic`` and ``rmse``. All of them
        exclude the zero-padded leading lags, so ``n_samples`` here is the
        number of observations the mean equation defines, which is
        ``max(p, q)`` fewer than the regressor's ``n_samples``.

        ``ma_max_root`` is the largest MA root magnitude: below 1.0 the
        fitted MA block is invertible, at or above it the fit logged a
        warning and its one-step statistics cannot be trusted.

        Returns:
            Mapping of diagnostic name to value. ``aicc`` is ``None`` when
            too few effective observations remain to define it.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        if not self._fitted or self.model is None:
            raise FlexDataError(
                "ArimaRegressor has no fit yet; call fit(X, y) before "
                "fit_diagnostics()."
            )
        m_ = self.model_
        n = self.model.nobs_effective
        params = self._params()
        k = len(params)
        llf = float(m_["loglik"])
        aic = float(m_["aic"])
        bic = float(m_["bic"])
        aicc = m_["aicc"]
        sigma2 = float(m_["sigma2"])
        fitted_level = self.model.fittedvalues_level
        valid = ~np.isnan(fitted_level)
        rmse = math.sqrt(
            float(np.nansum((self._y_values[valid] - fitted_level[valid]) ** 2))
            / max(valid.sum(), 1)
        )

        return {
            "aic": aic,
            "bic": bic,
            "aicc": aicc,
            "log_likelihood": llf,
            "n_parameters": float(k),
            "n_samples": float(n),
            "sigma2": sigma2,
            "rmse": rmse,
            "free_run_rmse": self.model.free_run_rmse,
            "forecast_horizon": float(self.model.forecast_horizon),
            "ma_max_root": _ma_max_root(_collect_lags(params, "ma", self._order[2])),
        }


# -- helpers -------------------------------------------------------------------


def _single_column(frame: pd.DataFrame | pd.Series, role: str) -> pd.Series:
    """Return the one column of ``frame`` as a named Series.

    Args:
        frame: A one-column ``DataFrame`` or a ``Series``.
        role: ``"X"`` or ``"y"``, for the error message.

    Returns:
        The column as a named ``Series``.

    Raises:
        FlexDataError: If ``frame`` does not hold exactly one column.
    """
    if isinstance(frame, pd.Series):
        return frame
    if frame.shape[1] != 1:
        raise FlexDataError(
            f"ArimaRegressor fits one output column; {role} has "
            f"{frame.shape[1]} ({list(frame.columns)}). Select the "
            "single output column.",
            field=role,
        )
    return frame.iloc[:, 0]


def _exog_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Return exogenous columns as a DataFrame (empty when no columns).

    Args:
        X: Zero or more exogenous input columns.

    Returns:
        A ``DataFrame`` (possibly empty) with the same index as ``X``.
    """
    if isinstance(X, pd.Series):
        return X.to_frame()
    return X if X.shape[1] > 0 else pd.DataFrame(index=X.index)


def _collect_lags(params: dict[str, float], prefix: str, n: int) -> list[float]:
    """Collect ``params[prefix{j}]`` for ``j = 1 … n``.

    Missing lags default to 0.0 so the resulting list always has exactly
    ``n`` entries, matching the AR/MA order the caller declared.

    Args:
        params: Fitted parameter mapping (name -> value).
        prefix: Parameter name prefix, either ``"ar"`` or ``"ma"``.
        n: Number of lags to collect.

    Returns:
        A list of ``n`` floats.
    """
    return [params.get(f"{prefix}{j}", 0.0) for j in range(1, n + 1)]


def _step_seconds_from_index(training_index) -> float:
    """Return the sampling step of ``training_index`` in seconds.

    Args:
        training_index: The index of the fitted rows.

    Returns:
        The declared frequency when the index carries one, otherwise the
        gap between the first two rows, falling back to one hour for a
        single-row index.
    """
    if getattr(training_index, "freq", None) is not None:
        return float(pd.Timedelta(training_index.freq).total_seconds())
    if len(training_index) > 1:
        return float((training_index[1] - training_index[0]).total_seconds())
    return 3600.0


def _history_payload(
    eta_series: np.ndarray,
    residuals: np.ndarray,
    *,
    p: int,
    d: int,
    q: int,
    start_date: str,
    time_step_seconds: float,
) -> dict[str, object]:
    """Build the surrogate ``history`` block from a fitted disturbance series.

    The surrogate replays ``max(p + d, q)`` values ahead of its first
    modeled point. No true pre-sample data exists, so the prefix repeats the
    start of the series; both lists are padded to the same prefix length so
    a single offset indexes into either.

    Args:
        eta_series: Disturbance series ``y - X @ beta`` on the level scale.
        residuals: One-step innovations, aligned to the differenced series.
        p: Autoregressive order.
        d: Differencing order.
        q: Moving-average order.
        start_date: ISO-8601 timestamp of the first fitted row.
        time_step_seconds: Sampling step in seconds.

    Returns:
        The ``history`` mapping in the surrogate's persisted contract.
    """
    history_prefix = max(p + d, q)
    eta = np.asarray(eta_series, dtype=float)
    prefix = [float(eta[0])] * (history_prefix - (p + d)) + [
        float(value) for value in eta[: p + d]
    ]
    return {
        "start_date": start_date,
        "time_step_seconds": float(time_step_seconds),
        "y_values": prefix + [float(value) for value in eta],
        "eps_values": [0.0] * (history_prefix + d)
        + [float(value) for value in np.asarray(residuals, dtype=float)],
    }


def _extract_order(fitted_model) -> tuple[int, int, int]:
    """Return ``(p, d, q)`` from a fitted :class:`_DirectResults`.

    Args:
        fitted_model: A fitted result object with ``model.k_ar``,
            ``model.k_diff``, and ``model.k_ma`` attributes.

    Returns:
        ``(p, d, q)`` as a tuple of ints.
    """
    return (
        int(fitted_model.model.k_ar),
        int(fitted_model.model.k_diff),
        int(fitted_model.model.k_ma),
    )


def _extract_seasonal_order(fitted_model) -> tuple[int, int, int, int] | None:
    """Return ``(P, D, Q, m)`` from a fitted result, or ``None``.

    Returns ``None`` when all seasonal terms are zero (i.e. a plain
    non-seasonal ARIMA).

    Args:
        fitted_model: A fitted result object with a ``model.seasonal_order``
            attribute.

    Returns:
        ``(P, D, Q, m)`` as a tuple of ints, or ``None`` when all seasonal
        terms are zero.
    """
    m_spec = fitted_model.model.seasonal_order
    P, D, Q, m = int(m_spec[0]), int(m_spec[1]), int(m_spec[2]), int(m_spec[3])
    if P == 0 and D == 0 and Q == 0:
        return None
    return (P, D, Q, m)


# -- Direct OLS/NLS fitting backend -------------------------------------------


class _DirectModel:
    """Structural descriptor for a direct-fit ARIMA model.

    This lightweight container mirrors the structural attributes of a
    statsmodels ``SARIMAX`` model so that the rest of
    ``ArimaRegressor`` (``_extract_order``, ``_extract_seasonal_order``,
    ``to_fit_result``, ``to_surrogate_spec``) can read fitted-model
    metadata without caring which backend produced it.

    Notes:
        Final parameter estimation always uses the direct scipy fit in
        this module; ``statsforecast`` is only ever used for
        ``auto=True`` order *selection*, never for final estimation.

    Attributes:
        param_names: Ordered parameter names, e.g.
            ``["const", "ar1", "ma1", "feed_volume_kg"]``.
        k_ar: Number of AR terms (``p``).
        k_ma: Number of MA terms (``q``).
        k_diff: Differencing order (``d``).
        seasonal_order: Seasonal ``(P, D, Q, m)`` tuple, or ``(0, 0, 0, 0)``
            when no seasonal terms are present.
        has_const: Whether the model includes a constant/intercept term.
        has_drift: Whether the model includes a drift term (only possible
            when ``d == 1``).
    """

    def __init__(
        self,
        param_names: list[str],
        k_ar: int,
        k_ma: int,
        k_diff: int,
        seasonal_order: tuple[int, int, int, int],
        has_const: bool,
        has_drift: bool = False,
    ) -> None:
        self._param_names = list(param_names)
        self.k_ar = k_ar
        self.k_ma = k_ma
        self.k_diff = k_diff
        self.seasonal_order = seasonal_order
        self.has_const = has_const
        self.has_drift = has_drift

    @property
    def param_names(self) -> list[str]:
        """Ordered names of all fitted parameters."""
        return self._param_names


class _DirectResults:
    """Fitted ARIMA model result, mirroring the statsmodels
    ``SARIMAXResults`` interface.

    Returned by :func:`_fit_direct` so the rest of ``ArimaRegressor`` can
    consume fitted models without knowing which backend produced them.

    Attributes:
        params: Fitted parameter vector in the same order as
            ``model.param_names``.
        resid: In-sample residuals, aligned to the differenced series
            ``y`` (length ``nobs``; the leading ``max(k_ar, k_ma)`` entries
            are zero-padded, since the mean equation defines no residual
            there).
        fittedvalues: In-sample fitted values on the *differenced* scale
            when ``d > 0``; on the level scale when ``d == 0``.
        free_run_resid: Windowed free-run residuals, the error the Pyomo
            surrogate commits when simulating forward.
        forecast_horizon: Free-run window length in steps.
        nobs: Number of effective observations (length of the differenced
            series when ``d > 0``).
        df_model: Number of fitted parameters, including the constant if
            present.
        llf: Log-likelihood computed from the residual sum of squares.
        aic: Akaike information criterion.
        bic: Bayesian information criterion.
        sigma2: Residual variance estimate.

    Note:
        ``llf``, ``aic``, ``bic``, and ``sigma2`` are all computed over
        :attr:`effective_resid` -- that is, with the zero-padded leading
        lags excluded -- so they use ``nobs_effective`` observations rather
        than ``nobs``.
    """

    def __init__(
        self,
        params: np.ndarray,
        residuals: np.ndarray,
        fitted_values: np.ndarray,
        y: np.ndarray,
        k_params: int,
        param_names: list[str],
        k_ar: int,
        k_ma: int,
        k_diff: int,
        seasonal_order: tuple[int, int, int, int],
        has_const: bool,
        has_drift: bool,
        y_original: np.ndarray,
        x: np.ndarray | None,
        fittedvalues_level: np.ndarray,
        free_run_resid: np.ndarray,
        forecast_horizon: int,
    ) -> None:
        self._params = np.asarray(params, dtype=float)
        self._resid = np.asarray(residuals, dtype=float)
        self._fittedvalues = np.asarray(fitted_values, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._y_original = np.asarray(y_original, dtype=float)
        self._x = np.asarray(x, dtype=float) if x is not None else None
        self._fittedvalues_level = np.asarray(fittedvalues_level, dtype=float)
        self._free_run_resid = np.asarray(free_run_resid, dtype=float)
        self._forecast_horizon = int(forecast_horizon)
        self._k_params = k_params
        self._model = _DirectModel(
            param_names, k_ar, k_ma, k_diff, seasonal_order, has_const, has_drift
        )

    @property
    def params(self) -> np.ndarray:
        """Fitted parameter vector, ordered to match :attr:`model.param_names`."""
        return self._params

    @property
    def model(self) -> _DirectModel:
        """Structural model descriptor (order, seasonal order, etc.)."""
        return self._model

    @property
    def resid(self) -> np.ndarray:
        """In-sample residuals from the mean equation.

        Length equals ``nobs``; leading lags up to ``max(k_ar, k_ma)``
        are zero-padded because no fitted values are defined there.
        Use :attr:`effective_resid` for the residuals the mean equation
        actually defines.
        """
        return self._resid

    @property
    def effective_resid(self) -> np.ndarray:
        """Residuals with the zero-padded leading lags dropped."""
        return self._resid[max(self._model.k_ar, self._model.k_ma) :]

    @property
    def fittedvalues(self) -> np.ndarray:
        """In-sample fitted values.

        On the differenced scale when ``k_diff > 0``; on the level scale
        when ``k_diff == 0``.  Use :attr:`fittedvalues_level` to always
        obtain level-scale fitted values.
        """
        return self._fittedvalues

    @property
    def fittedvalues_level(self) -> np.ndarray:
        """In-sample fitted values integrated back to the level scale.

        ``NaN`` wherever the mean equation defines no fitted value.
        """
        return self._fittedvalues_level

    @property
    def free_run_resid(self) -> np.ndarray:
        """Windowed free-run residuals at the fitted parameters.

        See :func:`_output_error_residuals`. Reported for both fitting
        objectives, since it is the error the Pyomo surrogate commits when
        it simulates forward.
        """
        return self._free_run_resid

    @property
    def free_run_rmse(self) -> float:
        """Root-mean-square of :attr:`free_run_resid`."""
        if self._free_run_resid.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(self._free_run_resid**2)))

    @property
    def forecast_horizon(self) -> int:
        """Free-run window length, in steps, used for the output-error fit."""
        return self._forecast_horizon

    @property
    def nobs(self) -> int:
        """Number of observations in the fitted (differenced) series."""
        return int(len(self._y))

    @property
    def nobs_effective(self) -> int:
        """Number of observations the mean equation actually defines.

        This is :attr:`nobs` less the ``max(k_ar, k_ma)`` leading lags that
        carry no residual, and is the sample size used by :attr:`llf`,
        :attr:`aic`, :attr:`bic`, and :attr:`sigma2`.
        """
        return int(len(self.effective_resid))

    @property
    def df_model(self) -> int:
        """Number of fitted parameters, counting the constant if present."""
        return int(self._k_params)

    def _llf(self) -> float:
        resid = self.effective_resid
        n_eff = len(resid)
        rss = float(np.sum(resid**2))
        if rss <= 0 or n_eff == 0:
            return -np.inf
        return (
            -n_eff / 2.0 * np.log(2.0 * np.pi)
            - n_eff / 2.0 * np.log(rss / n_eff)
            - n_eff / 2.0
        )

    @property
    def llf(self) -> float:
        """Gaussian log-likelihood over :attr:`effective_resid`."""
        return self._llf()

    @property
    def aic(self) -> float:
        """Akaike information criterion, from :attr:`llf`."""
        k = self.df_model + 1
        return -2.0 * self.llf + 2.0 * k

    @property
    def bic(self) -> float:
        """Bayesian information criterion, penalized by ``nobs_effective``."""
        k = self.df_model + 1
        n = self.nobs_effective
        if n == 0:
            return np.inf
        return -2.0 * self.llf + k * np.log(n)

    @property
    def sigma2(self) -> float:
        """Residual variance over :attr:`effective_resid`."""
        resid = self.effective_resid
        n_eff = len(resid)
        return float(np.sum(resid**2)) / n_eff if n_eff > 0 else 0.0

    def predict(
        self,
        steps: int = 1,
        exog: np.ndarray | None = None,
        start: int | None = None,
        dynamic: bool = True,
    ) -> np.ndarray:
        """Recursive multi-step forecast using the mean equation.

        Forecasts are generated recursively: each predicted value feeds
        back as the AR lag for the next step, and MA terms are zeroed after
        the first step (matching the Pyomo surrogate's forecast behaviour).
        ``steps=0`` returns an empty array; use :attr:`fittedvalues` or
        :attr:`fittedvalues_level` for in-sample values.

        Args:
            steps: Number of steps to forecast ahead.
            exog: Future exogenous values of shape ``(steps, n_exog)``.
                Pass ``None`` when the model has no exogenous regressors.
            start: Optional start index within the training data for
                in-sample dynamic prediction.  When ``start`` is provided,
                the method uses training data up to ``start`` as initial
                history and recursively predicts from ``start`` onward.
                This matches the Pyomo surrogate's 0DOF solve behaviour.
            dynamic: Must be ``True`` (the default). Recursive prediction
                (each predicted value feeds back as an AR lag) is the only
                mode this direct-fit backend implements; one-step-ahead
                prediction using actual observed lags is not implemented.

        Returns:
            Array of ``steps`` forecasts on the level scale of ``y``,
            including the exogenous contribution when ``exog`` is given.

        Raises:
            FlexConfigError: If ``dynamic=False`` is passed.
        """
        if not dynamic:
            raise FlexConfigError(
                "ArimaRegressor's direct-fit predict() only implements "
                "dynamic=True (recursive) prediction; one-step-ahead "
                "dynamic=False prediction is not implemented. Use "
                "`fittedvalues` for in-sample one-step-ahead values, or "
                "omit `dynamic` (it defaults to True)."
            )
        p = self.model.k_ar
        q = self.model.k_ma
        _d = self.model.k_diff
        has_const = self.model.has_const
        has_drift = self.model.has_drift

        idx = 0
        c = float(self._params[idx]) if (has_const or has_drift) else 0.0
        idx += has_const or has_drift

        ar = self._params[idx : idx + p]
        idx += p
        ma = self._params[idx : idx + q]
        idx += q
        n_exog = self._params.shape[0] - idx
        beta = self._params[idx : idx + n_exog] if n_exog > 0 else np.zeros(0)

        y_source = self._y_original if self._y_original is not None else self._y
        if n_exog > 0 and self._x is not None:
            eta_source = y_source - self._x @ beta
        else:
            eta_source = y_source

        if start is not None:
            # In-sample dynamic prediction: use disturbance data BEFORE `start`
            # as initial history, then recursively predict forward from `start`.
            if _d == 0:
                eta_hist = list(eta_source[max(0, start - p) : start])
                eps_hist = list(self._resid[max(0, start - q) : start]) if q > 0 else []
            else:
                eta_hist = list(eta_source[max(0, start - p - 1) : start])
                eps_hist = (
                    list(self._resid[max(0, start - q - _d) : start - _d])
                    if q > 0
                    else []
                )
                if len(eta_hist) < p + 1:
                    eta_hist = list(eta_source[: p + 1])
            forecasts = []

            for step in range(steps):
                if _d == 0:
                    ar_part = sum(ar[j] * eta_hist[-(j + 1)] for j in range(p))
                    ma_part = (
                        sum(ma[j] * eps_hist[-(j + 1)] for j in range(q))
                        if q > 0 and len(eps_hist) >= q
                        else 0.0
                    )
                    eta_hat = c + ar_part + ma_part
                else:
                    ar_diff_part = sum(
                        ar[j] * (eta_hist[-(j + 1)] - eta_hist[-(j + 2)])
                        for j in range(p)
                    )
                    ma_part = (
                        sum(ma[j] * eps_hist[-(j + 1)] for j in range(q))
                        if q > 0 and len(eps_hist) >= q
                        else 0.0
                    )
                    eta_hat = eta_hist[-1] + c + ar_diff_part + ma_part

                exog_part = (
                    sum(beta[k] * exog[step, k] for k in range(n_exog))
                    if exog is not None and n_exog > 0
                    else 0.0
                )
                y_hat = exog_part + eta_hat
                forecasts.append(y_hat)
                eta_hist.append(eta_hat)
                if q > 0:
                    eps_hist.append(0.0)

            return np.array(forecasts)

        # Out-of-sample forecast from end of training data
        if _d == 0:
            eta_hist = list(eta_source[-p:] if p > 0 else [])
        else:
            eta_hist = list(eta_source[-(p + 1) :] if p > 0 else eta_source[-2:])
        eps_hist = list(self._resid[-q:] if q > 0 else [])
        forecasts = []

        for step in range(steps):
            if _d == 0:
                ar_part = sum(ar[j] * eta_hist[-(j + 1)] for j in range(p))
                ma_part = sum(ma[j] * eps_hist[-(j + 1)] for j in range(q))
                eta_hat = c + ar_part + ma_part
            else:
                ar_diff_part = sum(
                    ar[j] * (eta_hist[-(j + 1)] - eta_hist[-(j + 2)]) for j in range(p)
                )
                ma_part = sum(ma[j] * eps_hist[-(j + 1)] for j in range(q))
                eta_hat = eta_hist[-1] + c + ar_diff_part + ma_part

            exog_part = (
                sum(beta[k] * exog[step, k] for k in range(n_exog))
                if exog is not None and n_exog > 0
                else 0.0
            )
            y_hat = exog_part + eta_hat
            forecasts.append(y_hat)
            eta_hist.append(eta_hat)
            if q > 0:
                eps_hist.append(0.0)

        return np.array(forecasts)


def _fit_pure_regression(
    y: np.ndarray,
    x_values: np.ndarray | None,
    exog_names: list[str],
    has_const: bool,
    has_drift: bool,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Fit the no-lag case ``y = c + drift + exog + noise`` via OLS.

    Used when ``p == q == 0``.

    Args:
        y: Endogenous series (already differenced when ``d > 0``).
        x_values: Exogenous regressor matrix, or ``None``.
        exog_names: Column names matching ``x_values``.
        has_const: Whether to fit a level intercept named ``const``.
        has_drift: Whether to fit a differenced-equation constant named
            ``drift``. At most one of ``has_const``/``has_drift`` is set by
            :func:`_fit_direct`; both share the same column of ones.

    Returns:
        ``(theta, param_names, residuals, fitted, fitted_level)``. With no
        regressors at all, ``theta`` is empty and the series mean is used.
    """
    n = len(y)
    cols: list[np.ndarray] = []
    if has_const or has_drift:
        cols.append(np.ones(n))
    if x_values is not None and x_values.shape[1] > 0:
        for k in range(x_values.shape[1]):
            cols.append(x_values[:, k])

    param_names: list[str] = []
    if has_const:
        param_names.append("const")
    if has_drift:
        param_names.append("drift")
    param_names.extend(exog_names)

    if not cols:
        c = float(np.mean(y))
        residuals = y - c
        fitted = np.full(n, c)
        return np.array([]), param_names, residuals, fitted, fitted

    X = np.column_stack(cols)
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ theta
    return theta, param_names, y - y_hat, y_hat, y_hat


def _fit_ar_ols(
    y: np.ndarray,
    p: int,
    x_values: np.ndarray | None,
    exog_names: list[str],
    has_const: bool,
    has_drift: bool,
    max_ar_persistence: float | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Fit a pure AR(p) model by (optionally bounded) linear least squares.

    Only reached when there are no exogenous regressors, so ``x_values``
    and ``exog_names`` are accepted for a uniform helper signature and are
    expected to be empty.

    Args:
        y: Endogenous series (already differenced when ``d > 0``).
        p: Autoregressive order, at least 1.
        x_values: Unused; expected to be ``None`` or empty.
        exog_names: Unused; expected to be empty.
        has_const: Whether to fit a level intercept named ``const``.
        has_drift: Whether to fit a differenced-equation constant named
            ``drift``.
        max_ar_persistence: When set, bounds every AR coefficient to
            ``[-max_ar_persistence, max_ar_persistence]`` via
            ``scipy.optimize.lsq_linear`` instead of an unbounded
            ``numpy.linalg.lstsq``.

    Returns:
        ``(theta, param_names, residuals, fitted, fitted_level)``.
        ``residuals`` has the length of ``y`` with its first ``p`` entries
        zero-padded; ``fitted`` is ``NaN`` there.
    """
    n = len(y)
    max_lag = p
    T = n - max_lag

    cols: list[np.ndarray] = []
    if has_const or has_drift:
        cols.append(np.ones(T))
    for j in range(1, p + 1):
        cols.append(y[max_lag - j : n - j])

    param_names: list[str] = []
    if has_const:
        param_names.append("const")
    if has_drift:
        param_names.append("drift")
    for j in range(1, p + 1):
        param_names.append(f"ar{j}")
    param_names.extend(exog_names)

    X = np.column_stack(cols)
    y_vec = y[max_lag:]
    if max_ar_persistence is None:
        theta, _, _, _ = np.linalg.lstsq(X, y_vec, rcond=None)
    else:
        from scipy.optimize import lsq_linear

        ar_offset = has_const + has_drift
        lb = np.full(X.shape[1], -np.inf)
        ub = np.full(X.shape[1], np.inf)
        lb[ar_offset : ar_offset + p] = -max_ar_persistence
        ub[ar_offset : ar_offset + p] = max_ar_persistence
        theta = lsq_linear(X, y_vec, bounds=(lb, ub)).x

    residuals = np.zeros(n)
    fitted = np.full(n, np.nan)
    if T > 0:
        y_hat = X @ theta
        residuals[max_lag:] = y_vec - y_hat
        fitted[max_lag:] = y_hat

    return theta, param_names, residuals, fitted, fitted


def _unpack_theta(
    theta: np.ndarray,
    p: int,
    q: int,
    n_exog: int,
    has_deterministic: bool,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Split a flat parameter vector into its named blocks.

    The layout is ``[c?, ar_1..ar_p, ma_1..ma_q, beta_1..beta_n_exog]``, and
    is identical across every fitting helper in this module.

    Args:
        theta: Flat parameter vector.
        p: Autoregressive order.
        q: Moving-average order.
        n_exog: Number of exogenous regressors.
        has_deterministic: Whether ``theta`` opens with a constant (the
            ``const`` intercept when ``d=0``, or the ``drift`` term when
            ``d=1``).

    Returns:
        ``(c, ar, ma, beta)``, where ``c`` is 0.0 when there is no
        deterministic term and the arrays are empty at order zero.
    """
    idx = 1 if has_deterministic else 0
    c = float(theta[0]) if has_deterministic else 0.0
    ar = theta[idx : idx + p]
    idx += p
    ma = theta[idx : idx + q]
    idx += q
    beta = theta[idx : idx + n_exog] if n_exog > 0 else np.zeros(0)
    return c, ar, ma, beta


def _one_step_innovations(
    theta: np.ndarray,
    y: np.ndarray,
    x: np.ndarray | None,
    p: int,
    d: int,
    q: int,
    has_const: bool,
    has_drift: bool,
    n_exog: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the one-step mean-equation recursion over the whole series.

    For each observation, using the *actual* lagged values::

        eta[t] = y[t] - X[t] @ beta
        z[t]   = eta[t] (d=0) or diff(eta)[t] (d=1)
        eps[t] = z[t] - (c + sum(ar_j * z[t-j]) + sum(ma_j * eps[t-j]))

    Args:
        theta: Flat parameter vector (see :func:`_unpack_theta`).
        y: Endogenous series on the level scale.
        x: Exogenous regressor matrix, or ``None``.
        p: Autoregressive order.
        d: Differencing order, 0 or 1.
        q: Moving-average order.
        has_const: Whether ``theta`` opens with a level intercept.
        has_drift: Whether ``theta`` opens with a differenced-equation
            constant.
        n_exog: Number of exogenous regressors.

    Returns:
        ``(eta, z, eps)``. ``eps`` spans the differenced series with its
        leading ``max(p, q)`` entries left at zero, since the recursion
        defines no innovation there.
    """
    c, ar, ma, beta = _unpack_theta(theta, p, q, n_exog, has_const or has_drift)
    eta = y - x @ beta if (n_exog > 0 and x is not None) else y

    z = eta if d == 0 else np.diff(eta)
    n_z = len(z)
    max_lag = max(p, q)
    eps = np.zeros(n_z)

    for t in range(max_lag, n_z):
        ar_part = 0.0
        for j in range(p):
            ar_part += ar[j] * z[t - j - 1]

        ma_part = 0.0
        for j in range(q):
            ma_part += ma[j] * eps[t - j - 1]

        eps[t] = z[t] - (c + ar_part + ma_part)

    return eta, z, eps


def _arma_residuals(
    theta: np.ndarray,
    y: np.ndarray,
    x: np.ndarray | None,
    p: int,
    d: int,
    q: int,
    has_const: bool,
    has_drift: bool,
    n_exog: int,
) -> np.ndarray:
    """Equation-error residual for ``scipy.optimize.least_squares``.

    The one-step-ahead innovations of :func:`_one_step_innovations`, with the
    undefined leading lags dropped. This is the criterion a classical
    conditional-least-squares ARIMA fit minimizes.

    Args:
        theta: Flat parameter vector (see :func:`_unpack_theta`).
        y: Endogenous series on the level scale.
        x: Exogenous regressor matrix, or ``None``.
        p: Autoregressive order.
        d: Differencing order, 0 or 1.
        q: Moving-average order.
        has_const: Whether ``theta`` opens with a level intercept.
        has_drift: Whether ``theta`` opens with a differenced-equation
            constant.
        n_exog: Number of exogenous regressors.

    Returns:
        The innovations from lag ``max(p, q)`` onward.
    """
    _eta, _z, eps = _one_step_innovations(
        theta, y, x, p, d, q, has_const, has_drift, n_exog
    )
    return eps[max(p, q) :]


def _output_error_residuals(
    theta: np.ndarray,
    y: np.ndarray,
    x: np.ndarray | None,
    p: int,
    d: int,
    q: int,
    has_const: bool,
    has_drift: bool,
    n_exog: int,
    horizon: int,
) -> np.ndarray:
    """Output-error residual: windowed free-run error.

    The series is cut into consecutive windows of ``horizon`` steps. Each
    window is seeded from *actual* levels and *actual* one-step innovations,
    then simulated forward with its own predictions feeding the lags and its
    innovations held at zero -- exactly what the Pyomo surrogate does when it
    solves forward with ``eps`` fixed at zero, and exactly what
    :meth:`_DirectResults.predict` does with ``start`` and ``dynamic=True``.

    Unlike :func:`_arma_residuals`, an error in a parameter that accumulates
    over the horizon -- above all the ``d=1`` drift term -- is charged its
    full accumulated cost here rather than its per-step cost.

    A window length of 1 reproduces :func:`_arma_residuals` exactly, since a
    single step seeded from actual data *is* the one-step residual.

    Args:
        theta: Flat parameter vector (see :func:`_unpack_theta`).
        y: Endogenous series on the level scale.
        x: Exogenous regressor matrix, or ``None``.
        p: Autoregressive order.
        d: Differencing order, 0 or 1.
        q: Moving-average order.
        has_const: Whether ``theta`` opens with a level intercept.
        has_drift: Whether ``theta`` opens with a differenced-equation
            constant.
        n_exog: Number of exogenous regressors.
        horizon: Window length in steps, at least 1.

    Returns:
        Level-scale residuals for every position from ``max(p + d, q)``
        onward. The exogenous contribution cancels, so each residual is
        ``eta[t]`` less its simulated value.
    """
    c, ar, ma, beta = _unpack_theta(theta, p, q, n_exog, has_const or has_drift)
    eta, _z, eps = _one_step_innovations(
        theta, y, x, p, d, q, has_const, has_drift, n_exog
    )
    n = len(eta)
    seed = max(p + d, q)
    residuals: list[float] = []

    start = seed
    while start < n:
        window = min(horizon, n - start)
        eta_history = list(eta[start - (p + d) : start]) if p + d > 0 else []
        if q > 0:
            # eps is indexed on the differenced series, so for d=1 the lag
            # behind position `start` sits one place earlier than for d=0.
            stop = start - d
            lags = list(eps[max(stop - q, 0) : max(stop, 0)])
            eps_history = [0.0] * (q - len(lags)) + lags
        else:
            eps_history = []

        for step in range(window):
            if d == 0:
                mean = c + sum(ar[j] * eta_history[-(j + 1)] for j in range(p))
            else:
                mean = (
                    eta_history[-1]
                    + c
                    + sum(
                        ar[j] * (eta_history[-(j + 1)] - eta_history[-(j + 2)])
                        for j in range(p)
                    )
                )
            mean += sum(ma[j] * eps_history[-(j + 1)] for j in range(q))

            residuals.append(eta[start + step] - mean)
            eta_history.append(mean)
            if q > 0:
                eps_history.append(0.0)

        start += window

    return np.asarray(residuals, dtype=float)


def _one_step_report(
    theta: np.ndarray,
    y: np.ndarray,
    x: np.ndarray | None,
    p: int,
    d: int,
    q: int,
    has_const: bool,
    has_drift: bool,
    n_exog: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the one-step residuals and fitted values at ``theta``.

    Reported statistics stay on the one-step basis whichever objective was
    minimized, so that AIC keeps its usual meaning and stays comparable
    across fitting modes.

    Args:
        theta: Flat parameter vector (see :func:`_unpack_theta`).
        y: Endogenous series on the level scale.
        x: Exogenous regressor matrix, or ``None``.
        p: Autoregressive order.
        d: Differencing order, 0 or 1.
        q: Moving-average order.
        has_const: Whether ``theta`` opens with a level intercept.
        has_drift: Whether ``theta`` opens with a differenced-equation
            constant.
        n_exog: Number of exogenous regressors.

    Returns:
        ``(residuals, fitted_z, fitted_level)``, with ``residuals`` and
        ``fitted_z`` on the differenced scale and ``fitted_level``
        integrated back to the scale of ``y``. Positions the mean equation
        does not define are zero in ``residuals`` and ``NaN`` in the fitted
        arrays.
    """
    _c, _ar, _ma, beta = _unpack_theta(theta, p, q, n_exog, has_const or has_drift)
    eta, z, eps = _one_step_innovations(
        theta, y, x, p, d, q, has_const, has_drift, n_exog
    )
    n = len(y)
    n_z = len(z)
    max_lag = max(p, q)

    fitted_z = np.full(n_z, np.nan)
    fitted_z[max_lag:] = z[max_lag:] - eps[max_lag:]
    residuals = np.zeros(n_z)
    residuals[max_lag:] = eps[max_lag:]

    exog_full = x @ beta if (n_exog > 0 and x is not None) else np.zeros(n)
    if d == 0:
        fitted_level = exog_full + fitted_z
    else:
        fitted_level = np.full(n, np.nan)
        fitted_level[0] = y[0]
        for t in range(1, n):
            if not np.isnan(fitted_z[t - 1]):
                fitted_level[t] = exog_full[t] + eta[t - 1] + fitted_z[t - 1]

    return residuals, fitted_z, fitted_level


def _least_squares_bounds(
    theta0: np.ndarray,
    p: int,
    has_deterministic: bool,
    max_ar_persistence: float | None,
) -> tuple[np.ndarray, tuple, str]:
    """Build the AR-bounded least-squares setup shared by both objectives.

    Args:
        theta0: Initial parameter vector.
        p: Autoregressive order.
        has_deterministic: Whether ``theta0`` opens with a constant.
        max_ar_persistence: Bound applied to every AR coefficient, or
            ``None`` to fit unconstrained.

    Returns:
        ``(theta0, bounds, method)`` where ``theta0`` has its AR block
        clipped into the bounds and ``method`` is ``"lm"`` when unbounded or
        ``"trf"`` when bounded.
    """
    if max_ar_persistence is None:
        return theta0, (-np.inf, np.inf), "lm"

    ar_offset = 1 if has_deterministic else 0
    lower = np.full(theta0.shape, -np.inf)
    upper = np.full(theta0.shape, np.inf)
    lower[ar_offset : ar_offset + p] = -max_ar_persistence
    upper[ar_offset : ar_offset + p] = max_ar_persistence
    theta0 = theta0.copy()
    theta0[ar_offset : ar_offset + p] = np.clip(
        theta0[ar_offset : ar_offset + p],
        lower[ar_offset : ar_offset + p],
        upper[ar_offset : ar_offset + p],
    )
    return theta0, (lower, upper), "trf"


def _fit_output_error(
    theta0: np.ndarray,
    y: np.ndarray,
    x_values: np.ndarray | None,
    p: int,
    d: int,
    q: int,
    has_const: bool,
    has_drift: bool,
    n_exog: int,
    horizon: int,
    max_ar_persistence: float | None,
) -> np.ndarray:
    """Refine ``theta0`` by minimizing :func:`_output_error_residuals`.

    Warm-started from the equation-error solution, which is both a good
    starting point and far cheaper to obtain than a cold solve of the
    nonconvex free-run objective.

    Args:
        theta0: Equation-error solution to start from.
        y: Endogenous series on the level scale.
        x_values: Exogenous regressor matrix, or ``None``.
        p: Autoregressive order.
        d: Differencing order, 0 or 1.
        q: Moving-average order.
        has_const: Whether ``theta0`` opens with a level intercept.
        has_drift: Whether ``theta0`` opens with a differenced-equation
            constant.
        n_exog: Number of exogenous regressors.
        horizon: Free-run window length in steps.
        max_ar_persistence: Bound applied to every AR coefficient, or
            ``None``.

    Returns:
        The refined parameter vector, or ``theta0`` unchanged when there are
        no free parameters to refine.
    """
    from scipy.optimize import least_squares

    if theta0.size == 0:
        return theta0

    theta0, bounds, method = _least_squares_bounds(
        theta0, p, has_const or has_drift, max_ar_persistence
    )
    result = least_squares(
        _output_error_residuals,
        theta0,
        args=(y, x_values, p, d, q, has_const, has_drift, n_exog, horizon),
        method=method,
        bounds=bounds,
        verbose=0,
        max_nfev=5000,
        ftol=1e-8,
        xtol=1e-8,
    )
    return result.x


def _fit_arma_nls(
    y: np.ndarray,
    p: int,
    d: int,
    q: int,
    x_values: np.ndarray | None,
    exog_names: list[str],
    has_const: bool,
    has_drift: bool,
    max_ar_persistence: float | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Fit regression with ARIMA(p, d, q) errors by nonlinear least squares.

    Minimizes :func:`_arma_residuals` with ``scipy.optimize.least_squares``,
    starting from an OLS estimate of the exogenous coefficients and an OLS
    AR fit on the implied disturbance. Uses the ``lm`` method when
    unbounded and ``trf`` when ``max_ar_persistence`` bounds the AR block.

    Args:
        y: Endogenous series on the level scale (undifferenced).
        p: Autoregressive order.
        d: Differencing order, 0 or 1.
        q: Moving-average order.
        x_values: Exogenous regressor matrix, or ``None``.
        exog_names: Column names matching ``x_values``.
        has_const: Whether to fit a level intercept named ``const``.
        has_drift: Whether to fit a differenced-equation constant named
            ``drift``.
        max_ar_persistence: When set, bounds every AR coefficient to
            ``[-max_ar_persistence, max_ar_persistence]``.

    Returns:
        ``(theta, param_names, residuals, fitted_z, fitted_level)``, where
        ``fitted_z`` is on the differenced scale and ``fitted_level`` is
        integrated back to the scale of ``y``. ``residuals`` covers the
        differenced series with its first ``max(p, q)`` entries zero-padded.
    """
    from scipy.optimize import least_squares

    n = len(y)
    n_exog = x_values.shape[1] if x_values is not None else 0

    theta0 = np.zeros((1 if (has_const or has_drift) else 0) + p + q + n_exog)

    # Initial guess: estimate beta via OLS, then estimate AR on resulting disturbance
    if n_exog > 0 and x_values is not None:
        try:
            if has_const:
                X_mat = np.column_stack([np.ones(n), x_values])
                init_res, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
                if has_const:
                    theta0[0] = init_res[0]
                beta0 = init_res[1:]
            else:
                beta0, _, _, _ = np.linalg.lstsq(x_values, y, rcond=None)
            eta0 = y - x_values @ beta0
        except Exception:
            beta0 = np.zeros(n_exog)
            eta0 = y
    else:
        beta0 = np.zeros(0)
        eta0 = y

    z0 = eta0 if d == 0 else np.diff(eta0)
    n_z = len(z0)
    max_lag = p
    if max_lag > 0 and n_z > max_lag:
        cols = []
        if (has_const or has_drift) and n_exog == 0:
            cols.append(np.ones(n_z - max_lag))
        for j in range(1, p + 1):
            cols.append(z0[max_lag - j : n_z - j])
        if cols:
            try:
                ar_ols, _, _, _ = np.linalg.lstsq(
                    np.column_stack(cols), z0[max_lag:], rcond=None
                )
                ols_idx = 0
                if (has_const or has_drift) and n_exog == 0:
                    theta0[0] = ar_ols[0]
                    ols_idx = 1
                ar_offset = 1 if (has_const or has_drift) else 0
                theta0[ar_offset : ar_offset + p] = ar_ols[ols_idx : ols_idx + p]
            except Exception:
                pass

    idx_set = 1 if (has_const or has_drift) else 0
    idx_set += p + q
    if n_exog > 0:
        theta0[idx_set : idx_set + n_exog] = beta0

    theta0, bounds, method = _least_squares_bounds(
        theta0, p, has_const or has_drift, max_ar_persistence
    )

    result = least_squares(
        _arma_residuals,
        theta0,
        args=(y, x_values, p, d, q, has_const, has_drift, n_exog),
        method=method,
        bounds=bounds,
        verbose=0,
        max_nfev=5000,
        ftol=1e-8,
        xtol=1e-8,
    )
    theta_opt = result.x

    residuals, fitted_z, fitted_level = _one_step_report(
        theta_opt, y, x_values, p, d, q, has_const, has_drift, n_exog
    )

    param_names: list[str] = []
    if has_const:
        param_names.append("const")
    if has_drift:
        param_names.append("drift")
    for j in range(1, p + 1):
        param_names.append(f"ar{j}")
    for j in range(1, q + 1):
        param_names.append(f"ma{j}")
    param_names.extend(exog_names)

    return theta_opt, param_names, residuals, fitted_z, fitted_level


def _fit_output_error_ipopt(
    theta0: np.ndarray,
    y: np.ndarray,
    x_values: np.ndarray | None,
    *,
    order: tuple[int, int, int],
    has_const: bool,
    has_drift: bool,
    n_exog: int,
    exog_names: list[str],
    input_units: dict[str, str],
    output_name: str,
    output_units: str,
    training_index,
    max_ar_persistence: float | None,
) -> np.ndarray:
    """Minimize output error through the real Pyomo surrogate, with ipopt.

    Builds the training series as an
    :class:`~flexops.surrogates.arima.ArimaSurrogate` over a
    :class:`~flexops.core.time_block.TimeBlock`, frees the coefficients and
    the pre-horizon state, holds the innovations at zero so the block
    free-runs, and minimizes ``sum((y[t] - data[t])**2)``. The fitted
    coefficients are therefore optimal for the exact equation the surrogate
    will later solve, not for a Python transcription of it.

    Unlike the scipy backend this cannot window the free run -- one model
    spans the whole series. The pre-horizon state is seeded from data via
    the surrogate's ``history`` block, matching how the scipy backend seeds
    each window: leaving it free instead lets the solver fit coefficients
    that suit its own estimated state and transfer poorly, which measured
    0.087 free-run rmse against 0.005 for the seeded form. The MA block is
    bounded to keep the fit invertible; without that bound the solver trades
    the bounded AR block against MA and drives the MA roots out of the unit
    circle.

    Args:
        theta0: Equation-error solution, used as the warm start.
        y: Endogenous series on the level scale.
        x_values: Exogenous regressor matrix, or ``None``.
        order: ``(p, d, q)``.
        has_const: Whether ``theta0`` opens with a level intercept.
        has_drift: Whether ``theta0`` opens with a differenced-equation
            constant.
        n_exog: Number of exogenous regressors.
        exog_names: Exogenous column names, in fitted order.
        input_units: Units of every exogenous column, keyed by name.
        output_name: Name of the fitted output column.
        output_units: Units of the fitted output column.
        training_index: The fitted rows' index; must be a regular
            ``DatetimeIndex`` so a ``TimeBlock`` grid can be derived.
        max_ar_persistence: Bound applied to every AR coefficient, or
            ``None``.

    Returns:
        The refined parameter vector, in :func:`_unpack_theta` layout.

    Raises:
        FlexConfigError: If ``flexops`` is unavailable, the output or an
            input carries no units, or ``training_index`` is not a regular
            ``DatetimeIndex``.
    """
    from dateutil.relativedelta import relativedelta

    try:
        import pyomo.environ as pyo
        from pyomo.environ import units as pyunits

        from flexcore.solvers import ProblemClass, get_solver
        from flexops.core.time_block import TimeBlock
        from flexops.core.units import parse_units
        from flexops.surrogates.arima import ArimaSurrogate
    except ImportError as exc:  # pragma: no cover - flexops is a hard dep
        raise FlexConfigError(
            'ArimaRegressor fit_solver="ipopt" requires flexops and pyomo. '
            'Install the full package or use fit_solver="scipy".'
        ) from exc

    p, d, q = order
    if not output_units:
        raise FlexConfigError(
            'ArimaRegressor fit_solver="ipopt" builds a real Pyomo '
            "surrogate, which needs declared units. Pass output_units to "
            "fit().",
            field="output_units",
            value=output_units,
        )
    missing_units = [name for name in exog_names if not input_units.get(name)]
    if missing_units:
        raise FlexConfigError(
            'ArimaRegressor fit_solver="ipopt" needs units for every input; '
            f"{missing_units} have none. Pass input_units to fit().",
            field="input_units",
            value=missing_units,
        )
    if output_name in exog_names:
        raise FlexConfigError(
            f"ArimaRegressor cannot fit output {output_name!r} against an "
            "input of the same name.",
            field="output_variables",
            value=output_name,
        )

    if not isinstance(training_index, pd.DatetimeIndex) or len(training_index) < 2:
        raise FlexConfigError(
            'ArimaRegressor fit_solver="ipopt" needs a regular DatetimeIndex '
            "of at least two rows to build a TimeBlock grid; got "
            f"{type(training_index).__name__} of length "
            f'{len(training_index)}. Use fit_solver="scipy".',
            field="fit_solver",
            value="ipopt",
        )
    if getattr(training_index, "freq", None) is not None:
        step_seconds = float(pd.Timedelta(training_index.freq).total_seconds())
    else:
        step_seconds = float((training_index[1] - training_index[0]).total_seconds())
    if step_seconds <= 0:
        raise FlexConfigError(
            'ArimaRegressor fit_solver="ipopt" could not derive a positive '
            f"time step from the training index (got {step_seconds} s).",
            field="fit_solver",
            value="ipopt",
        )

    n = len(y)
    start = training_index[0].to_pydatetime()
    end = start + pd.Timedelta(seconds=step_seconds * n).to_pytimedelta()

    model = pyo.ConcreteModel()
    model.time_block = TimeBlock(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        time_step=step_seconds * pyunits.s,
        max_length=relativedelta(seconds=int(step_seconds * (n + 1))),
    )
    time_index = list(model.time_block.time_index)
    if len(time_index) != n:
        raise FlexConfigError(
            'ArimaRegressor fit_solver="ipopt" derived a TimeBlock of '
            f"{len(time_index)} steps for {n} training rows, so the index is "
            'not on a regular grid. Use fit_solver="scipy".',
            field="fit_solver",
            value="ipopt",
        )

    unit = pyo.Block(concrete=True)
    model.unit = unit
    unit.add_component(
        output_name,
        pyo.Var(time_index, initialize=0.0, units=parse_units(output_units)),
    )
    target = unit.find_component(output_name)
    for name in exog_names:
        unit.add_component(
            name,
            pyo.Var(time_index, initialize=0.0, units=parse_units(input_units[name])),
        )

    def _resolve_variable(self, name, field=None):
        """Stand in for OpsBlockData.resolve_variable for the surrogate."""
        component = self.find_component(name)
        if component is None:
            raise FlexConfigError(
                f"ARIMA input {name!r} is not on the fitting block.",
                field=field,
                value=name,
            )
        return component

    unit.resolve_variable = MethodType(_resolve_variable, unit)

    constant, ar, ma, beta = _unpack_theta(theta0, p, q, n_exog, has_const or has_drift)
    coefficients: dict[str, object] = {"order": [int(p), int(d), int(q)]}
    if has_const:
        coefficients["intercept"] = float(constant)
    elif has_drift:
        coefficients["drift"] = float(constant)
    if p > 0:
        coefficients["ar_coefs"] = [float(v) for v in ar]
    if q > 0:
        coefficients["ma_coefs"] = [float(v) for v in ma]
    if n_exog > 0:
        coefficients["exog_coefs"] = [float(v) for v in beta]

    eta_series, _z, eps = _one_step_innovations(
        theta0, y, x_values, p, d, q, has_const, has_drift, n_exog
    )
    surrogate = ArimaSurrogate(
        {
            "input_variables": {name: input_units[name] for name in exog_names},
            "output_variables": {output_name: output_units},
            "coefficients": coefficients,
            "history": _history_payload(
                eta_series,
                eps,
                p=p,
                d=d,
                q=q,
                start_date=training_index[0].isoformat(),
                time_step_seconds=step_seconds,
            ),
        },
        max_ar_coeff=max_ar_persistence,
    )
    block, body = surrogate.build(unit, target)
    unit.arima = block
    unit.fitted = pyo.Constraint(time_index, rule=lambda _b, t: target[t] == body(t))

    for position, t in enumerate(time_index):
        for index, name in enumerate(exog_names):
            unit.find_component(name)[t].fix(float(x_values[position, index]))
        target[t].set_value(float(y[position]))

    block.coefficients.unfix()
    if q > 0:
        for index in range(1, q + 1):
            block.ma_coefs[index].setlb(-_IPOPT_MA_BOUND)
            block.ma_coefs[index].setub(_IPOPT_MA_BOUND)

    model.objective = pyo.Objective(
        expr=sum(
            (target[t] - float(y[position])) ** 2
            for position, t in enumerate(time_index)
        )
    )
    results = get_solver(problem_class=ProblemClass.NLP, prefer="ipopt").solve(model)
    if not pyo.check_optimal_termination(results):
        _log.warning(
            'ArimaRegressor fit_solver="ipopt" terminated %s rather than '
            "optimally for order=%s. The returned coefficients are the "
            "solver's last iterate; check free_run_rmse before using them, "
            'or refit with fit_solver="scipy".',
            results.solver.termination_condition,
            order,
        )

    fitted = surrogate.get_surrogate_spec(block, target)["coefficients"]
    deterministic = fitted.get("intercept" if has_const else "drift", 0.0)
    return np.array(
        ([float(deterministic)] if (has_const or has_drift) else [])
        + [float(v) for v in fitted.get("ar_coefs", [])]
        + [float(v) for v in fitted.get("ma_coefs", [])]
        + [float(v) for v in fitted.get("exog_coefs", [])],
        dtype=float,
    )


def _fit_direct(
    y_values: np.ndarray,
    x_df: pd.DataFrame | None,
    *,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    include_mean: bool,
    include_drift: bool = False,
    max_ar_persistence: float | None = None,
    fit_objective: str = "equation_error",
    forecast_horizon: int = 1,
    refine_theta=None,
) -> _DirectResults:
    """Fit ARIMA directly via OLS/NLS, matching the Pyomo surrogate's objective.

    Dispatches to :func:`_fit_pure_regression`, :func:`_fit_ar_ols`, or
    :func:`_fit_arma_nls` depending on the order and whether exogenous
    regressors are present, then wraps the outcome in
    :class:`_DirectResults`.

    Args:
        y_values: Endogenous series on the level scale.
        x_df: Exogenous regressors, or ``None`` for a pure ARIMA.
        order: ``(p, d, q)``, with ``d`` in ``(0, 1)``.
        seasonal_order: ``(P, D, Q, m)``; must be trivial in ``P``, ``D``,
            and ``Q``.
        include_mean: Whether to fit the deterministic term (``const`` when
            ``d=0``, ``drift`` when ``d=1``).
        include_drift: Whether to force the ``d=1`` drift term on.
        max_ar_persistence: When set, bounds every AR coefficient to
            ``[-max_ar_persistence, max_ar_persistence]``.
        fit_objective: ``"equation_error"`` stops at the one-step fit;
            ``"output_error"`` refines it against
            :func:`_output_error_residuals`.
        refine_theta: Optional ``(theta, has_const, has_drift, n_exog) ->
            theta`` callback used in place of the built-in scipy refinement
            when ``fit_objective`` is ``"output_error"``. This is how the
            ipopt backend is injected without this function needing to know
            about Pyomo.
        forecast_horizon: Free-run window length in steps. Used by an
            ``"output_error"`` fit, and for the reported free-run error in
            either mode.

    Returns:
        The fitted :class:`_DirectResults`. Reported residuals and fitted
        values are on the one-step basis under both objectives.

    Raises:
        FlexConfigError: If any seasonal AR/MA/differencing term is nonzero.
    """
    p, d, q = order
    P, D, Q, m = seasonal_order

    if P > 0 or D > 0 or Q > 0:
        raise FlexConfigError(
            f"Seasonal ARIMA terms are not supported by the direct fit "
            f"backend. Got seasonal_order={seasonal_order}.",
            field="seasonal_order",
            value=seasonal_order,
        )

    exog_names = list(x_df.columns) if x_df is not None else []
    x_values = x_df.values if x_df is not None else None
    n_exog = x_values.shape[1] if x_values is not None else 0

    has_const = include_mean and d == 0
    has_drift = d == 1 and (include_mean or include_drift)
    k_params = (1 if has_const else 0) + (1 if has_drift else 0) + p + q + n_exog

    y_original = y_values.copy()
    diff_y = np.diff(y_values, n=d) if d > 0 else y_values

    if n_exog == 0:
        if p == 0 and q == 0:
            theta, param_names, residuals, fitted, _ = _fit_pure_regression(
                diff_y, x_values, exog_names, has_const, has_drift
            )
        elif q == 0:
            theta, param_names, residuals, fitted, _ = _fit_ar_ols(
                diff_y,
                p,
                x_values,
                exog_names,
                has_const,
                has_drift,
                max_ar_persistence,
            )
        else:
            theta, param_names, residuals, fitted, fitted_level = _fit_arma_nls(
                y_values,
                p,
                d,
                q,
                x_values,
                exog_names,
                has_const,
                has_drift,
                max_ar_persistence,
            )
        if p == 0 or q == 0:
            if d == 1:
                fitted_level = np.full(len(y_values), np.nan)
                fitted_level[0] = y_values[0]
                for t in range(1, len(y_values)):
                    if not np.isnan(fitted[t - 1]):
                        fitted_level[t] = y_values[t - 1] + fitted[t - 1]
            else:
                fitted_level = fitted
    else:
        theta, param_names, residuals, fitted, fitted_level = _fit_arma_nls(
            y_values,
            p,
            d,
            q,
            x_values,
            exog_names,
            has_const,
            has_drift,
            max_ar_persistence,
        )

    if fit_objective == "output_error" and theta.size > 0:
        if refine_theta is not None:
            theta = refine_theta(theta, has_const, has_drift, n_exog)
        else:
            theta = _fit_output_error(
                theta,
                y_values,
                x_values,
                p,
                d,
                q,
                has_const,
                has_drift,
                n_exog,
                forecast_horizon,
                max_ar_persistence,
            )
        residuals, fitted, fitted_level = _one_step_report(
            theta, y_values, x_values, p, d, q, has_const, has_drift, n_exog
        )

    if theta.size > 0:
        free_run_resid = _output_error_residuals(
            theta,
            y_values,
            x_values,
            p,
            d,
            q,
            has_const,
            has_drift,
            n_exog,
            forecast_horizon,
        )
    else:
        # No free parameters: _fit_pure_regression centers on the series
        # mean without exposing it as a coefficient, so there is no theta to
        # simulate from. The one-step residuals are the whole story.
        free_run_resid = residuals

    return _DirectResults(
        params=theta,
        residuals=residuals,
        fitted_values=fitted,
        y=diff_y,
        k_params=k_params,
        param_names=param_names,
        k_ar=p,
        k_ma=q,
        k_diff=d,
        seasonal_order=(P, D, Q, m),
        has_const=has_const,
        has_drift=has_drift,
        y_original=y_original,
        x=x_values,
        fittedvalues_level=fitted_level,
        free_run_resid=free_run_resid,
        forecast_horizon=forecast_horizon,
    )


def _aicc_from_direct(results: _DirectResults) -> float | None:
    """Compute AICc from a :class:`_DirectResults` object.

    The small-sample correction ``2k(k+1) / (n - k - 1)`` is undefined once
    the effective sample size ``n`` drops to ``k + 1`` or below, which a
    short series with a rich order can reach. Rather than raising, this
    logs a warning and returns ``None``; AICc is reported to the user and
    never consumed by the fit itself.

    Args:
        results: Fitted :class:`_DirectResults`.

    Returns:
        The corrected AIC value, or ``None`` when too few effective
        observations remain to define the correction term.
    """
    k = results.df_model + 1
    n = results.nobs_effective
    denominator = n - k - 1
    if denominator <= 0:
        _log.warning(
            "AICc is undefined for this fit: %d effective observation(s) "
            "against %d parameter(s) leaves a correction denominator of %d. "
            "Reporting AICc as None; use AIC or BIC, or fit a lower order "
            "on more data.",
            n,
            k,
            denominator,
        )
        return None
    return float(results.aic) + (2.0 * k * (k + 1)) / denominator


def _auto_select_order(
    y_values: np.ndarray,
    x_values: np.ndarray | None,
    *,
    seasonal: bool,
    stationary: bool,
    **auto_kwargs: object,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int] | None]:
    """Use statsforecast AutoARIMA to select the best order.

    Seasonal differencing ``D`` defaults to 0, since the backend cannot
    fit it.  Non-seasonal ``d`` is left to AutoARIMA and only validated
    afterwards, so pass ``max_d=1`` to keep the search inside what this
    backend supports.  The caller is responsible for refitting the returned
    order with the direct scipy backend for Pyomo-compatible parameters.

    Args:
        y_values: Endogenous time-series values.
        x_values: Exogenous regressor matrix, or ``None``.
        seasonal: Whether to include seasonal terms in the search.
        stationary: If ``True``, restrict the search to stationary models.
        **auto_kwargs: Extra keyword arguments forwarded to AutoARIMA
            (e.g. ``max_p``, ``max_q``, ``season_length``).

    Returns:
        ``(order, seasonal_order)`` where ``order`` is ``(p, d, q)`` with
        ``d`` in ``{0, 1}``, and ``seasonal_order`` is ``(P, 0, Q, m)`` or
        ``None``.

    Raises:
        FlexConfigError: If statsforecast is not installed.
    """
    try:
        from statsforecast.models import AutoARIMA
    except ImportError as exc:
        raise FlexConfigError(
            "ArimaRegressor auto=True requires statsforecast for order "
            "selection. Install it with `pip install 'flex-pse[parameterize]'`."
        ) from exc

    kwargs = dict(auto_kwargs)
    kwargs.setdefault("D", 0)
    if stationary:
        kwargs.setdefault("stationary", True)

    auto_model = AutoARIMA(seasonal=seasonal, ic="aic", **kwargs)
    fitted = auto_model.fit(y_values, X=x_values)

    arma = fitted.model_["arma"]
    p, q, P, Q, m_sf, d, D = arma
    d = int(d)
    if d not in (0, 1):
        raise FlexConfigError(
            f"ArimaRegressor auto=True selected d={d}, but the direct-fit "
            f"backend and Pyomo surrogate only support d=0 or d=1. "
            f"Restrict the search via auto_kwargs (e.g. max_d=1).",
            field="order",
            value=(int(p), d, int(q)),
        )
    order = (int(p), d, int(q))
    seasonal_order: tuple[int, int, int, int] | None
    if P == 0 and Q == 0:
        seasonal_order = None
    else:
        seasonal_order = (int(P), 0, int(Q), int(m_sf))
    return order, seasonal_order
