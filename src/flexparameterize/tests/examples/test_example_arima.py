"""Integration tests for ``src/flexparameterize/examples/example_arima.py``.

These tests exercise the full surrogate-behavior path that the example
demonstrates: load biogas data, fit an ``ArimaRegressor``, build an
``ArimaSurrogate`` inside a Pyomo ``OpsBlock`` via the documented
``register_relation`` / ``swap_relation`` seam, solve with ipopt, and confirm
that the Pyomo predictions match the direct-fit forecasts.

The tier is ``component`` because the tests build and solve a real Pyomo model;
they do not require network access.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.opt import assert_optimal_termination

from flexops.core.ops_block import OpsBlock
from flexops.core.time_block import TimeBlock
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.surrogates import ArimaSurrogate
from flexparameterize.regression.arima import ArimaRegressor

# ---------------------------------------------------------------------------
# Paths / data
# ---------------------------------------------------------------------------

_TEST_DIR = os.path.dirname(__file__)
_DATA_PATH = os.path.join(
    _TEST_DIR, "..", "test_time_series_data", "imputed_bio_gas_generation.csv"
)

_TARGET_COL = "biogas_m3_hour"
_EXOG_COLS = ["feed_volume_kg", "TS_pct"]

_HOURS_PER_DAY = 24
_N_VALID_DAYS = 2
_N_OPT_DAYS = 1
_N_VALID_HOURS = _N_VALID_DAYS * _HOURS_PER_DAY
_N_OPT_HOURS = _N_OPT_DAYS * _HOURS_PER_DAY
_N_INSAMPLE_HOURS = _HOURS_PER_DAY


def _load_bio_gas_dataframe() -> pd.DataFrame:
    """Load and resample the biogas test data to 1-hour mean."""
    df = (
        pd.read_csv(_DATA_PATH, parse_dates=["timestamp"])
        .set_index("timestamp")
        .sort_index()
    )
    return df.resample("1h").mean().dropna(subset=[_TARGET_COL] + _EXOG_COLS)


# ---------------------------------------------------------------------------
# Helper: build a Pyomo model with ArimaSurrogate (mirrors the example)
# ---------------------------------------------------------------------------


def _build_pyomo_model(start_idx, n_points, spec_data):
    """Return (model, unit, y_var, constraint_indexed).

    Attaches the ``ArimaSurrogate`` the documented way — a placeholder
    relation registered via ``register_relation`` and then replaced via
    ``swap_relation`` — rather than calling ``ArimaSurrogate.build()``
    directly.
    """
    m = pyo.ConcreteModel()
    end = start_idx + pd.Timedelta(hours=n_points)
    m.time_block = TimeBlock(
        start_date=start_idx.strftime("%Y-%m-%dT%H:%M"),
        end_date=end.strftime("%Y-%m-%dT%H:%M"),
        time_step=1 * pyunits.hr,
    )
    m.props = SimpleAqueousFlow(has_pressure=False)
    m.unit = OpsBlock(property_package=m.props)
    m.unit.add_stream_ports()

    m.unit.add_component(
        _TARGET_COL,
        pyo.Var(
            m.time_block.time_index,
            initialize=0.0,
            units=pyunits.m**3 / pyunits.hr,
        ),
    )
    m.unit.register_io_variable(m.unit.biogas_m3_hour, role="output")

    for col in _EXOG_COLS:
        m.unit.add_component(
            col,
            pyo.Var(
                m.time_block.time_index,
                initialize=0.0,
                units=pyunits.dimensionless,
            ),
        )
        m.unit.register_io_variable(getattr(m.unit, col), role="input")

    relation_name = f"{_TARGET_COL}_relation"
    m.unit.add_component(
        relation_name,
        pyo.Constraint(
            m.time_block.time_index,
            rule=lambda b, t: pyo.Constraint.Skip,
            doc="Placeholder relation, replaced by swap_relation below.",
        ),
    )
    m.unit.register_relation(
        getattr(m.unit, relation_name), target=m.unit.biogas_m3_hour
    )

    surrogate = ArimaSurrogate(spec_data)
    m.unit.swap_relation(relation_name, surrogate)

    con = m.unit.find_component(f"{relation_name}_fitted")
    return m, m.unit, m.unit.biogas_m3_hour, con


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_arima_example_end_to_end_arima_1_0_0():
    """ARIMA(1,0,0): fit, Pyomo surrogate matches direct fit, optimization works."""
    df_hour = _load_bio_gas_dataframe()

    train_df = df_hour.iloc[:-_N_VALID_HOURS]
    valid_df = df_hour.iloc[-_N_VALID_HOURS:]

    X_train = train_df[_EXOG_COLS]
    y_train = train_df[[_TARGET_COL]]
    X_valid = valid_df[_EXOG_COLS]
    y_valid = valid_df[[_TARGET_COL]]

    # Fit ARIMA(1,0,0)
    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
        X_train, y_train
    )
    assert regressor.fitted is True

    spec = regressor.to_surrogate_spec(
        input_units={col: "dimensionless" for col in _EXOG_COLS},
        output_units="m^3/hr",
    )

    n_insample = _N_INSAMPLE_HOURS
    n_fcst = _N_VALID_HOURS
    n_opt = _N_OPT_HOURS
    n_total = n_insample + n_fcst + n_opt

    train_end_idx = train_df.index[-1]
    forecast_exog = X_valid
    forecast_obs = y_valid[_TARGET_COL]

    insample_exog = train_df[_EXOG_COLS].iloc[-n_insample:].values
    insample_start = len(train_df) - n_insample
    all_exog = np.concatenate([insample_exog, forecast_exog.values])
    sm_all = regressor.model.predict(
        steps=n_insample + n_fcst,
        exog=all_exog,
        start=insample_start,
        dynamic=True,
    )
    sm_insample = sm_all[:n_insample]
    sm_fcst_mean = sm_all[n_insample:]

    target_biogas = float(forecast_obs.mean())

    exog_bounds = {}
    for col in _EXOG_COLS:
        exog_bounds[col] = (
            float(X_valid[col].min()),
            float(X_valid[col].max()),
        )

    m, unit, y_var, con = _build_pyomo_model(
        start_idx=train_end_idx - pd.Timedelta(hours=n_insample - 1),
        n_points=n_total,
        spec_data=spec.data,
    )

    for col in _EXOG_COLS:
        exog_var = getattr(unit, col)
        insample_vals = train_df[col].iloc[-n_insample:].values
        valid_vals = valid_df[col].values
        all_vals = np.concatenate([insample_vals, valid_vals])
        for t in range(n_insample + n_fcst):
            exog_var[t].set_value(float(all_vals[t]))
            exog_var[t].fix()

    for col in _EXOG_COLS:
        exog_var = getattr(unit, col)
        for t in range(n_insample + n_fcst, n_total):
            exog_var[t].unfix()
            exog_var[t].set_value(float(X_valid[col].mean()))

    for col in _EXOG_COLS:
        exog_var = getattr(unit, col)
        col_min, col_max = exog_bounds[col]
        for t in range(n_insample + n_fcst, n_total):
            exog_var[t].setlb(col_min)
            exog_var[t].setub(col_max)

    m.obj = pyo.Objective(
        expr=sum(
            (y_var[t] - target_biogas) ** 2 for t in range(n_insample + n_fcst, n_total)
        ),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory("ipopt")
    result = solver.solve(m, tee=False)
    assert_optimal_termination(result)

    time_set = list(y_var.index_set())
    pyomo_result = np.array([float(y_var[t].value) for t in time_set])
    pyomo_insample = pyomo_result[:n_insample]
    pyomo_fcst = pyomo_result[n_insample : n_insample + n_fcst]
    pyomo_opt = np.array(
        [float(y_var[t].value) for t in range(n_insample + n_fcst, n_total)]
    )

    insample_rmse = float(np.sqrt(np.mean((pyomo_insample - sm_insample) ** 2)))
    forecast_rmse = float(np.sqrt(np.mean((pyomo_fcst - sm_fcst_mean) ** 2)))

    # In-sample predictions should match the direct fit almost exactly
    assert insample_rmse < 1e-4, f"In-sample RMSE too high: {insample_rmse}"
    # Forecasts should also match (the surrogate implements the mean ARIMA equation)
    assert forecast_rmse < 1e-4, f"Forecast RMSE too high: {forecast_rmse}"

    # Optimization should drive the mean close to the target
    assert (
        abs(float(pyomo_opt.mean()) - target_biogas) < 0.5
    ), f"Optimized mean {pyomo_opt.mean():.4f} not close to target {target_biogas:.4f}"


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_arima_example_surrogate_spec_contract():
    """The spec emitted by the example's ArimaRegressor carries the full contract."""
    df_hour = _load_bio_gas_dataframe()

    train_df = df_hour.iloc[:-_N_VALID_HOURS]
    X_train = train_df[_EXOG_COLS]
    y_train = train_df[[_TARGET_COL]]

    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
        X_train, y_train
    )

    spec = regressor.to_surrogate_spec(
        input_units={col: "dimensionless" for col in _EXOG_COLS},
        output_units="m^3/hr",
    )

    data = spec.data
    assert data["order"] == [1, 0, 0]
    assert data["exogenous_variables"] == _EXOG_COLS
    assert len(data["exog_coefs"]) == 2
    assert isinstance(data["const"], float)
    assert isinstance(data["ar_coefs"], list)
    assert isinstance(data["ma_coefs"], list)
    assert isinstance(data["_residuals"], list)
    assert len(data["_residuals"]) == len(train_df)


@pytest.mark.component
@pytest.mark.needs_ipopt
def test_arima_example_build_pyomo_model_creates_fitted_constraint():
    """The example's _build_pyomo_model produces an active fitted constraint."""
    df_hour = _load_bio_gas_dataframe()
    train_df = df_hour.iloc[:-_N_VALID_HOURS]
    X_train = train_df[_EXOG_COLS]
    y_train = train_df[[_TARGET_COL]]

    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
        X_train, y_train
    )
    spec = regressor.to_surrogate_spec(
        input_units={col: "dimensionless" for col in _EXOG_COLS},
        output_units="m^3/hr",
    )

    train_end_idx = train_df.index[-1]
    m, unit, y_var, con = _build_pyomo_model(
        start_idx=train_end_idx - pd.Timedelta(hours=_N_INSAMPLE_HOURS - 1),
        n_points=_N_INSAMPLE_HOURS + _N_VALID_HOURS,
        spec_data=spec.data,
    )

    # The swap_relation call must have produced a fitted constraint
    assert con is not None
    assert con[0].active
