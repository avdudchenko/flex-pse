"""Tests for ArimaSurrogate: the Pyomo Constraint built from a fitted ARIMA.

``ArimaSurrogate`` is unit-tested in isolation: a bare ``OpsBlock`` with the
necessary IO variables is built, :meth:`ArimaSurrogate.build` is called, and
the resulting constraint expression is evaluated numerically to confirm it
reproduces the fitted ARIMA formula.

A key cross-validation test (``test_pyomo_matches_statsforecast``) evaluates
the Pyomo constraint against the statsforecast fitted values to ensure
identical output within tolerance.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlock
from flexops.core.time_block import TimeBlock
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.surrogates import ArimaSurrogate, surrogate_from_spec

_TEST_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "flexparameterize",
    "tests",
    "test_time_series_data",
)
_BIO_GAS_PATH = os.path.join(_TEST_DATA_DIR, "imputed_bio_gas_generation.csv")


def _make_unit(n_points: int = 96):
    """Return a bare OpsBlock with a TimeBlock, output, and two exog inputs."""
    m = pyo.ConcreteModel()
    # Create a TimeBlock with n_points time steps (15 min each)
    start = pd.Timestamp("2025-01-01")
    end = start + pd.Timedelta(minutes=15 * n_points)
    m.time_block = TimeBlock(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%dT%H:%M"),
        time_step=15 * pyunits.min,
    )
    m.props = SimpleAqueousFlow(has_pressure=False)
    m.unit = OpsBlock(property_package=m.props)
    m.unit.add_stream_ports()

    m.unit.add_component(
        "biogas_m3_hour",
        pyo.Var(
            m.time_block.time_index,
            initialize=0.0,
            units=pyunits.m**3 / pyunits.hr,
        ),
    )
    m.unit.register_io_variable(m.unit.biogas_m3_hour, role="output")

    for name, units in [
        ("feed_volume_kg", pyunits.kg),
        ("TS_pct", pyunits.dimensionless),
    ]:
        m.unit.add_component(
            name,
            pyo.Var(m.time_block.time_index, initialize=0.0, units=units),
        )
        m.unit.register_io_variable(getattr(m.unit, name), role="input")

    return m, m.unit


def _minimal_arima_data(p=1, q=0, n_exog=2, n_resid=96, include_residuals=True):
    """Return a minimal ARIMA SurrogateSpec.data dict."""
    exog_names = ["feed_volume_kg", "TS_pct"][:n_exog]
    input_vars = {
        name: ("kg" if name == "feed_volume_kg" else "dimensionless")
        for name in exog_names
    }
    data = {
        "input_variables": input_vars,
        "output_variables": {"biogas_m3_hour": "m^3/hr"},
        "exogenous_variables": exog_names,
        "order": [p, 0, q],
        "const": 0.01,
        "ar_coefs": [0.5] * p,
        "ma_coefs": [0.2] * q,
        "exog_coefs": [2.0] * n_exog,
        "init_values": [0.0] * p,
        "training_start_date": "2025-01-01T00:00:00",
        "training_time_step_seconds": 900.0,
        "training_y_values": [0.0] * n_resid,
    }
    if include_residuals:
        data["_residuals"] = [0.0] * n_resid
    return data


# -- _validate ---------------------------------------------------------------


@pytest.mark.unit
def test_validate_accepts_minimal_data():
    data = _minimal_arima_data(p=1, q=1, n_exog=2, n_resid=10)
    surrogate = ArimaSurrogate(data)
    assert surrogate.surrogate_type == SurrogateType.ARIMA


@pytest.mark.unit
def test_validate_rejects_unknown_key():
    data = _minimal_arima_data()
    data["unknown_key"] = 42
    with pytest.raises(FlexConfigError, match="unknown_key"):
        ArimaSurrogate(data)


@pytest.mark.unit
def test_validate_rejects_missing_key():
    data = _minimal_arima_data()
    del data["const"]
    with pytest.raises(FlexConfigError, match="const"):
        ArimaSurrogate(data)


@pytest.mark.unit
def test_validate_rejects_wrong_ar_coefs_length():
    data = _minimal_arima_data(p=2)
    data["ar_coefs"] = [0.5]  # should be 2
    with pytest.raises(FlexConfigError, match="ar_coefs"):
        ArimaSurrogate(data)


@pytest.mark.unit
def test_validate_rejects_exog_not_in_inputs():
    data = _minimal_arima_data()
    data["exogenous_variables"] = ["nonexistent"]
    with pytest.raises(FlexConfigError, match="nonexistent"):
        ArimaSurrogate(data)


@pytest.mark.unit
def test_validate_accepts_seasonal_order():
    data = _minimal_arima_data(p=1, q=1)
    data["seasonal_order"] = [1, 0, 0, 24]
    surrogate = ArimaSurrogate(data)
    assert surrogate.data["seasonal_order"] == [1, 0, 0, 24]


@pytest.mark.unit
def test_validate_rejects_non_list_residuals():
    data = _minimal_arima_data()
    data["_residuals"] = "not_a_list"
    with pytest.raises(FlexConfigError, match="_residuals"):
        ArimaSurrogate(data)


# -- input_variables / output_variables --------------------------------------


@pytest.mark.unit
def test_input_variables():
    data = _minimal_arima_data(n_exog=2)
    assert ArimaSurrogate(data).input_variables == {
        "feed_volume_kg": "kg",
        "TS_pct": "dimensionless",
    }


@pytest.mark.unit
def test_output_variables():
    data = _minimal_arima_data()
    assert ArimaSurrogate(data).output_variables == {"biogas_m3_hour": "m^3/hr"}


# -- build() ----------------------------------------------------------------


def _build_and_get_body(p=1, q=0, n_exog=2, n_resid=96):
    m, unit = _make_unit(n_points=n_resid)
    data = _minimal_arima_data(p=p, q=q, n_exog=n_exog, n_resid=n_resid)
    surrogate = ArimaSurrogate(data)
    body = surrogate.build(unit, unit.biogas_m3_hour)
    return m, unit, body, surrogate


@pytest.mark.unit
def test_build_creates_constraint_and_params():
    m, unit, body, surrogate = _build_and_get_body(p=1, q=0, n_exog=2, n_resid=10)
    assert unit.find_component("biogas_m3_hour_arima_eq") is not None
    assert unit.find_component("biogas_m3_hour_arima_const") is not None
    assert unit.find_component("biogas_m3_hour_arima_ar0") is not None
    assert unit.find_component("biogas_m3_hour_arima_exog0") is not None
    assert unit.find_component("biogas_m3_hour_arima_resid0") is not None
    assert unit.find_component("biogas_m3_hour_arima_y00") is not None


@pytest.mark.unit
def test_build_ar1_no_exog_matches_formula():
    m, unit = _make_unit(n_points=5)
    data = _minimal_arima_data(p=1, q=0, n_exog=0, n_resid=5)
    data["const"] = 0.01
    data["ar_coefs"] = [0.5]
    data["init_values"] = [1.0]
    data["_residuals"] = [0.1, -0.2, 0.3, -0.1, 0.0]
    surrogate = ArimaSurrogate(data)
    body = surrogate.build(unit, unit.biogas_m3_hour)

    unit.biogas_m3_hour[0].set_value(1.0)
    unit.biogas_m3_hour[1].set_value(2.0)
    unit.biogas_m3_hour[2].set_value(3.0)

    # t=1: y[1] == 0.01 + 0.5*y[0]  (no MA terms when q=0)
    expr = body(1)
    val = pyo.value(expr)
    expected = 0.01 + 0.5 * 1.0
    assert val == pytest.approx(expected)


@pytest.mark.unit
def test_build_with_exogenous_variables():
    m, unit = _make_unit(n_points=5)
    data = _minimal_arima_data(p=1, q=0, n_exog=2, n_resid=5)
    data["const"] = 0.0
    data["ar_coefs"] = [0.3]
    data["exog_coefs"] = [2.0, 0.5]
    data["init_values"] = [0.0]
    data["_residuals"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    surrogate = ArimaSurrogate(data)
    body = surrogate.build(unit, unit.biogas_m3_hour)

    unit.biogas_m3_hour[0].set_value(1.0)
    unit.feed_volume_kg[1].set_value(3.0)
    unit.TS_pct[1].set_value(10.0)

    expr = body(1)
    val = pyo.value(expr)
    expected = 0.0 + 0.3 * 1.0 + 2.0 * 3.0 + 0.5 * 10.0
    assert val == pytest.approx(expected)


@pytest.mark.unit
def test_build_ma_component():
    m, unit = _make_unit(n_points=5)
    data = _minimal_arima_data(p=0, q=1, n_exog=0, n_resid=5)
    data["const"] = 0.0
    data["ma_coefs"] = [0.4]
    data["init_values"] = []
    data["_residuals"] = [0.1, -0.2, 0.3, -0.1, 0.0]
    surrogate = ArimaSurrogate(data)
    body = surrogate.build(unit, unit.biogas_m3_hour)

    expr = body(1)
    val = pyo.value(expr)
    expected = 0.0 + 0.4 * 0.1  # theta_1 * resid[0]
    assert val == pytest.approx(expected)


@pytest.mark.unit
def test_build_ma_residual_negative_index_is_zero():
    m, unit = _make_unit(n_points=5)
    data = _minimal_arima_data(p=0, q=2, n_exog=0, n_resid=5)
    data["const"] = 0.0
    data["ma_coefs"] = [0.4, 0.3]
    data["init_values"] = []
    data["_residuals"] = [0.1, -0.2, 0.3, -0.1, 0.0]
    surrogate = ArimaSurrogate(data)
    body = surrogate.build(unit, unit.biogas_m3_hour)

    # t=0: MA terms use resid[-1] and resid[-2] -> both 0
    expr = body(0)
    val = pyo.value(expr)
    assert val == pytest.approx(0.0)


@pytest.mark.unit
def test_build_init_values_used_for_ar_lag():
    m, unit = _make_unit(n_points=5)
    data = _minimal_arima_data(p=2, q=0, n_exog=0, n_resid=5)
    data["const"] = 0.0
    data["ar_coefs"] = [0.5, 0.3]
    data["init_values"] = [1.0, 2.0]
    data["_residuals"] = [0.0] * 5
    surrogate = ArimaSurrogate(data)
    body = surrogate.build(unit, unit.biogas_m3_hour)

    # t=1: y[1] == 0.5*y[0] + 0.3*y[-1] = 0.5*0 + 0.3*2.0 = 0.6
    # (y[0] defaults to 0.0 since not fixed)
    expr = body(1)
    val = pyo.value(expr)
    assert val == pytest.approx(0.3 * 2.0)


# -- surrogate_from_spec integration -----------------------------------------


@pytest.mark.unit
def test_surrogate_from_spec():
    spec = SurrogateSpec(
        surrogate_type=SurrogateType.ARIMA,
        data={
            "input_variables": {"feed_volume_kg": "kg"},
            "output_variables": {"biogas_m3_hour": "m^3/hr"},
            "exogenous_variables": ["feed_volume_kg"],
            "order": [1, 0, 0],
            "const": 0.01,
            "ar_coefs": [0.5],
            "ma_coefs": [],
            "exog_coefs": [2.0],
            "_residuals": [0.0, 0.1, -0.2],
            "training_start_date": "2025-01-01T00:00:00",
            "training_time_step_seconds": 900.0,
            "training_y_values": [0.0] * 3,
        },
    )
    surrogate = surrogate_from_spec(spec)
    assert isinstance(surrogate, ArimaSurrogate)
    assert surrogate.input_variables == {"feed_volume_kg": "kg"}
    assert surrogate.output_variables == {"biogas_m3_hour": "m^3/hr"}


@pytest.mark.unit
def test_surrogate_from_spec_rejects_malformed():
    spec = SurrogateSpec(
        surrogate_type=SurrogateType.ARIMA,
        data={
            "input_variables": {"feed_volume_kg": "kg"},
            "output_variables": {"biogas_m3_hour": "m^3/hr"},
            "exogenous_variables": ["feed_volume_kg"],
            "order": [1, 0, 0],
            "const": 0.0,
            "ar_coefs": [0.5],
            "ma_coefs": [],
            "exog_coefs": [2.0],
            "_residuals": "not_a_list",
            "training_start_date": "2025-01-01T00:00:00",
            "training_time_step_seconds": 900.0,
            "training_y_values": [0.0] * 3,
        },
    )
    with pytest.raises(FlexConfigError, match="_residuals"):
        surrogate_from_spec(spec)


# -- cross-validation: Pyomo vs statsforecast --------------------------------


@pytest.mark.unit
def test_pyomo_matches_arima_formula():
    """Pyomo ArimaSurrogate reproduces the manual ARIMAX equation.

    Fits a small ARIMAX(1,0,1) with one exogenous regressor using
    statsforecast, builds the ArimaSurrogate in Pyomo, and evaluates
    the Pyomo constraint at each time step using the observed y values
    as the y[t-j] inputs.  The Pyomo result should match the manual
    AR/MA + exog formula within tolerance.
    """
    pytest.importorskip("statsforecast")

    rng = np.random.default_rng(42)
    n = 60
    idx = pd.date_range("2025-01-01", periods=n, freq="15min")
    feed = pd.Series(rng.uniform(0.1, 1.0, size=n), index=idx, name="feed")
    biogas = np.zeros(n)
    for t in range(1, n):
        biogas[t] = 0.5 * biogas[t - 1] + 2.0 * feed.iloc[t] + rng.normal(0, 0.05)

    from statsforecast.models import ARIMA

    model = ARIMA(order=(1, 0, 1), season_length=1, seasonal_order=(0, 0, 0))
    model = model.fit(biogas, X=feed.values.reshape(-1, 1))

    coef = model.model_["coef"]
    const = float(coef.get("intercept", 0.0))
    ar1 = float(coef["ar1"])
    ma1 = float(coef["ma1"])
    ex1 = float(coef["ex_1"])
    residuals = model.model_["residuals"]

    m, unit = _make_unit(n_points=n)
    data = {
        "input_variables": {"feed_volume_kg": "kg"},
        "output_variables": {"biogas_m3_hour": "m^3/hr"},
        "exogenous_variables": ["feed_volume_kg"],
        "order": [1, 0, 1],
        "const": const,
        "ar_coefs": [ar1],
        "ma_coefs": [ma1],
        "exog_coefs": [ex1],
        "init_values": [float(biogas[0])],
        "_residuals": residuals.tolist(),
        "training_start_date": "2025-01-01T00:00:00",
        "training_time_step_seconds": 900.0,
        "training_y_values": biogas.tolist(),
    }
    surrogate = ArimaSurrogate(data)
    body = surrogate.build(unit, unit.biogas_m3_hour)

    for t in range(n):
        unit.biogas_m3_hour[t].set_value(float(biogas[t]))
        unit.biogas_m3_hour[t].fix()
        unit.feed_volume_kg[t].set_value(float(feed.iloc[t]))
        unit.feed_volume_kg[t].fix()

    for t in range(1, n):
        pyomo_val = pyo.value(body(t))
        expected = (
            const + ar1 * biogas[t - 1] + ma1 * residuals[t - 1] + ex1 * feed.iloc[t]
        )
        assert pyomo_val == pytest.approx(expected, rel=1e-3), (
            f"t={t}: Pyomo={pyomo_val:.6f}, expected={expected:.6f}, "
            f"diff={abs(pyomo_val - expected):.6e}"
        )


@pytest.mark.unit
def test_pyomo_ar1_matches_manual_formula():
    """Pyomo AR(1) reproduces the manual AR(1) formula.

    Fits a small AR(1) model using statsforecast, builds the
    ArimaSurrogate in Pyomo, and evaluates the Pyomo constraint at each
    time step using the observed y values as the y[t-j] inputs.  The
    Pyomo result should match the manual AR formula within tolerance.
    """
    pytest.importorskip("statsforecast")

    rng = np.random.default_rng(0)
    n = 12
    phi = 0.7
    y_vals = np.zeros(n)
    for t in range(1, n):
        y_vals[t] = phi * y_vals[t - 1] + rng.normal(0, 0.1)

    from statsforecast.models import ARIMA

    model = ARIMA(order=(1, 0, 0), season_length=1, seasonal_order=(0, 0, 0))
    model = model.fit(y_vals)

    coef = model.model_["coef"]
    const = float(coef.get("intercept", 0.0))
    ar1 = float(coef["ar1"])

    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date="2025-01-01", end_date="2025-01-01T12:00", time_step=1 * pyunits.hr
    )
    m.props = SimpleAqueousFlow(has_pressure=False)
    m.unit = OpsBlock(property_package=m.props)
    m.unit.add_stream_ports()
    m.unit.add_component(
        "y",
        pyo.Var(
            m.time_block.time_index, initialize=0.0, units=pyunits.m**3 / pyunits.hr
        ),
    )
    m.unit.register_io_variable(m.unit.y, role="output")

    data = {
        "input_variables": {},
        "output_variables": {"y": "m^3/hr"},
        "exogenous_variables": [],
        "order": [1, 0, 0],
        "const": const,
        "ar_coefs": [ar1],
        "ma_coefs": [],
        "exog_coefs": [],
        "init_values": [float(y_vals[0])],
        "_residuals": [0.0] * n,
        "training_start_date": "2025-01-01T00:00:00",
        "training_time_step_seconds": 3600.0,
        "training_y_values": y_vals.tolist(),
    }
    surrogate = ArimaSurrogate(data)
    body = surrogate.build(m.unit, m.unit.y)

    for t in range(n):
        m.unit.y[t].set_value(float(y_vals[t]))
        m.unit.y[t].fix()

    for t in range(1, n):
        pyomo_val = pyo.value(body(t))
        expected = const + ar1 * y_vals[t - 1]
        assert pyomo_val == pytest.approx(expected, rel=1e-3)
