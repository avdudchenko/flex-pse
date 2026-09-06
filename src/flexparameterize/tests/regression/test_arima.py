"""Tests for ArimaRegressor: ARIMA time-series regression with exogenous inputs.

Importing this module (and ``ArimaRegressor`` itself) never requires
scipy or statsforecast -- only :meth:`ArimaRegressor.fit` does, lazily.
Tests that exercise a real fit call ``pytest.importorskip("scipy")``
themselves so the absence test below still runs (and passes) without the
extra installed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from flexcore.config.schema import SurrogateType
from flexcore.exceptions import FlexConfigError, FlexDataError
from flexparameterize.regression import Regressor
from flexparameterize.regression.arima import ArimaRegressor

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "test_time_series_data")
_BIO_GAS_PATH = os.path.join(_TEST_DATA_DIR, "imputed_bio_gas_generation.csv")


def _bio_gas_dataframe() -> pd.DataFrame:
    """Load the biogas test data set, parsed with a DatetimeIndex."""
    return pd.read_csv(_BIO_GAS_PATH, parse_dates=["timestamp"]).set_index("timestamp")


# -- absence tests -----------------------------------------------------------


@pytest.mark.unit
def test_scipy_absent_raises(monkeypatch):
    """With scipy unimportable, fitting raises FlexConfigError."""
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)

    y = pd.DataFrame({"biogas_m3_hour": [1.0, 2.0, 3.0]})
    with pytest.raises(FlexConfigError, match=r"flex-pse\[parameterize\]"):
        ArimaRegressor(order=(1, 0, 0)).fit(pd.DataFrame(index=y.index), y)


@pytest.mark.unit
def test_no_order_no_auto_raises():
    """Passing neither order nor auto=True raises FlexConfigError."""
    y = pd.DataFrame({"biogas_m3_hour": [1.0, 2.0, 3.0]})
    with pytest.raises(FlexConfigError, match="order"):
        ArimaRegressor().fit(pd.DataFrame(index=y.index), y)


# -- synthetic-data fitting tests --------------------------------------------


@pytest.mark.unit
def test_fits_ar1_no_exog():
    """AR(1) with no exogenous regressors recovers a known coefficient."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(0)
    n = 300
    phi = 0.7
    y_values = np.zeros(n)
    for t in range(1, n):
        y_values[t] = phi * y_values[t - 1] + rng.normal(0, 0.1)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    y = pd.DataFrame({"biogas": y_values}, index=idx)

    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
        pd.DataFrame(index=idx), y
    )
    assert regressor.coefficients is not None
    assert "ar1" in regressor.coefficients
    assert regressor.coefficients["ar1"] == pytest.approx(phi, rel=0.1)


@pytest.mark.unit
def test_fits_arima_with_exog():
    """ARIMA(1,1,1) with one exogenous regressor."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(7)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    feed = pd.Series(rng.uniform(0.1, 1.0, size=n), index=idx, name="feed")
    biogas = np.zeros(n)
    for t in range(1, n):
        biogas[t] = 0.5 * biogas[t - 1] + 2.0 * feed.iloc[t] + rng.normal(0, 0.05)
    y = pd.DataFrame({"biogas": biogas}, index=idx)

    regressor = ArimaRegressor(order=(1, 0, 1), max_ar_persistence=None).fit(
        pd.DataFrame({"feed": feed}), y
    )
    assert regressor.exogenous_variables == ["feed"]
    assert "feed" in regressor.coefficients
    assert regressor.coefficients["feed"] > 0.5


@pytest.mark.unit
def test_fits_multiple_exog_columns():
    """Two exogenous columns are both recovered."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(99)
    n = 150
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    x1 = pd.Series(rng.uniform(1.0, 10.0, size=n), index=idx, name="x1")
    x2 = pd.Series(rng.uniform(1e5, 5e5, size=n), index=idx, name="x2")
    y_vals = 0.4 * x1 + 1e-5 * x2 + rng.normal(0, 0.01, size=n)
    y = pd.DataFrame({"output": y_vals}, index=idx)

    regressor = ArimaRegressor(order=(0, 0, 0)).fit(
        pd.DataFrame({"x1": x1, "x2": x2}), y
    )
    assert sorted(regressor.exogenous_variables) == ["x1", "x2"]
    assert "x1" in regressor.coefficients
    assert "x2" in regressor.coefficients
    assert regressor.coefficients["x1"] == pytest.approx(0.4, rel=0.2)
    assert regressor.coefficients["x2"] == pytest.approx(1e-5, rel=0.2)


# -- protocol conformance ----------------------------------------------------


@pytest.mark.unit
def test_isinstance_regressor():
    """ArimaRegressor structurally conforms to Regressor and behaves as one."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(1)
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vals = np.cumsum(rng.normal(0, 0.1, size=n))
    y = pd.DataFrame({"y": vals}, index=idx)

    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
        pd.DataFrame(index=idx), y
    )
    assert isinstance(regressor, Regressor)
    result = regressor.to_fit_result()
    assert isinstance(result.coefficients, dict)
    assert result.n_samples == n
    assert np.isfinite(result.metrics["aic"])
    assert np.isfinite(result.metrics["rmse"])


@pytest.mark.unit
def test_provenance_populated():
    """Fit metrics are finite and the emitted spec's provenance is JSON-safe."""
    import json

    pytest.importorskip("scipy")

    rng = np.random.default_rng(2)
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vals = np.cumsum(rng.normal(0, 0.1, size=n))
    y = pd.DataFrame({"y": vals}, index=idx)

    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
        pd.DataFrame(index=idx), y
    )
    result = regressor.to_fit_result()
    assert np.isfinite(result.metrics["aic"])
    assert np.isfinite(result.metrics["rmse"])
    assert result.n_samples == n
    assert len(result.data_window) == 2

    spec = regressor.to_surrogate_spec(input_units={}, output_units="m^3/hr")
    assert spec.surrogate_type == SurrogateType.ARIMA
    provenance = {"n_samples": result.n_samples, **result.metrics}
    json.dumps(provenance)


# -- SurrogateSpec data contract ----------------------------------------------


@pytest.mark.unit
def test_surrogate_spec_data_contract():
    """SurrogateSpec.data contains all keys the ArimaSurrogate build needs."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(3)
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    feed = pd.Series(rng.uniform(0.1, 1.0, size=n), index=idx, name="feed")
    vals = 2.0 * feed + rng.normal(0, 0.05, size=n)
    y = pd.DataFrame({"biogas": vals}, index=idx)

    regressor = ArimaRegressor(order=(0, 0, 0)).fit(pd.DataFrame({"feed": feed}), y)
    spec = regressor.to_surrogate_spec(
        input_units={"feed": "kg"}, output_units="m^3/hr"
    )

    data = spec.data
    assert set(data) == {
        "input_variables",
        "output_variables",
        "exogenous_variables",
        "order",
        "seasonal_order",
        "const",
        "ar_coefs",
        "ma_coefs",
        "exog_coefs",
        "_residuals",
        "init_values",
        "training_start_date",
        "training_time_step_seconds",
        "training_y_values",
    }
    assert data["exogenous_variables"] == ["feed"]
    assert data["order"] == [0, 0, 0]
    assert data["output_variables"] == {"biogas": "m^3/hr"}
    assert isinstance(data["const"], float)
    assert isinstance(data["ar_coefs"], list)
    assert isinstance(data["ma_coefs"], list)
    assert isinstance(data["exog_coefs"], list)
    assert len(data["exog_coefs"]) == 1


@pytest.mark.unit
def test_surrogate_spec_seasonal_order_none_without_seasonal():
    """seasonal_order is ``None`` when no seasonal terms were fitted."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(4)
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vals = np.cumsum(rng.normal(0, 0.1, size=n))
    y = pd.DataFrame({"y": vals}, index=idx)

    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
        pd.DataFrame(index=idx), y
    )
    spec = regressor.to_surrogate_spec(input_units={}, output_units="unit")
    assert spec.data.get("seasonal_order") is None


# -- to_fit_result / to_surrogate_spec guards ---------------------------------


@pytest.mark.unit
def test_to_fit_result_before_fit_raises():
    """to_fit_result before fit raises FlexDataError."""
    regressor = ArimaRegressor(order=(1, 0, 0))
    with pytest.raises(FlexDataError, match="no fit yet"):
        regressor.to_fit_result()


@pytest.mark.unit
def test_to_surrogate_spec_before_fit_raises():
    """to_surrogate_spec before fit raises FlexDataError."""
    regressor = ArimaRegressor(order=(1, 0, 0))
    with pytest.raises(FlexDataError, match="no fit yet"):
        regressor.to_surrogate_spec(input_units={}, output_units="unit")


@pytest.mark.unit
def test_missing_input_units_raises():
    """Missing input_units for an exogenous column raises FlexConfigError."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(5)
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    feed = pd.Series(rng.uniform(0.1, 1.0, size=n), index=idx, name="feed")
    vals = 2.0 * feed.values + rng.normal(0, 0.05, size=n)
    y = pd.DataFrame({"biogas": vals}, index=idx)

    regressor = ArimaRegressor(order=(0, 0, 0)).fit(pd.DataFrame({"feed": feed}), y)
    with pytest.raises(FlexConfigError, match="feed"):
        regressor.to_surrogate_spec(
            input_units={"other_col": "kg"}, output_units="m^3/hr"
        )


# -- diagnostics -------------------------------------------------------------


@pytest.mark.unit
def test_fit_diagnostics_keys():
    """fit_diagnostics returns AIC, BIC, AICc, RMSE, log-likelihood."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(6)
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vals = np.cumsum(rng.normal(0, 0.1, size=n))
    y = pd.DataFrame({"y": vals}, index=idx)

    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
        pd.DataFrame(index=idx), y
    )
    diag = regressor.fit_diagnostics()
    assert np.isfinite(diag["aic"])
    assert np.isfinite(diag["bic"])
    assert np.isfinite(diag["aicc"])
    assert np.isfinite(diag["rmse"])
    assert np.isfinite(diag["log_likelihood"])


@pytest.mark.unit
def test_fit_diagnostics_before_fit_raises():
    """fit_diagnostics before fit raises FlexDataError."""
    regressor = ArimaRegressor(order=(1, 0, 0))
    with pytest.raises(FlexDataError, match="no fit yet"):
        regressor.fit_diagnostics()


# -- auto_arima --------------------------------------------------------------


@pytest.mark.unit
def test_auto_arima_selects_order():
    """auto=True runs AutoARIMA and stores a valid order."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(8)
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vals = np.cumsum(rng.normal(0, 0.1, size=n))
    y = pd.DataFrame({"y": vals}, index=idx)

    regressor = ArimaRegressor(
        auto=True, max_p=2, max_q=2, max_ar_persistence=None
    ).fit(pd.DataFrame(index=idx), y)
    assert regressor._order is not None
    assert len(regressor._order) == 3
    assert all(isinstance(v, int) for v in regressor._order)
    result = regressor.to_fit_result()
    assert np.isfinite(result.metrics["aic"])


# -- biogas test data --------------------------------------------------------


@pytest.mark.unit
def test_fits_biogas_with_feed_and_ts_exog():
    """Fits biogas_m3_hour from feed_volume_kg and TS_pct as exogenous."""
    pytest.importorskip("scipy")

    if not os.path.exists(_BIO_GAS_PATH):
        pytest.skip(f"Test data not found at {_BIO_GAS_PATH}")

    df = _bio_gas_dataframe()
    X = df[["feed_volume_kg", "TS_pct"]]
    y = df[["biogas_m3_hour"]]

    regressor = ArimaRegressor(order=(1, 0, 1), max_ar_persistence=None).fit(X, y)

    assert regressor.exogenous_variables == ["feed_volume_kg", "TS_pct"]
    assert regressor.output_variable == "biogas_m3_hour"
    assert regressor.n_samples == len(
        df.dropna(subset=["feed_volume_kg", "TS_pct", "biogas_m3_hour"])
    )
    assert np.isfinite(regressor.metrics["aic"])
    assert np.isfinite(regressor.metrics["rmse"])

    spec = regressor.to_surrogate_spec(
        input_units={"feed_volume_kg": "kg", "TS_pct": "%"},
        output_units="m^3/hr",
    )
    assert spec.surrogate_type == SurrogateType.ARIMA
    assert spec.data["exogenous_variables"] == ["feed_volume_kg", "TS_pct"]
    assert len(spec.data["exog_coefs"]) == 2
    assert spec.data["order"] == [1, 0, 1]
    assert spec.data["output_variables"] == {"biogas_m3_hour": "m^3/hr"}


@pytest.mark.unit
def test_biogas_surrogate_spec_has_all_keys():
    """Spec produced from biogas data carries the full ArimaSurrogate contract."""
    pytest.importorskip("scipy")

    if not os.path.exists(_BIO_GAS_PATH):
        pytest.skip(f"Test data not found at {_BIO_GAS_PATH}")

    df = _bio_gas_dataframe()
    X = df[["feed_volume_kg", "TS_pct"]]
    y = df[["biogas_m3_hour"]]

    regressor = ArimaRegressor(order=(2, 0, 1), max_ar_persistence=None).fit(X, y)
    spec = regressor.to_surrogate_spec(
        input_units={"feed_volume_kg": "kg", "TS_pct": "%"},
        output_units="m^3/hr",
    )

    data = spec.data
    assert "input_variables" in data
    assert "output_variables" in data
    assert "exogenous_variables" in data
    assert "order" in data
    assert "const" in data
    assert "ar_coefs" in data
    assert "ma_coefs" in data
    assert "exog_coefs" in data

    assert len(data["ar_coefs"]) == 2  # AR(2)
    assert len(data["ma_coefs"]) == 1  # MA(1)
    assert len(data["exog_coefs"]) == 2  # two exogenous columns


# -- validation: non-differenced only -----------------------------------------


@pytest.mark.unit
def test_d_greater_than_zero_raises():
    """ARIMA order with d>0 raises FlexConfigError."""
    with pytest.raises(FlexConfigError, match="d=0"):
        ArimaRegressor(order=(1, 1, 0))


@pytest.mark.unit
def test_D_greater_than_zero_raises():
    """Seasonal order with D>0 raises FlexConfigError."""
    with pytest.raises(FlexConfigError, match="D=0"):
        ArimaRegressor(order=(1, 0, 0), seasonal_order=(0, 1, 0, 24))


@pytest.mark.unit
def test_include_drift_raises():
    """include_drift=True raises FlexConfigError."""
    with pytest.raises(FlexConfigError, match="include_drift"):
        ArimaRegressor(order=(1, 0, 0), include_drift=True)


@pytest.mark.unit
def test_auto_arima_respects_d_zero():
    """auto=True forces d=0 and D=0 even if user passes different kwargs."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(42)
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vals = np.cumsum(rng.normal(0, 0.1, size=n))
    y = pd.DataFrame({"y": vals}, index=idx)

    # User requests max_d=1, but auto mode should override to d=0
    regressor = ArimaRegressor(
        auto=True, max_p=2, max_q=2, max_d=1, max_ar_persistence=None
    ).fit(pd.DataFrame(index=idx), y)
    assert regressor._order is not None
    assert regressor._order[1] == 0  # d must be 0


@pytest.mark.unit
def test_high_ar_persistence_raises():
    """AR coefficient exceeding max_ar_persistence raises FlexConfigError."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(99)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    vals = np.cumsum(rng.normal(0, 0.1, size=n))
    y = pd.DataFrame({"y": vals}, index=idx)

    # AR(1) with default max_ar_persistence=0.85 should raise
    with pytest.raises(FlexConfigError, match="max_ar_persistence"):
        ArimaRegressor(order=(1, 0, 0)).fit(pd.DataFrame(index=idx), y)


@pytest.mark.unit
def test_low_ar_persistence_passes():
    """AR coefficient below max_ar_persistence fits successfully."""
    pytest.importorskip("scipy")

    rng = np.random.default_rng(1)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    phi = 0.5
    vals = np.zeros(n)
    for t in range(1, n):
        vals[t] = phi * vals[t - 1] + rng.normal(0, 0.1)
    y = pd.DataFrame({"y": vals}, index=idx)

    # AR(1) with relaxed persistence threshold should fit
    regressor = ArimaRegressor(order=(1, 0, 0), max_ar_persistence=1.0).fit(
        pd.DataFrame(index=idx), y
    )
    assert regressor._fitted is True


# -- registry -----------------------------------------------------------------


@pytest.mark.unit
def test_arima_in_registry():
    """get_regressor('arima') now returns ArimaRegressor (not raises)."""
    from flexparameterize.regression import get_regressor

    assert get_regressor(SurrogateType.ARIMA) is ArimaRegressor
    assert get_regressor("arima") is ArimaRegressor
