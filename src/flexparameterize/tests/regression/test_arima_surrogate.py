"""Cross-validation tests for ArimaSurrogate against ArimaRegressor.

These tests require ``flexparameterize`` and validate that the Pyomo
surrogate reproduces the predictions of the fitted regressor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexops.core.ops_block import OpsBlock
from flexops.core.time_block import TimeBlock
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.surrogates import ArimaSurrogate
from flexparameterize.regression.arima import ArimaRegressor


@pytest.mark.unit
def test_pyomo_matches_direct_fit_for_multiple_arima_orders():
    """Pyomo surrogate matches ArimaRegressor predictions for multiple orders."""
    np.random.seed(42)
    n_train = 100
    n_insample = 20
    n_fcst = 10
    idx = pd.date_range("2024-01-01", periods=n_train, freq="1h")
    feed = pd.Series(np.random.uniform(0.1, 1.0, size=n_train), index=idx, name="feed")

    orders_to_test = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 1),
    ]

    for order in orders_to_test:
        p, d, q = order

        # Generate synthetic data matching the ARIMA structure
        y_values = np.zeros(n_train)
        params = {
            (1, 0, 0): {"phi": 0.5, "const": 0.1},
            (0, 1, 0): {"phi": 0.3, "const": 0.05},
            (0, 0, 1): {"theta": 0.4, "const": 0.2},
            (1, 1, 1): {"phi": 0.4, "theta": 0.3, "const": 0.08},
        }
        param = params.get(order, {})

        for t in range(1, n_train):
            if d == 0:
                if p == 1 and q == 0:
                    y_values[t] = (
                        param["const"]
                        + param["phi"] * y_values[t - 1]
                        + np.random.normal(0, 0.05)
                    )
                elif p == 0 and q == 1:
                    y_values[t] = param["const"] + np.random.normal(0, 0.05)
                elif p == 1 and q == 1:
                    y_values[t] = (
                        param["const"]
                        + param["phi"] * y_values[t - 1]
                        + param["theta"]
                        * (np.random.normal(0, 0.05) if t == 1 else 0.0)
                        + np.random.normal(0, 0.05)
                    )
            else:  # d == 1
                if p == 1 and q == 1:
                    y_values[t] = (
                        y_values[t - 1]
                        + param["const"]
                        + param["phi"] * (y_values[t - 1] - y_values[t - 2])
                        + np.random.normal(0, 0.05)
                    )

        y = pd.DataFrame({"biogas": y_values}, index=idx)

        # Fit model using our own ArimaRegressor
        regressor = ArimaRegressor(order=order, max_ar_persistence=None).fit(
            pd.DataFrame({"feed": feed}), y
        )
        assert regressor.fitted is True

        # Get predictions from the direct fit model for the entire horizon.
        # Use a single predict call with start to ensure consistent
        # recursive forecasting.
        all_exog = np.zeros((n_insample + n_fcst, 1))
        all_exog[:n_insample] = feed.iloc[-n_insample:].values.reshape(-1, 1)
        direct_all = np.asarray(
            regressor.model.predict(
                steps=n_insample + n_fcst,
                exog=all_exog,
                start=n_train - n_insample,
                dynamic=True,
            )
        )
        direct_insample = direct_all[:n_insample]
        direct_fcst = direct_all[n_insample:]

        # Build Pyomo model
        m = pyo.ConcreteModel()
        start_idx = idx[-n_insample]
        m.time_block = TimeBlock(
            start_date=start_idx.strftime("%Y-%m-%dT%H:%M"),
            end_date=(start_idx + pd.Timedelta(hours=n_insample + n_fcst)).strftime(
                "%Y-%m-%dT%H:%M"
            ),
            time_step=1 * pyunits.hr,
        )
        m.props = SimpleAqueousFlow(has_pressure=False)
        m.unit = OpsBlock(property_package=m.props)
        m.unit.add_stream_ports()
        m.unit.add_component(
            "biogas_m3_hour",
            pyo.Var(
                m.time_block.time_index, initialize=0.0, units=pyunits.m**3 / pyunits.hr
            ),
        )
        m.unit.register_io_variable(m.unit.biogas_m3_hour, role="output")
        m.unit.add_component(
            "feed",
            pyo.Var(
                m.time_block.time_index, initialize=0.0, units=pyunits.dimensionless
            ),
        )
        m.unit.register_io_variable(m.unit.feed, role="input")

        # Attach the surrogate the documented way: a placeholder relation
        # registered via register_relation, then replaced via swap_relation.
        # ArimaSurrogate.build() itself only attaches auxiliary Params; it is
        # swap_relation that builds the constraint enforcing the equation.
        m.unit.add_component(
            "biogas_m3_hour_relation",
            pyo.Constraint(
                m.time_block.time_index,
                rule=lambda b, t: pyo.Constraint.Skip,
                doc="Placeholder relation, replaced by swap_relation below.",
            ),
        )
        m.unit.register_relation(
            m.unit.biogas_m3_hour_relation, target=m.unit.biogas_m3_hour
        )

        spec = regressor.to_surrogate_spec(
            input_units={"feed": "dimensionless"},
            output_units="m^3/hr",
        )
        surrogate = ArimaSurrogate(spec.data)
        m.unit.swap_relation("biogas_m3_hour_relation", surrogate)

        # Fix exog: use actual values for in-sample, zeros for forecast
        for t in range(n_insample):
            m.unit.feed[t].set_value(float(feed.iloc[-n_insample + t]))
            m.unit.feed[t].fix()
        for t in range(n_fcst):
            m.unit.feed[n_insample + t].set_value(0.0)
            m.unit.feed[n_insample + t].fix()

        # Initialize target with direct fit predictions
        for t in range(n_insample + n_fcst):
            m.unit.biogas_m3_hour[t].set_value(float(direct_all[t]))

        # Add dummy objective (0DOF problem - all variables determined by constraints)
        m.obj = pyo.Objective(expr=0.0)

        # Solve with ipopt
        solver = pyo.SolverFactory("ipopt")
        result = solver.solve(m, tee=False)

        assert (
            result.solver.termination_condition == pyo.TerminationCondition.optimal
        ), f"ARIMA{order} solve failed: {result.solver.termination_condition}"

        # Extract solved values
        pyomo_insample = np.array(
            [float(m.unit.biogas_m3_hour[t].value) for t in range(n_insample)]
        )
        pyomo_fcst = np.array(
            [
                float(m.unit.biogas_m3_hour[t].value)
                for t in range(n_insample, n_insample + n_fcst)
            ]
        )

        # Compare Pyomo solved values to direct fit predictions
        insample_rmse = np.sqrt(np.mean((pyomo_insample - direct_insample) ** 2))
        fcst_rmse = np.sqrt(np.mean((pyomo_fcst - direct_fcst) ** 2))

        print(
            f"ARIMA{order}: in-sample RMSE={insample_rmse:.6e},"
            f" forecast RMSE={fcst_rmse:.6e}"
        )
        assert insample_rmse < 1e-4, f"ARIMA{order} in-sample mismatch: {insample_rmse}"
        assert fcst_rmse < 1e-4, f"ARIMA{order} forecast mismatch: {fcst_rmse}"


@pytest.mark.unit
def test_reswapping_arima_relation_succeeds_and_uses_latest_coefficients():
    """A second re-fit-and-reswap of an ArimaSurrogate-backed relation must:

    - not raise (H1: ArimaSurrogate no longer self-adds an enforcing
      Constraint that collides with swap_relation's own on a second build);
    - leave exactly one *active* equality constraint enforcing the relation
      (no duplicate from an internal ArimaSurrogate constraint);
    - not silently keep serving the *first* fit's coefficients (H2): the
      first fit's Params are untouched but unreferenced, and solving after
      the second swap reflects only the second fit's coefficients.
    """
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")

    def _fit(phi, const, seed):
        rng = np.random.default_rng(seed)
        y_values = np.zeros(n)
        for t in range(1, n):
            y_values[t] = const + phi * y_values[t - 1] + rng.normal(0, 0.01)
        y = pd.DataFrame({"y": y_values}, index=idx)
        return ArimaRegressor(order=(1, 0, 0), max_ar_persistence=None).fit(
            pd.DataFrame(index=idx), y
        )

    regressor1 = _fit(phi=0.3, const=0.1, seed=1)
    regressor2 = _fit(phi=0.6, const=0.4, seed=2)

    m = pyo.ConcreteModel()
    m.time_block = TimeBlock(
        start_date=idx[10].strftime("%Y-%m-%dT%H:%M"),
        end_date=(idx[10] + pd.Timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M"),
        time_step=1 * pyunits.hr,
    )
    m.props = SimpleAqueousFlow(has_pressure=False)
    m.unit = OpsBlock(property_package=m.props)
    m.unit.add_stream_ports()
    m.unit.add_component(
        "y",
        pyo.Var(
            m.time_block.time_index,
            initialize=0.0,
            units=pyunits.m**3 / pyunits.hr,
        ),
    )
    m.unit.register_io_variable(m.unit.y, role="output")
    m.unit.add_component(
        "y_relation",
        pyo.Constraint(m.time_block.time_index, rule=lambda b, t: pyo.Constraint.Skip),
    )
    m.unit.register_relation(m.unit.y_relation, target=m.unit.y)

    spec1 = regressor1.to_surrogate_spec(input_units={}, output_units="m^3/hr")
    m.unit.swap_relation("y_relation", ArimaSurrogate(spec1.data))

    fitted_1 = m.unit.find_component("y_relation_fitted")
    assert fitted_1 is not None
    assert fitted_1[0].active

    # Re-fit and reswap -- must not raise.
    spec2 = regressor2.to_surrogate_spec(input_units={}, output_units="m^3/hr")
    m.unit.swap_relation("y_relation", ArimaSurrogate(spec2.data))

    # The first swap's fitted constraint is deactivated, not deleted; the
    # second gets its own uniquely-named fitted constraint.
    assert not fitted_1[0].active
    fitted_2 = m.unit.find_component("y_relation_fitted_2")
    assert fitted_2 is not None
    assert fitted_2[0].active

    # Exactly one active equality constraint enforces the relation -- no
    # duplicate left over from an ArimaSurrogate-internal Constraint.
    active_relation_constraints = [
        c.name
        for c in m.unit.component_objects(pyo.Constraint, active=True)
        if "y_relation" in c.name or "arima" in c.name.lower()
    ]
    assert active_relation_constraints == ["unit.y_relation_fitted_2"]

    # Solving after the second swap reflects only the second fit's
    # coefficients (all y[t] free; the AR(1) recursion is fully determined
    # from the baked-in training data at t=0, so no exogenous fixing is
    # needed).
    m.obj = pyo.Objective(expr=0.0)
    solver = pyo.SolverFactory("ipopt")
    result = solver.solve(m, tee=False)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal

    coef2 = regressor2.coefficients
    expected_y1 = coef2.get("const", 0.0) + coef2["ar1"] * float(m.unit.y[0].value)
    assert float(m.unit.y[1].value) == pytest.approx(expected_y1, rel=1e-4)


@pytest.mark.component
@pytest.mark.needs_ipopt
@pytest.mark.parametrize(
    "order,auto,auto_kwargs",
    [
        ((1, 0, 0), False, {}),
        (None, True, {"max_p": 3, "max_q": 3}),
    ],
    ids=["explicit-ar1", "auto"],
)
def test_arima_roundtrip(order, auto, auto_kwargs):
    """Fit, build Pyomo surrogate, optimize exog controls, verify against direct fit."""
    np.random.seed(0)
    n_train = 100
    n_insample = 20
    n_fcst = 10
    n_opt = 10
    n_total = n_insample + n_fcst + n_opt
    idx = pd.date_range("2024-01-01", periods=n_train, freq="1h")

    # Generate AR(1) data
    y_values = np.zeros(n_train)
    for t in range(1, n_train):
        y_values[t] = 0.1 + 0.5 * y_values[t - 1] + np.random.normal(0, 0.05)
    y = pd.DataFrame({"biogas": y_values}, index=idx)

    feed = pd.Series(np.random.uniform(0.1, 1.0, size=n_train), index=idx, name="feed")
    X = pd.DataFrame({"feed": feed})

    regressor = ArimaRegressor(
        order=order, auto=auto, max_ar_persistence=None, **auto_kwargs
    ).fit(X, y)
    assert regressor.fitted is True

    spec = regressor.to_surrogate_spec(
        input_units={"feed": "dimensionless"},
        output_units="m^3/hr",
    )

    # Direct fit predictions for in-sample + forecast horizon
    insample_exog = X.iloc[-n_insample:].values
    forecast_exog = np.zeros((n_fcst, 1))
    all_exog = np.concatenate([insample_exog, forecast_exog])
    sm_all = np.asarray(
        regressor.model.predict(
            steps=n_insample + n_fcst,
            exog=all_exog,
            start=n_train - n_insample,
            dynamic=True,
        )
    )
    direct_insample = sm_all[:n_insample]
    direct_fcst = sm_all[n_insample:]

    target_biogas = float(y.iloc[-n_fcst:]["biogas"].mean())
    exog_bounds = {"feed": (float(X["feed"].min()), float(X["feed"].max()))}

    # Build Pyomo model
    m = pyo.ConcreteModel()
    start_idx = idx[n_train - n_insample]
    m.time_block = TimeBlock(
        start_date=start_idx.strftime("%Y-%m-%dT%H:%M"),
        end_date=(start_idx + pd.Timedelta(hours=n_total)).strftime("%Y-%m-%dT%H:%M"),
        time_step=1 * pyunits.hr,
    )
    m.props = SimpleAqueousFlow(has_pressure=False)
    m.unit = OpsBlock(property_package=m.props)
    m.unit.add_stream_ports()

    m.unit.add_component(
        "biogas_m3_hour",
        pyo.Var(
            m.time_block.time_index, initialize=0.0, units=pyunits.m**3 / pyunits.hr
        ),
    )
    m.unit.register_io_variable(m.unit.biogas_m3_hour, role="output")
    m.unit.add_component(
        "feed",
        pyo.Var(m.time_block.time_index, initialize=0.0, units=pyunits.dimensionless),
    )
    m.unit.register_io_variable(m.unit.feed, role="input")

    m.unit.add_component(
        "biogas_m3_hour_relation",
        pyo.Constraint(m.time_block.time_index, rule=lambda b, t: pyo.Constraint.Skip),
    )
    m.unit.register_relation(
        m.unit.biogas_m3_hour_relation, target=m.unit.biogas_m3_hour
    )

    surrogate = ArimaSurrogate(spec.data)
    m.unit.swap_relation("biogas_m3_hour_relation", surrogate)

    # Initialize with direct fit predictions
    for t in range(n_insample + n_fcst):
        m.unit.biogas_m3_hour[t].set_value(float(sm_all[t]))

    # Fix exog: in-sample + forecast
    for t in range(n_insample + n_fcst):
        m.unit.feed[t].set_value(float(all_exog[t].item()))
        m.unit.feed[t].fix()
    # Unfix and bound optimization window
    for t in range(n_insample + n_fcst, n_total):
        m.unit.feed[t].unfix()
        m.unit.feed[t].set_value(float(X["feed"].mean()))
        col_min, col_max = exog_bounds["feed"]
        m.unit.feed[t].setlb(col_min)
        m.unit.feed[t].setub(col_max)

    m.obj = pyo.Objective(
        expr=sum(
            (m.unit.biogas_m3_hour[t] - target_biogas) ** 2
            for t in range(n_insample + n_fcst, n_total)
        ),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory("ipopt")
    result = solver.solve(m, tee=False)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal

    pyomo_insample = np.array(
        [float(m.unit.biogas_m3_hour[t].value) for t in range(n_insample)]
    )
    pyomo_fcst = np.array(
        [
            float(m.unit.biogas_m3_hour[t].value)
            for t in range(n_insample, n_insample + n_fcst)
        ]
    )
    pyomo_opt = np.array(
        [
            float(m.unit.biogas_m3_hour[t].value)
            for t in range(n_insample + n_fcst, n_total)
        ]
    )

    insample_rmse = float(np.sqrt(np.mean((pyomo_insample - direct_insample) ** 2)))
    forecast_rmse = float(np.sqrt(np.mean((pyomo_fcst - direct_fcst) ** 2)))

    assert insample_rmse < 1e-4, f"In-sample RMSE too high: {insample_rmse}"
    assert forecast_rmse < 1e-4, f"Forecast RMSE too high: {forecast_rmse}"
    assert (
        abs(float(pyomo_opt.mean()) - target_biogas) < 0.5
    ), f"Optimized mean {pyomo_opt.mean():.4f} not close to target {target_biogas:.4f}"
