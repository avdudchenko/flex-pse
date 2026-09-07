"""Example: ARIMA surrogate for biogas generation optimization.

This script demonstrates the Pyomo ArimaSurrogate built from a
directly-fitted ARIMA model.  The surrogate encodes the **mean
ARIMAX relationship** as a Pyomo Constraint, making it directly usable
inside optimization and control problems.

**Restriction**: ``d`` may be 0 or 1; seasonal differencing (D) must be 0,
and seasonal AR/MA terms are not supported at all.

**How it works**:
- Historical data is fixed to observed values
- Future exogenous inputs are decision variables (control inputs)
- The surrogate constraint determines future y[t] based on:
  * AR lag from y[t-1] (fixed for historical, variable for future)
  * MA terms from baked-in residuals
  * Exogenous inputs (the control variables)

**What this example shows**:
1. **Train/validation split**: Fit on all data except the last 2 days,
     validate on the held-out period.
2. **In-sample validation**: Compare Pyomo surrogate vs direct fit
     fitted values on the last day of training data.
3. **Out-of-sample forecast**: 2-day ahead forecast from both methods,
     compared against observed validation data.
4. **Optimization**: Choose feed volume to maximize biogas while
     staying within bounds.
5. **Model comparison**: Compare multiple ARIMA orders:
     ARIMA(1,0,0), ARIMA(0,1,0), ARIMA(0,0,1), ARIMA(1,1,1), and auto.

**Important notes**:
- The surrogate implements the MEAN ARIMA equation, not a Kalman filter.
    Therefore:
   * In-sample predictions match the direct fit's fitted values exactly
   * Multi-step forecasts use the mean-equation recursion
   * The surrogate is deterministic: same inputs -> same output
   * This is IDEAL for optimization/control where you need derivatives

Figures are saved to ``examples/figures/``.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import units as pyunits

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from pyomo.opt import assert_optimal_termination

from flexops.core.ops_block import OpsBlock
from flexops.core.time_block import TimeBlock
from flexops.properties.simple_aqueous import SimpleAqueousFlow
from flexops.surrogates import ArimaSurrogate
from flexparameterize.regression.arima import ArimaRegressor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
FIG_DIR = os.path.join(_HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "tests",
    "test_time_series_data",
    "imputed_bio_gas_generation.csv",
)
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).set_index("timestamp")
df = df.sort_index()

TARGET_COL = "biogas_m3_hour"
EXOG_COLS = ["feed_volume_kg", "TS_pct"]

df_hour = df.resample("1h").mean().dropna(subset=[TARGET_COL] + EXOG_COLS)

print(f"Data shape: {df_hour.shape}")
print(f"Date range: {df_hour.index.min()} -> {df_hour.index.max()}")

# ---------------------------------------------------------------------------
# Fit ARIMA model with smart fallback
# ---------------------------------------------------------------------------
# Train on all data except the last 2 days; hold those out for validation.
_N_VALID_DAYS = 2
_N_OPT_DAYS = 1
_HOURS_PER_DAY = 24
_N_VALID_HOURS = _N_VALID_DAYS * _HOURS_PER_DAY
_N_OPT_HOURS = _N_OPT_DAYS * _HOURS_PER_DAY
train_df = df_hour.iloc[:-_N_VALID_HOURS]
valid_df = df_hour.iloc[-_N_VALID_HOURS:]

y_train = train_df[[TARGET_COL]]
X_train = train_df[EXOG_COLS]
y_valid = valid_df[[TARGET_COL]]
X_valid = valid_df[EXOG_COLS]


def fit_model(X, y, order=None, auto=False, max_p=3, max_q=3):
    """Fit an ARIMA model with the given order or auto-select."""
    if auto:
        print("Trying auto ARIMA (stationary=True)...")
        regressor = ArimaRegressor(auto=True, max_p=max_p, max_q=max_q, stationary=True)
        regressor.fit(X, y)
        print(f"  Auto selected: order={regressor.order}")
        return regressor
    elif order is not None:
        print(f"Fitting ARIMA{order}...")
        regressor = ArimaRegressor(order=order)
        regressor.fit(X, y)
        print(f"  Success: order={order}")
        return regressor
    else:
        raise ValueError("Must specify either order or auto=True")


def run_scenario_for_order(order=None, auto=False, label=""):
    """Run in-sample + forecast comparison for a specific ARIMA order.

    Args:
        order: ARIMA order tuple (p, d, q), or None if auto=True
        auto: if True, use auto ARIMA selection
        label: string label for the figure filename

    Returns:
        dict with keys: 'regressor', 'spec', 'sm_insample', 'sm_fcst_mean',
        'pyomo_insample', 'pyomo_fcst', 'pyomo_opt', 'target_biogas'
    """
    print(f"\n{'='*60}")
    if auto:
        print(f"Running scenario for AUTO ARIMA ({label})")
    else:
        print(f"Running scenario for ARIMA{order} ({label})")
    print(f"{'='*60}")

    # Fit model
    regressor = fit_model(X_train, y_train, order=order, auto=auto)
    spec = regressor.to_surrogate_spec(
        input_units={col: "dimensionless" for col in EXOG_COLS},
        output_units="m^3/hr",
    )

    print(f"Fitted order: {tuple(spec.data['order'])}")
    print(f"AR coefs: {spec.data['ar_coefs']}")
    print(f"MA coefs: {spec.data['ma_coefs']}")
    print(f"Exog coefs: {spec.data['exog_coefs']}")
    print(f"Const: {spec.data['const']}")
    print(f"AIC: {regressor.metrics['aic']:.2f}")
    print(f"RMSE: {regressor.metrics['rmse']:.4f}")

    n_insample = _HOURS_PER_DAY
    n_fcst = _N_VALID_HOURS
    n_opt = _N_OPT_HOURS

    train_end_idx = train_df.index[-1]
    insample_data = train_df.iloc[-n_insample:]
    forecast_exog = X_valid
    forecast_obs = y_valid[TARGET_COL]

    insample_exog = train_df[EXOG_COLS].iloc[-n_insample:].values
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
    print(f"Target biogas for optimization: {target_biogas:.6f} m³/hr")

    exog_bounds = {}
    for col in EXOG_COLS:
        exog_bounds[col] = (
            float(X_valid[col].min()),
            float(X_valid[col].max()),
        )

    n_total = n_insample + n_fcst + n_opt
    m, unit, y_var, con = build_pyomo_model(
        start_idx=train_end_idx - pd.Timedelta(hours=n_insample - 1),
        n_points=n_total,
        spec_data=spec.data,
    )

    for col in EXOG_COLS:
        exog_var = getattr(unit, col)
        insample_vals = train_df[col].iloc[-n_insample:].values
        valid_vals = valid_df[col].values
        all_vals = np.concatenate([insample_vals, valid_vals])
        for t in range(n_insample + n_fcst):
            exog_var[t].set_value(float(all_vals[t]))
            exog_var[t].fix()

    for col in EXOG_COLS:
        exog_var = getattr(unit, col)
        for t in range(n_insample + n_fcst, n_total):
            exog_var[t].unfix()
            exog_var[t].set_value(float(X_valid[col].mean()))

    for col in EXOG_COLS:
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
    print(f"  In-sample: Pyomo RMSE vs direct fit: {insample_rmse:.6f}")
    print(f"  Forecast: Pyomo RMSE vs direct fit: {forecast_rmse:.6f}")
    print(
        f"  Optimization: Pyomo mean: {pyomo_opt.mean():.6f}"
        f"  (target: {target_biogas:.6f})"
    )

    fig = plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(3, 1, 1)
    insample_idx = insample_data.index
    valid_idx = valid_df.index
    opt_idx = pd.date_range(
        start=valid_df.index[-1] + pd.Timedelta(hours=1),
        periods=n_opt,
        freq="1h",
    )

    ax1.plot(
        insample_idx,
        insample_data[TARGET_COL].values,
        "k-",
        label="Observed (in-sample, last day of train)",
        linewidth=2,
    )
    ax1.plot(
        insample_idx,
        sm_insample,
        "b--",
        label="Direct fit (in-sample)",
        linewidth=2,
        alpha=0.8,
    )
    ax1.plot(
        insample_idx,
        pyomo_insample,
        "r--",
        label="Pyomo surrogate (in-sample)",
        linewidth=2,
        alpha=0.8,
    )
    ax1.plot(
        valid_idx,
        sm_fcst_mean,
        "b-",
        label="Direct fit (2-day forecast)",
        linewidth=2,
        marker="o",
    )
    ax1.plot(
        valid_idx,
        pyomo_fcst,
        "r-",
        label="Pyomo surrogate (2-day forecast)",
        linewidth=2,
        marker="x",
    )
    ax1.plot(
        valid_idx,
        forecast_obs.values,
        "k--",
        label="Observed (validation)",
        linewidth=1.5,
        alpha=0.7,
    )
    if pyomo_opt is not None:
        ax1.plot(
            opt_idx,
            pyomo_opt,
            "g-",
            label="Pyomo optimized (1-day)",
            linewidth=2,
            marker="^",
        )
        ax1.axhline(
            y=target_biogas,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label=f"Target ({target_biogas:.3f})",
        )

    ax1.axvline(x=insample_idx[-1], color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(x=valid_idx[-1], color="gray", linestyle="--", alpha=0.5)
    fitted_order = tuple(spec.data["order"])
    ax1.set_title(f"Scenario: ARIMA{fitted_order} ({label})")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(f"{TARGET_COL} (m³/hr)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(
        insample_idx,
        train_df["feed_volume_kg"].iloc[-n_insample:].values,
        "k-",
        label="Observed (in-sample)",
        linewidth=2,
    )
    ax2.plot(
        valid_idx,
        valid_df["feed_volume_kg"].values,
        "b--",
        label="Validation (fixed)",
        linewidth=2,
    )
    if pyomo_opt is not None:
        opt_feed = [
            float(unit.feed_volume_kg[t].value)
            for t in range(n_insample + n_fcst, n_total)
        ]
        ax2.plot(
            opt_idx,
            opt_feed,
            "g-",
            label="Optimized",
            linewidth=2,
            marker="^",
        )
    ax2.axvline(x=insample_idx[-1], color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(x=valid_idx[-1], color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("feed_volume_kg")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(
        insample_idx,
        train_df["TS_pct"].iloc[-n_insample:].values,
        "k-",
        label="Observed (in-sample)",
        linewidth=2,
    )
    ax3.plot(
        valid_idx,
        valid_df["TS_pct"].values,
        "b--",
        label="Validation (fixed)",
        linewidth=2,
    )
    if pyomo_opt is not None:
        opt_ts = [
            float(unit.TS_pct[t].value) for t in range(n_insample + n_fcst, n_total)
        ]
        ax3.plot(
            opt_idx,
            opt_ts,
            "g-",
            label="Optimized",
            linewidth=2,
            marker="^",
        )
    ax3.axvline(x=insample_idx[-1], color="gray", linestyle="--", alpha=0.5)
    ax3.axvline(x=valid_idx[-1], color="gray", linestyle="--", alpha=0.5)
    ax3.set_ylabel("TS_pct")
    ax3.set_xlabel("Time")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, f"arima_{label}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    return {
        "regressor": regressor,
        "spec": spec,
        "sm_insample": sm_insample,
        "sm_fcst_mean": sm_fcst_mean,
        "pyomo_insample": pyomo_insample,
        "pyomo_fcst": pyomo_fcst,
        "pyomo_opt": pyomo_opt,
        "target_biogas": target_biogas,
        "forecast_obs": forecast_obs,
        "insample_data": insample_data,
        "n_insample": n_insample,
        "n_fcst": n_fcst,
        "n_opt": n_opt,
    }


print(f"Train size: {len(train_df)} rows ({len(train_df)//24} days)")
print(f"Validation size: {len(valid_df)} rows ({len(valid_df)//24} days)")


# ---------------------------------------------------------------------------
# Helper: build a Pyomo model with ArimaSurrogate
# ---------------------------------------------------------------------------
def build_pyomo_model(start_idx, n_points, spec_data):
    """Return (model, unit, target_var, constraint_indexed).

    Attaches the ``ArimaSurrogate`` the documented way — a placeholder
    relation registered via ``register_relation`` and then replaced via
    ``swap_relation`` — rather than calling ``ArimaSurrogate.build()``
    directly, since ``build()`` itself no longer enforces the relationship
    (``swap_relation`` is the sole place that constraint is built; see
    ``flexops.surrogates.base.Surrogate``).

    Args:
        start_idx: pandas Timestamp for the start of the modelling window.
        n_points: number of time points to include.
        spec_data: the SurrogateSpec ``data`` dict.

    Returns:
        ``(m, unit, y_var, con)`` where ``con[t]`` is the fitted ARIMA
        equality constraint at time index ``t``.
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
        TARGET_COL,
        pyo.Var(
            m.time_block.time_index, initialize=0.0, units=pyunits.m**3 / pyunits.hr
        ),
    )
    m.unit.register_io_variable(m.unit.biogas_m3_hour, role="output")

    for col in EXOG_COLS:
        m.unit.add_component(
            col,
            pyo.Var(
                m.time_block.time_index, initialize=0.0, units=pyunits.dimensionless
            ),
        )
        m.unit.register_io_variable(getattr(m.unit, col), role="input")

    relation_name = f"{TARGET_COL}_relation"
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
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    orders_to_try = [
        ((1, 0, 0), "arima_1_0_0"),
        ((0, 1, 0), "arima_0_1_0"),
        ((0, 0, 1), "arima_0_0_1"),
        ((1, 1, 1), "arima_1_1_1"),
        ("auto", "arima_auto"),
    ]

    results = {}
    for order, label in orders_to_try:
        try:
            if order == "auto":
                results[label] = run_scenario_for_order(
                    order=None, auto=True, label=label
                )
            else:
                results[label] = run_scenario_for_order(order=order, label=label)
        except Exception as e:
            print(f"Failed for {label}: {e}")

    print(f"\nCompleted {len(results)}/{len(orders_to_try)} scenarios successfully.")
