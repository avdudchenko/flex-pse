"""Example: ARIMA surrogate for biogas generation optimization.

This script demonstrates the Pyomo ArimaSurrogate built from a
directly-fitted ARIMA model.  The surrogate encodes the **mean
ARIMAX relationship** as a Pyomo Constraint, making it directly usable
inside optimization and control problems.

**Restriction**: Only non-differenced AR/MA models are supported (d=0, D=0).

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
    "..",
    "..",
    "flexparameterize",
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


def fit_best_arima(X, y, max_p=3, max_q=3):
    """Try auto ARIMA first, then simplify the selected model.

    If the auto-selected model is rejected (e.g. high AR persistence),
    this function systematically tries simpler models derived from the
    selected order:

    1. Reduce p by 1, keep q
    2. Reduce q by 1, keep p
    3. Pure MA(q)
    4. Pure AR(p)
    5. Mixed ARMA with smaller orders
    6. Fall back to MA(1) and AR(1)

    Returns the first successfully fitted ArimaRegressor that passes
    validation.
    """
    # Try auto mode first with stationary=True to enforce stable AR
    try:
        print("Trying auto ARIMA (stationary=True)...")
        regressor = ArimaRegressor(auto=True, max_p=max_p, max_q=max_q, stationary=True)
        regressor.fit(X, y)
        print(f"  Auto selected: order={regressor._order}")
        return regressor
    except Exception as e:
        print(f"  Auto ARIMA failed: {e}")

    # We don't know the auto-selected order because it failed,
    # so we try a systematic simplification of likely candidates
    candidates = []

    # Generate candidates by simplifying from likely auto selections
    # Start with mixed ARMA, then simplify
    for p in range(max_p, 0, -1):
        for q in range(max_q, 0, -1):
            if p == 0 or q == 0:
                continue  # Skip pure AR/MA here, add them later
            candidates.append((p, 0, q))

    # Add pure MA models
    for q in range(max_q, 0, -1):
        candidates.append((0, 0, q))

    # Add pure AR models
    for p in range(max_p, 0, -1):
        candidates.append((p, 0, 0))

    # Try each candidate
    for order in candidates:
        try:
            print(f"Trying ARIMA{order}...")
            regressor = ArimaRegressor(order=order)
            regressor.fit(X, y)
            print(f"  Success: order={order}")
            return regressor
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    raise RuntimeError("No valid ARIMA model found. Try different data or parameters.")


print(f"Train size: {len(train_df)} rows ({len(train_df)//24} days)")
print(f"Validation size: {len(valid_df)} rows ({len(valid_df)//24} days)")

regressor = fit_best_arima(X_train, y_train, max_p=3, max_q=3)
spec = regressor.to_surrogate_spec(
    input_units={col: "dimensionless" for col in EXOG_COLS},
    output_units="m^3/hr",
)

print(f"Fitted order: {spec.data['order']}")
print(f"AR coefs: {spec.data['ar_coefs']}")
print(f"MA coefs: {spec.data['ma_coefs']}")
print(f"Exog coefs: {spec.data['exog_coefs']}")
print(f"Const: {spec.data['const']}")
print(f"AIC: {regressor.metrics['aic']:.2f}")
print(f"RMSE: {regressor.metrics['rmse']:.4f}")


# ---------------------------------------------------------------------------
# Helper: build a Pyomo model with ArimaSurrogate
# ---------------------------------------------------------------------------
def build_pyomo_model(start_idx, n_points, spec_data):
    """Return (model, unit, target_var, constraint_indexed).

    Args:
        start_idx: pandas Timestamp for the start of the modelling window.
        n_points: number of time points to include.
        spec_data: the SurrogateSpec ``data`` dict.

    Returns:
        ``(m, unit, y_var, con)`` where ``con[t]`` is the ARIMA equality
        constraint at time index ``t``.
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

    surrogate = ArimaSurrogate(spec_data)
    surrogate.build(m.unit, m.unit.biogas_m3_hour)

    con = m.unit.find_component(f"{TARGET_COL}_arima_eq")
    return m, m.unit, m.unit.biogas_m3_hour, con


# ---------------------------------------------------------------------------
# Scenario 1: in-sample + out-of-sample forecast validation
# ---------------------------------------------------------------------------
def scenario_insample_and_forecast():
    """Compare Pyomo surrogate and direct fit on train/validation split, then optimize.

    The model is fitted on ``train_df`` (all data except the last 2 days).

    - **In-sample**: the last day of training data.  Pyomo uses observed
      y[t-1] as AR lag inputs; direct fit uses its own fittedvalues.
    - **Out-of-sample forecast**: the 2-day held-out validation period.
      Both methods forecast recursively from the end of training.
    - **Optimization**: the 2 days after validation.  Exogenous inputs are
      free decision variables bounded by validation-period ranges.  The
      objective drives biogas output toward the validation-period mean.
    """
    n_insample = _HOURS_PER_DAY  # last day of training data
    n_fcst = _N_VALID_HOURS  # 2-day validation forecast
    n_opt = _N_OPT_HOURS  # 1-day optimization after validation

    # Last day of training data (in-sample window)
    train_end_idx = train_df.index[-1]
    insample_data = train_df.iloc[-n_insample:]
    # Validation data (out-of-sample forecast window)
    forecast_exog = X_valid
    forecast_obs = y_valid[TARGET_COL]

    # Direct fit in-sample dynamic prediction (recursive, matching Pyomo behavior)
    insample_exog = train_df[EXOG_COLS].iloc[-n_insample:].values
    insample_start = len(train_df) - n_insample
    sm_insample = regressor.model.predict(
        steps=n_insample,
        exog=insample_exog,
        start=insample_start,
        dynamic=True,
    )

    # Direct fit out-of-sample forecast (mean-equation recursion)
    sm_fcst_mean = regressor.model.predict(steps=n_fcst, exog=forecast_exog.values)

    # Target for optimization: mean biogas during validation period
    target_biogas = float(forecast_obs.mean())
    print(f"Target biogas for optimization: {target_biogas:.6f} m³/hr")

    # Bounds for exogenous variables from validation data
    exog_bounds = {}
    for col in EXOG_COLS:
        exog_bounds[col] = (
            float(X_valid[col].min()),
            float(X_valid[col].max()),
        )

    # Build Pyomo model: in-sample (1 day) + validation (2 days) + optimization (2 days)
    n_total = n_insample + n_fcst + n_opt
    m, unit, y_var, con = build_pyomo_model(
        start_idx=train_end_idx - pd.Timedelta(hours=n_insample - 1),
        n_points=n_total,
        spec_data=spec.data,
    )

    # Fix exog inputs for in-sample + validation
    for col in EXOG_COLS:
        exog_var = getattr(unit, col)
        insample_vals = train_df[col].iloc[-n_insample:].values
        valid_vals = valid_df[col].values
        all_vals = np.concatenate([insample_vals, valid_vals])
        for t in range(n_insample + n_fcst):
            exog_var[t].set_value(float(all_vals[t]))
            exog_var[t].fix()
    # Now optimize the last 2 days
    # Unfix exog variables for optimization period
    for col in EXOG_COLS:
        exog_var = getattr(unit, col)
        for t in range(n_insample + n_fcst, n_total):
            exog_var[t].unfix()
            # Set initial values to validation-period means
            exog_var[t].set_value(float(X_valid[col].mean()))

    # Add bounds on exog variables for optimization period
    for col in EXOG_COLS:
        exog_var = getattr(unit, col)
        col_min, col_max = exog_bounds[col]
        for t in range(n_insample + n_fcst, n_total):
            exog_var[t].setlb(col_min)
            exog_var[t].setub(col_max)
    # Objective: drive biogas output toward target during optimization period
    m.obj = pyo.Objective(
        expr=sum(
            (y_var[t] - target_biogas) ** 2 for t in range(n_insample + n_fcst, n_total)
        ),
        sense=pyo.minimize,
    )

    # Solve

    solver = pyo.SolverFactory("ipopt")
    result = solver.solve(m, tee=True)
    assert_optimal_termination(result)
    # Extract the solved y values

    time_set = list(y_var.index_set())
    pyomo_result = np.array([float(y_var[t].value) for t in time_set])
    pyomo_insample = pyomo_result[:n_insample]
    pyomo_fcst = pyomo_result[n_insample : n_insample + n_fcst]
    pyomo_opt = np.array(
        [float(y_var[t].value) for t in range(n_insample + n_fcst, n_total)]
    )

    # Print diagnostics
    print("\nScenario 1: In-sample + Validation + Optimization")
    print(f"  Train size: {len(train_df)} rows")
    print(f"  Validation size: {n_fcst} rows ({_N_VALID_DAYS} days)")
    print(f"  Optimization size: {n_opt} rows ({_N_VALID_DAYS} days)")
    print(
        "  In-sample: Direct fit RMSE vs observed:"
        f" {np.sqrt(np.mean((sm_insample - insample_data[TARGET_COL].values)**2)):.6f}"
    )
    print(
        "  In-sample: Pyomo RMSE vs observed:"
        f" {np.sqrt(np.mean((pyomo_insample - insample_data[TARGET_COL].values)**2)):.6f}"  # noqa: E501
    )
    print(
        "  In-sample: Pyomo RMSE vs direct fit:"
        f" {np.sqrt(np.mean((pyomo_insample - sm_insample)**2)):.6f}"
    )
    print(
        "  Forecast: Direct fit RMSE vs observed:"
        f" {np.sqrt(np.mean((sm_fcst_mean - forecast_obs.values)**2)):.6f}"
    )
    print(
        "  Forecast: Pyomo RMSE vs observed:"
        f" {np.sqrt(np.mean((pyomo_fcst - forecast_obs.values)**2)):.6f}"
    )
    print(
        "  Forecast: Direct fit mean:"
        f" {sm_fcst_mean.mean():.6f}  (observed: {forecast_obs.mean():.6f})"
    )
    print(
        "  Forecast: Pyomo mean:"
        f" {pyomo_fcst.mean():.6f}  (observed: {forecast_obs.mean():.6f})"
    )
    if pyomo_opt is not None:
        print(
            "  Optimization: Pyomo mean:"
            f" {pyomo_opt.mean():.6f}  (target: {target_biogas:.6f})"
        )

    # Plot
    fig = plt.figure(figsize=(14, 10))

    # Main plot: biogas output
    ax1 = plt.subplot(3, 1, 1)
    insample_idx = insample_data.index
    valid_idx = valid_df.index
    opt_idx = pd.date_range(
        start=valid_df.index[-1] + pd.Timedelta(hours=1),
        periods=n_opt,
        freq="1h",
    )

    # In-sample (last day of training)
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

    # Forecast (2-day validation)
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

    # Optimization (2 days after validation)
    if pyomo_opt is not None:
        ax1.plot(
            opt_idx,
            pyomo_opt,
            "g-",
            label="Pyomo optimized (2-day)",
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
    ax1.set_title("Scenario 1: In-Sample + 2-Day Forecast + 2-Day Optimization")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(f"{TARGET_COL} (m³/hr)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: feed_volume_kg
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

    # Subplot 3: TS_pct
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

    out_path = os.path.join(FIG_DIR, "scenario1_insample_forecast_optimization.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running Scenario 1: in-sample + forecast comparison ...")
    scenario_insample_and_forecast()
