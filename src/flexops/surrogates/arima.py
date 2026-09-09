"""ArimaSurrogate: a Pyomo time-indexed Constraint implementing the fitted
ARIMA relationship.

``data`` (the ``SurrogateSpec.data`` produced by
:class:`~flexparameterize.regression.arima.ArimaRegressor`) is::

    {"input_variables":  {"feed_volume_kg": "kg", "TS_pct": "dimensionless"},
     "output_variables": {"biogas_m3_hour": "m^3/hr"},
     "exogenous_variables": ["feed_volume_kg", "TS_pct"],
     "order": [p, d, q],
     "seasonal_order": [P, D, Q, m] | None,
     "const": <float>,
     "drift": <float> | None,        # only present when include_drift=True and d=1
     "ar_coefs": [phi_1, ..., phi_p],
     "ma_coefs": [theta_1, ..., theta_q],
     "exog_coefs": [beta_1, ..., beta_k],
     "_residuals": [r_0, ..., r_{n-1}],   # fitted in-sample residuals
     "init_values": [y_{-p}, ..., y_{-1}]}   # optional; zeros when absent

:meth:`build` returns a ``body(t)`` callable (per the
:class:`~flexops.surrogates.base.Surrogate` contract). All trained coefficients,
constants, residuals, and init values are **inlined directly into the returned
expression** — no Params are added to ``unit``. ``build`` itself does **not**
enforce ``target == body``;
:meth:`~flexops.core.ops_block.OpsBlockData.swap_relation` is the sole place
that constraint is built (as ``"{relation_name}_fitted"``), so
ArimaSurrogate does not double up on it. A re-fit-and-reswap simply builds a
fresh ``body`` with the new coefficients; ``swap_relation`` deactivates the old
constraint and attaches the new one.

Time indexing
~~~~~~~~~~~~~
The Pyomo ``TimeBlock`` uses integer indices ``0 … n-1``.  The ARIMA
equation at index ``t`` is::

    y[t] = c + sum(phi_j * y[t-j]) + sum(theta_j * resid[t-j])
           + sum(beta_k * x_k[t])

where:

* ``y[t-j]`` for ``t-j >= 0`` is the previous time step's target value;
  for ``t-j < 0`` it falls back to the ``init_values`` from ``data``
  (representing ``y[-p]`` … ``y[-1]``).
* ``resid[t-j]`` is the fitted residual at time ``t-j`` (0 for
  ``t-j < 0`` or ``t-j >= n``), taken directly from ``data["_residuals"]``.
* ``eta[t]`` is **not** introduced as a free Var; the surrogate is the
  *mean* ARIMAX relationship (innovations set to zero), which is the
  form useful for optimisation.  The fitted residuals are baked in as
  data so the in-sample path reproduces the statsforecast fitted values.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import units as pyunits

from flexcore.config.schema import SurrogateType
from flexcore.exceptions import FlexConfigError
from flexops.core.time_block import find_time_block
from flexops.core.units import parse_units
from flexops.surrogates.base import Surrogate

_DATA_KEYS = (
    "input_variables",
    "output_variables",
    "exogenous_variables",
    "order",
    "const",
    "ar_coefs",
    "ma_coefs",
    "exog_coefs",
    "_residuals",
    "training_start_date",
    "training_time_step_seconds",
    "training_y_values",
)
"""tuple: keys that must be present in ``data``."""

_OPTIONAL_KEYS = ("seasonal_order", "init_values", "drift")
"""tuple: keys that are optional in ``data``."""


class ArimaSurrogate(Surrogate):
    """A fitted ARIMA relationship, built as a Pyomo Constraint.

    ``data`` is the dict emitted by
    :meth:`~flexparameterize.regression.arima.ArimaRegressor.to_surrogate_spec`.

    Attributes:
        data: The validated data mapping.
    """

    surrogate_type: ClassVar[SurrogateType] = SurrogateType.ARIMA

    def _validate(self) -> None:
        """Validate the ARIMA ``data`` contract.

        Raises:
            FlexConfigError: If a required key is missing or a value has the
                wrong type or shape.
        """
        unknown = sorted(set(self.data) - set(_DATA_KEYS) - set(_OPTIONAL_KEYS))
        if unknown:
            raise FlexConfigError(
                f"ARIMA surrogate data carries unknown key(s) {unknown}; "
                f"it may only have {_DATA_KEYS} "
                f"(plus optional {_OPTIONAL_KEYS}).",
                field="data",
                value=unknown,
            )

        missing = sorted(set(_DATA_KEYS) - set(self.data))
        if missing:
            raise FlexConfigError(
                f"ARIMA surrogate data is missing key(s) {missing}; it "
                f"must have {_DATA_KEYS}.",
                field="data",
                value=missing,
            )

        inputs = self.data["input_variables"]
        if not isinstance(inputs, dict):
            raise FlexConfigError(
                "ARIMA surrogate 'input_variables' must be a "
                f"{{name: units}} mapping, got {inputs!r}.",
                field="input_variables",
                value=inputs,
            )
        exog = self.data["exogenous_variables"]
        if not exog and inputs:
            raise FlexConfigError(
                "ARIMA surrogate has exogenous_variables=[] but "
                f"'input_variables' is non-empty {sorted(inputs)}; "
                "remove the unused entries or list them in "
                "'exogenous_variables'.",
                field="input_variables",
                value=sorted(inputs),
            )

        outputs = self.data["output_variables"]
        if not isinstance(outputs, dict) or len(outputs) != 1:
            raise FlexConfigError(
                "ARIMA surrogate 'output_variables' must be a single "
                f"{{name: units}} entry, got {outputs!r}.",
                field="output_variables",
                value=outputs,
            )

        if not isinstance(exog, list):
            raise FlexConfigError(
                f"ARIMA surrogate 'exogenous_variables' must be a list "
                f"of input-variable names, got {exog!r}.",
                field="exogenous_variables",
                value=exog,
            )
        unknown_exog = [name for name in exog if name not in inputs]
        if unknown_exog:
            raise FlexConfigError(
                f"ARIMA 'exogenous_variables' names {unknown_exog}, not in "
                f"'input_variables' ({sorted(inputs)}).",
                field="exogenous_variables",
                value=unknown_exog,
            )

        order = self.data["order"]
        if (
            not isinstance(order, (list, tuple))
            or len(order) != 3
            or not all(isinstance(v, int) and v >= 0 for v in order)
        ):
            raise FlexConfigError(
                f"ARIMA 'order' must be a 3-element [p,d,q] list of "
                f"non-negative ints, got {order!r}.",
                field="order",
                value=order,
            )

        seasonal = self.data.get("seasonal_order")
        if seasonal is not None:
            if (
                not isinstance(seasonal, (list, tuple))
                or len(seasonal) != 4
                or not all(isinstance(v, int) and v >= 0 for v in seasonal)
            ):
                raise FlexConfigError(
                    f"ARIMA 'seasonal_order' must be a 4-element "
                    f"[P,D,Q,m] list of non-negative ints, got {seasonal!r}.",
                    field="seasonal_order",
                    value=seasonal,
                )

        p, d, q = order
        ar_coefs = self.data["ar_coefs"]
        ma_coefs = self.data["ma_coefs"]
        exog_coefs = self.data["exog_coefs"]
        residuals = self.data.get("_residuals", [])
        if not isinstance(ar_coefs, list) or len(ar_coefs) != p:
            raise FlexConfigError(
                f"ARIMA 'ar_coefs' must be a list of {p} floats "
                f"(matching order[0]={p}), got {ar_coefs!r}.",
                field="ar_coefs",
                value=ar_coefs,
            )
        if not isinstance(ma_coefs, list) or len(ma_coefs) != q:
            raise FlexConfigError(
                f"ARIMA 'ma_coefs' must be a list of {q} floats "
                f"(matching order[2]={q}), got {ma_coefs!r}.",
                field="ma_coefs",
                value=ma_coefs,
            )
        if not isinstance(exog_coefs, list) or len(exog_coefs) != len(exog):
            raise FlexConfigError(
                f"ARIMA 'exog_coefs' must be a list of {len(exog)} floats "
                f"(one per 'exogenous_variables' entry), got {exog_coefs!r}.",
                field="exog_coefs",
                value=exog_coefs,
            )
        if not isinstance(residuals, list):
            raise FlexConfigError(
                f"ARIMA '_residuals' must be a list, got {residuals!r}.",
                field="_residuals",
                value=residuals,
            )

        for field, values in (
            ("ar_coefs", ar_coefs),
            ("ma_coefs", ma_coefs),
            ("exog_coefs", exog_coefs),
            ("_residuals", residuals),
        ):
            for i, v in enumerate(values):
                try:
                    float(v)
                except (TypeError, ValueError) as exc:
                    raise FlexConfigError(
                        f"ARIMA '{field}[{i}]' must be a number, got {v!r}.",
                        field=field,
                        value=v,
                    ) from exc

        init = self.data.get("init_values")
        if init is not None:
            expected_len = p if d == 0 else d
            if not isinstance(init, list) or len(init) != expected_len:
                raise FlexConfigError(
                    f"ARIMA 'init_values' must be a list of {expected_len} floats "
                    f"(matching d={d}), got {init!r}.",
                    field="init_values",
                    value=init,
                )
            for i, v in enumerate(init):
                try:
                    float(v)
                except (TypeError, ValueError) as exc:
                    raise FlexConfigError(
                        f"ARIMA 'init_values[{i}]' must be a number, " f"got {v!r}.",
                        field="init_values",
                        value=v,
                    ) from exc

        for field, mapping in (
            ("input_variables", inputs),
            ("output_variables", outputs),
        ):
            for name, units in mapping.items():
                if not name or not isinstance(units, str) or not units:
                    raise FlexConfigError(
                        f"ARIMA surrogate '{field}' entry {name!r} must "
                        f"map to a non-empty units string, got {units!r}.",
                        field=field,
                        value=name,
                    )
                parse_units(units)

    @property
    def input_variables(self) -> dict[str, str]:
        """Return the exogenous input variable names and their declared units."""
        return dict(self.data["input_variables"])

    @property
    def output_variables(self) -> dict[str, str]:
        """Return the one declared output variable name and its units."""
        return dict(self.data["output_variables"])

    def build(self, unit, target):
        """Return ``body(t)`` evaluating the ARIMA equation at time ``t``.

        Per the :class:`~flexops.surrogates.base.Surrogate` contract, this
        does **not** itself enforce ``target[t] == body(t)`` —
        :meth:`~flexops.core.ops_block.OpsBlockData.swap_relation` is the
        sole place that constraint is built. All trained coefficients,
        constants, residuals, and init values are inlined directly into the
        returned expression; no Params or other components are added to
        ``unit``.

        ``body(t)`` is well-defined for every time index ``t`` in the model
        horizon. For ``d==0``, ``t < p`` falls back to ``init_values``; for
        ``d==1`` and ``offset==0`` (model starts exactly at training start),
        the first ``p+1`` time steps return ``pyomo.environ.Constraint.Skip``
        because the differenced recursion needs ``y[-1]``, which does not
        exist. The caller should fix ``target[0]...target[p]`` to the first
        ``p+1`` training values so the remaining constraints are
        well-determined. For ``d==1`` and ``offset>0``, the full horizon is
        built using the training value immediately before the model start.
        MA terms use zero for negative residual indices.

        Args:
            unit: The :class:`~flexops.core.ops_block.OpsBlockData` the
                surrogate is built on.
            target: The time-indexed ``Var``/``Reference`` the relationship
                determines.

        Returns:
            A callable ``body(t)`` returning a Pyomo expression for time
            index ``t``.

        Raises:
            FlexConfigError: If the model's ``TimeBlock`` time step does not
                match the training time step, if the model starts before or
                too far beyond the training window, or if ``d==1`` and the
                model starts more than one time step before the training data
                (so the required pre-start value is unavailable).
        """
        output_units = parse_units(next(iter(self.output_variables.values())))

        declared_inputs = {
            name: (
                unit.resolve_variable(name, field="input_variables"),
                parse_units(units),
            )
            for name, units in self.input_variables.items()
        }

        order = self.data["order"]
        p, d, q = order
        const = float(self.data["const"])
        ar_coefs = [float(v) for v in self.data["ar_coefs"]]
        ma_coefs = [float(v) for v in self.data["ma_coefs"]]
        exog_coefs = [float(v) for v in self.data["exog_coefs"]]
        exog_names: list[str] = list(self.data["exogenous_variables"])
        drift = self.data.get("drift", None)
        if drift is not None:
            drift = float(drift)

        # Fitted residuals from the direct-fit model (length n_train).
        residuals: list[float] = self.data.get("_residuals", [0.0] * p)
        n_resid = len(residuals)

        # Training metadata for time-alignment.
        training_start_str: str = self.data["training_start_date"]
        training_dt_seconds: float = float(self.data["training_time_step_seconds"])
        training_y_values: list[float] = self.data["training_y_values"]
        n_train = len(training_y_values)

        time_block = find_time_block(unit.model())
        model_start = time_block.datetime_index[0]
        training_start = pd.Timestamp(training_start_str)
        model_dt_seconds = float(time_block._step_seconds)

        if abs(model_dt_seconds - training_dt_seconds) > 1e-6:
            raise FlexConfigError(
                f"ARIMA surrogate training time step ({training_dt_seconds}s) "
                f"does not match the model TimeBlock time step ({model_dt_seconds}s). "
                f"Re-fit the model with data at the same resolution as the "
                f"TimeBlock, or adjust the TimeBlock time_step.",
                field="training_time_step_seconds",
                value=training_dt_seconds,
            )

        offset = int(
            round((model_start - training_start).total_seconds() / training_dt_seconds)
        )
        if offset < 0:
            raise FlexConfigError(
                f"ARIMA surrogate model starts at {model_start.isoformat()} "
                f"which is before the training data start "
                f"{training_start.isoformat()}. The model must start at or "
                f"after the training start.",
                field="training_start_date",
                value=training_start_str,
            )
        if offset >= n_train:
            raise FlexConfigError(
                f"ARIMA surrogate model starts at training time {offset} "
                f"which is beyond the training data length ({n_train}). "
                f"The model must start before the end of the training data.",
                field="training_start_date",
                value=training_start_str,
            )

        if d == 0:
            init_values = [float(v) for v in self.data.get("init_values", [0.0] * p)]
        else:
            if offset == 0:
                # d=1 and model starts exactly at training start: the first
                # p+1 constraints are skipped in body(t) because the
                # differenced recursion needs y[-1]. The caller should fix
                # target[0]...target[p] to the first p+1 training values.
                init_values = [0.0]
            else:
                init_values = [float(training_y_values[offset - 1])]

        exog_vars = [
            unit.resolve_variable(name, field="input_variables") for name in exog_names
        ]
        exog_units = [declared_inputs[name][1] for name in exog_names]

        def _y_lag(t_idx: int, lag: int):
            """Return ``y[t - lag]`` from training data or model target.

            For d=0, returns the level. For d=1, returns the first difference
            ``y[t-lag] - y[t-lag-1]``. Always carries ``output_units`` (even
            when reading a raw training-data float), so every caller can
            divide the result by ``output_units`` to get a consistent,
            dimensionless number regardless of which branch fired.
            """
            training_idx = offset + t_idx - lag
            if training_idx < 0:
                if offset == 0:
                    if d == 1:
                        raise FlexConfigError(
                            f"ARIMA d=1 AR lag {lag} at model time {t_idx} maps "
                            f"to training index {training_idx}, before training "
                            f"start. For d=1 at offset==0, body(t) skips t <= p; "
                            f"this path should be unreachable.",
                            field="training_start_date",
                            value=training_start_str,
                        )
                    return init_values[p - lag + t_idx] * output_units
                raise FlexConfigError(
                    f"ARIMA AR lag {lag} at model time {t_idx} maps to "
                    f"training index {training_idx}, which is before the "
                    f"training start (offset={offset}). The model must "
                    f"start no earlier than training time 0.",
                    field="training_start_date",
                    value=training_start_str,
                )
            if training_idx >= offset:
                if d == 0:
                    return target[t_idx - lag]
                else:
                    # d=1: return difference y[t-lag] - y[t-lag-1]
                    prev_idx = t_idx - lag - 1
                    if prev_idx < 0:
                        # Need value from training data before model start
                        prev_training_idx = offset + prev_idx
                        if prev_training_idx < 0:
                            raise FlexConfigError(
                                f"ARIMA d=1 AR lag {lag} at model time "
                                f"{t_idx} requires training index "
                                f"{prev_training_idx}, before training start.",
                                field="training_start_date",
                                value=training_start_str,
                            )
                        return target[t_idx - lag] - (
                            float(training_y_values[prev_training_idx]) * output_units
                        )
                    return target[t_idx - lag] - target[prev_idx]
            if d == 0:
                return float(training_y_values[training_idx]) * output_units
            else:
                # d=1: return difference from training data
                if training_idx == 0:
                    return float(training_y_values[0]) * output_units
                return (
                    float(training_y_values[training_idx])
                    - float(training_y_values[training_idx - 1])
                ) * output_units

        def _resid_lag(t_idx: int, lag: int):
            """Return the fitted residual at training time ``t - lag``.

            Always carries ``output_units`` (even the zero fallback), for
            the same reason as :func:`_y_lag`.
            """
            training_idx = offset + t_idx - lag
            if 0 <= training_idx < n_resid:
                return float(residuals[training_idx]) * output_units
            return 0.0 * output_units

        def body(t):
            if d == 1 and offset == 0 and int(t) <= p:
                return pyo.Constraint.Skip
            ar_sum = sum(
                ar_coefs[j] * (_y_lag(t, j + 1) / output_units) for j in range(p)
            )
            ma_sum = sum(
                ma_coefs[j] * (_resid_lag(t, j + 1) / output_units) for j in range(q)
            )
            exog_sum = sum(
                exog_coefs[k]
                * (pyunits.convert(exog_vars[k][t], exog_units[k]) / exog_units[k])
                for k in range(len(exog_names))
            )
            drift_part = 0.0
            if drift is not None:
                # Use absolute training time index (1-based) so the surrogate
                # reproduces the trained in-sample path before optimization.
                drift_part = drift * (offset + int(t) + 1)
            if d == 0:
                return (const + drift_part + ar_sum + ma_sum + exog_sum) * output_units
            else:
                # d=1: y[t] = y[t-1] + c + drift*t + AR(differences) + MA + exog
                if int(t) == 0:
                    prev_y = init_values[0]
                else:
                    prev_y = target[t - 1] / output_units
                return (
                    prev_y + const + drift_part + ar_sum + ma_sum + exog_sum
                ) * output_units

        return body
