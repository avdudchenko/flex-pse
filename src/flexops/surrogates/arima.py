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
     "ar_coefs": [phi_1, ..., phi_p],
     "ma_coefs": [theta_1, ..., theta_q],
     "exog_coefs": [beta_1, ..., beta_k],
     "_residuals": [r_0, ..., r_{n-1}],   # fitted in-sample residuals
     "init_values": [y_{-p}, ..., y_{-1}]}   # optional; zeros when absent

The surrogate builds a Pyomo Constraint named
``"{target_name}_arima_eq"`` and stores fitted residuals as Params so
:meth:`~flexops.core.ops_block.OpsBlockData.swap_relation` can track the
added components across swaps.

Time indexing
~~~~~~~~~~~~~
The Pyomo ``TimeBlock`` uses integer indices ``0 … n-1``.  The ARIMA
equation at index ``t`` is::

    y[t] = c + sum(phi_j * y[t-j]) + sum(theta_j * resid[t-j])
           + sum(beta_k * x_k[t])

where:

* ``y[t-j]`` for ``t-j >= 0`` is the previous time step's target value;
  for ``t-j < 0`` it falls back to the ``init_values`` Params (``y0_0``
  … ``y0_{p-1}`` representing ``y[-p]`` … ``y[-1]``).
* ``resid[t-j]`` is the fitted residual at time ``t-j`` (0 for
  ``t-j < 0`` or ``t-j >= n``).  These are stored as Params
  ``resid_0`` … ``resid_{n-1}``.
* ``eta[t]`` is **not** introduced as a free Var; the surrogate is the
  *mean* ARIMAX relationship (innovations set to zero), which is the
  form useful for optimisation.  The fitted residuals are baked in as
  data so the in-sample path reproduces the statsforecast fitted values.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd
from pyomo.environ import Constraint, Param
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

_OPTIONAL_KEYS = ("seasonal_order", "init_values")
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

        exog = self.data["exogenous_variables"]
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

        p, _d, q = order
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
            if not isinstance(init, list) or len(init) != p:
                raise FlexConfigError(
                    f"ARIMA 'init_values' must be a list of {p} floats "
                    f"(matching AR order p={p}), got {init!r}.",
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
        """Return ``body(t)`` enforcing the ARIMA equation at time ``t``.

        Builds the following Pyomo components on ``unit``:

        * ``"{target_name}_arima_eq"`` – time-indexed ``Constraint``.
        * ``"{target_name}_arima_const"`` – ``Param`` for the constant term.
        * ``"{target_name}_arima_ar{j}"`` – one ``Param`` per AR coefficient.
        * ``"{target_name}_arima_ma{j}"`` – one ``Param`` per MA coefficient.
        * ``"{target_name}_arima_exog{j}"`` – one ``Param`` per exog coef.
        * ``"{target_name}_arima_resid{t}"`` – one ``Param`` per fitted
          residual (length ``n``, the number of in-sample rows).
        * ``"{target_name}_arima_y0{j}"`` – one ``Param`` per initial
          ``y`` value (length ``p``, the AR order).

        The constraint is built for every time index ``t``.  For ``t < p``
        the AR lag terms fall back to the ``y0`` Params; the MA terms use
        zero for negative residual indices.  No ``Constraint.Skip`` is
        returned: the equation is well-defined for every ``t`` once the
        ``init_values`` are supplied.

        Args:
            unit: The :class:`~flexops.core.ops_block.OpsBlockData` the
                surrogate is built on.
            target: The time-indexed ``Var``/``Reference`` the relationship
                determines.

        Returns:
            A callable ``body(t)`` returning a Pyomo expression for time
            index ``t``.
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
        p, _d, q = order
        const = float(self.data["const"])
        ar_coefs = [float(v) for v in self.data["ar_coefs"]]
        ma_coefs = [float(v) for v in self.data["ma_coefs"]]
        exog_coefs = [float(v) for v in self.data["exog_coefs"]]
        exog_names: list[str] = list(self.data["exogenous_variables"])
        init_values = [float(v) for v in self.data.get("init_values", [0.0] * p)]

        # Fitted residuals from the direct-fit model (length n_train).
        residuals: list[float] = self.data.get("_residuals", [0.0] * len(init_values))
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

        time_index = target.index_set()

        _prefix = f"{target.local_name}_arima"

        def _add_param(name_suffix: str, value: float, units=None) -> Param:
            full_name = f"{_prefix}_{name_suffix}"
            if unit.find_component(full_name) is not None:
                return unit.find_component(full_name)
            p_obj = Param(
                initialize=value,
                mutable=True,
                units=units,
                doc=f"ARIMA: {name_suffix}",
            )
            unit.add_component(full_name, p_obj)
            return p_obj

        const_param = _add_param("const", const, units=output_units)
        ar_params = [_add_param(f"ar{j}", ar_coefs[j]) for j in range(p)]
        ma_params = [_add_param(f"ma{j}", ma_coefs[j]) for j in range(q)]
        exog_params = [
            _add_param(f"exog{j}", exog_coefs[j]) for j in range(len(exog_names))
        ]
        y0_params = [
            _add_param(f"y0{j}", init_values[j], units=output_units) for j in range(p)
        ]
        resid_params = [
            _add_param(f"resid{t}", float(residuals[t]), units=output_units)
            for t in range(n_resid)
        ]

        exog_vars = [
            unit.resolve_variable(name, field="input_variables") for name in exog_names
        ]
        exog_units = [declared_inputs[name][1] for name in exog_names]

        def _y_lag(t_idx: int, lag: int):
            """Return ``y[t - lag]`` from training data or model target.

            The training data starts at ``offset`` steps before the model's
            t=0.  For lags that land inside the model window we return the
            model's own ``target`` Var (which is fixed to observed values
            for in-sample steps and computed for forecast steps).  For
            lags that land in the training period before the model window
            we return the stored training value as a float.
            """
            training_idx = offset + t_idx - lag
            if training_idx < 0:
                if offset == 0:
                    return y0_params[p - lag + t_idx]
                raise FlexConfigError(
                    f"ARIMA AR lag {lag} at model time {t_idx} maps to "
                    f"training index {training_idx}, which is before the "
                    f"training start (offset={offset}). The model must "
                    f"start no earlier than training time 0.",
                    field="training_start_date",
                    value=training_start_str,
                )
            if training_idx >= offset:
                return target[t_idx - lag]
            return float(training_y_values[training_idx])

        def _resid_lag(t_idx: int, lag: int):
            """Return the fitted residual at training time ``t - lag``."""
            training_idx = offset + t_idx - lag
            if 0 <= training_idx < n_resid:
                return resid_params[training_idx]
            return 0.0

        def body(t):
            ar_sum = sum(
                ar_params[j] * (_y_lag(t, j + 1) / output_units) for j in range(p)
            )
            ma_sum = sum(
                ma_params[j] * (_resid_lag(t, j + 1) / output_units) for j in range(q)
            )
            exog_sum = sum(
                exog_params[k]
                * (pyunits.convert(exog_vars[k][t], exog_units[k]) / exog_units[k])
                for k in range(len(exog_names))
            )
            return (
                const_param / output_units + ar_sum + ma_sum + exog_sum
            ) * output_units

        constraint_name = f"{_prefix}_eq"
        unit.add_component(
            constraint_name,
            Constraint(
                time_index,
                rule=lambda b, t_idx: target[t_idx] == body(t_idx),
                doc=(
                    f"ARIMA({order[0]},{order[1]},{order[2]}) surrogate: "
                    f"{target.local_name}[t] = c + AR + MA + exog."
                ),
            ),
        )
        return body
