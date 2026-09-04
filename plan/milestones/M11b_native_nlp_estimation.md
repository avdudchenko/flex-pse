# M11b — Native-constraint NLP parameter estimation

**Effort:** 3–4 days · **Depends on:** M10, M11 (loosely) · **Parallelizable:** with M12–M16

## Update note (pre-implementation revision)

This milestone was scoped by design/research passes rather than drafted
ahead of time (the M10b precedent — `plan/milestones/future/M10b_*.md` — is
the model for "write the spec for a capability not yet in the numbered
sequence"). A runtime spike (not part of this repo) already answered the one
real open question below — read this section before the rest.

- **`pyomo.contrib.parmest.parmest.Estimator(obj_function=None)` does NOT
  default to a sum-of-squared-errors objective.** It silently skips adding
  any objective at all — `_create_parmest_model`'s `if self.obj_function:`
  guard means `Total_Cost_Objective` is never created, and `theta_est()`
  fails with a confusing `AttributeError: 'BlockData' object has no
  attribute 'Total_Cost_Objective'`. `estimate_parameters` **must** pass
  `obj_function="SSE"` (or `pyomo.contrib.parmest.parmest.ObjectiveType.SSE`)
  explicitly when constructing the `Estimator`.
- **`cov_method="reduced_hessian"` is verified, not a risk.** A spike against
  a division-nonlinear model shaped exactly like `Pump`'s hydraulic law
  (`power == delta_pressure * flow / efficiency`), solved with this repo's
  pinned idaes/HSL IPOPT, recovered a known synthetic `efficiency` to within
  noise and produced covariance estimates from `cov_est(method=
  "reduced_hessian")` and `cov_est(method="finite_difference")` that agree to
  five significant figures. **Default `cov_method="reduced_hessian"`** with
  confidence; `"finite_difference"` remains available as an explicit
  alternative (costs `2*n_parameters + 1` extra ordinary-IPOPT solves instead
  of one barrier-restoration pass). **Do not support
  `"automatic_differentiation_kaug"`** — needs the external `k_aug` binary,
  not in this repo's toolchain (`flexcore/solvers/facade.py`'s solver
  priority list never mentions it); reject it explicitly with a clear
  message rather than letting it fail deep inside parmest.
- **`Pump.efficiency` is already a registered, regressable parameter** —
  `_build_hydraulic_relation` (`src/flexops/unit_models/pump.py`) already
  calls `self.register_process_parameter(self.efficiency, regressable=True)`.
  No `flexops` change is needed to make Pump's hydraulic law fittable by this
  milestone; the only reason it isn't fittable *today* is that
  `flexparameterize.apply._attach_surrogate` only ever writes back the one
  hard-coded `COEFFICIENT_NAME` ("energy_intensity"). This milestone does not
  touch `apply.py`'s existing behavior — it adds a separate, explicit entry
  point that can target *any* registered `regressable` parameter by name.

## Goal

Give `flexparameterize` a second, structurally distinct way to fit a unit's
process parameters: instead of fitting bare data columns out-of-model
(`flexparameterize.regression`'s `Regressor` Protocol — exact and cheap for
relationships *linear* in their own coefficients), fix a unit's known,
time-series inputs on its own **built, live model** and solve for an unknown
parameter by minimizing the residual between the model's own native
(possibly nonlinear-in-parameter) constraint and the measured output, via
`pyomo.contrib.parmest`. This is the only way to fit a relationship like
`Pump`'s hydraulic power law (`power == delta_pressure * flow / efficiency`)
at all — no existing `Regressor` can represent a division-nonlinear
coefficient — and it gets parameter covariance/standard errors for free
(`parmest` uses `PyNumero` internally for
`cov_est(method="reduced_hessian")`).

This is **not** a new `Regressor` and does **not** produce a `SurrogateSpec`.
It lives in a new sibling subpackage, `flexparameterize.estimation`, next to
(never replacing, never modifying) `flexparameterize.regression`.

## Read first

- `plan/01_architecture.md` §5 (`flexparameterize`/`flexops` coupling) and
  §2.3/§7 (R10/R11 — `SurrogateSpec` describes a *swappable* relation;
  `constant_intensity` is the existing precedent for "just fix a registered
  parameter, no surrogate class involved," which this milestone generalizes)
- `src/flexparameterize/regression/base.py` (the `Regressor` Protocol this
  milestone deliberately does **not** extend, and why: `fit(X, y)` is
  decoupled from any live Pyomo model — this capability fundamentally needs
  one)
- `src/flexparameterize/apply.py` — `_attach_surrogate` (the existing
  "fix a registered parameter" pattern this milestone's `commit_estimate`
  generalizes off its one hard-coded `COEFFICIENT_NAME`), `_basis_aliases`/
  `_registered_alias`/`model_alias` (the alias-resolution pattern
  `UnitExperiment` reuses), `_require_sufficient_data` (the sufficiency-gate
  *pattern* — note in Specification below why it does not transfer directly)
- `src/flexops/core/registration.py` — `ParameterRecord(param, name,
  regressable)`, `IOVariableRecord(var, role, units, time_indexed)`,
  `iter_io_registry`
- `src/flexops/core/time_block.py` — `TimeBlockData.time_index` (the plain
  integer `Set` every unit's time-indexed Vars are built over),
  `.index_of(timestamp)`/`.timestamp_of(i)` (the exact, already-implemented
  conversions between a `DatetimeIndex`-keyed data frame and the model's
  int-keyed `time_index` — do not invent a new one), `find_time_block(model)`
- `src/flexops/unit_models/pump.py` — `_build_hydraulic_relation` (the
  motivating nonlinear-in-parameter relation this milestone's flagship test
  targets)
- `src/flexops/testing/harness.py` — `_fix_registered_inputs`,
  `_solve_with_inputs_fixed` (the closest existing "fix inputs, solve,
  read back" pattern; also its `pytest.skip`-when-no-solver pattern, mirrored
  in this milestone's tests)
- `src/flexcore/solvers/facade.py` — `get_solver`, solver priority list (no
  `k_aug` — see the update note above)
- `src/flexparameterize/tests/helpers.py` — `build_plant`, `evaluate_data`
  (the exact fixture shape — a model containing only one unit — this
  milestone's `estimate_parameters` requires; see Pitfall 3)
- `plan/00_conventions.md` §3 (exceptions), §7 (markers, fixed seeds)

## Files to create or modify

- `src/flexparameterize/estimation/__init__.py` — re-exports `EstimationResult`,
  `UnitExperiment`, `estimate_parameters`, `commit_estimate` (new).
- `src/flexparameterize/estimation/result.py` — `EstimationResult` dataclass (new).
- `src/flexparameterize/estimation/experiment.py` — `UnitExperiment` (new).
- `src/flexparameterize/estimation/estimator.py` — `estimate_parameters`,
  `commit_estimate` (new).
- `src/flexparameterize/__init__.py` — add one more `from
  flexparameterize.estimation import (...)` re-export block, additive to
  the existing `flexparameterize.regression` one.
- `src/flexparameterize/regression/`, `apply.py`, `emit.py` — **untouched.**
  No file moves, no behavioral changes.
- `src/flexparameterize/tests/estimation/test_experiment.py`,
  `test_estimator.py`, `test_pump_hydraulic.py` (new; see Tests below).
- Docs: `docs/reference/flexparameterize/index.rst`,
  `docs/how_to/parameterize_from_data.md` (new §3b),
  `docs/explanation/config_schema.md` (extend the M11 extension-points
  paragraph with a third extension point).
- `CHANGELOG.md` — one entry under "Unreleased".

## Specification

### `EstimationResult` (`estimation/result.py`)

```python
@dataclass(frozen=True)
class EstimationResult:
    """The result of a native-constraint NLP parameter fit."""

    point_estimates: dict[str, float]
    std_errors: dict[str, float]           # {} if cov_method=None
    covariance: pd.DataFrame | None         # index/columns = parameter names
    objective_value: float                   # SSE at the optimum
    n_observations: int
    n_parameters: int
    data_window: tuple                        # (first, last) timestamp used
    solver_status: str

    def to_fit_result(self) -> FitResult:
        """Lossy down-cast to flexparameterize.regression.base.FitResult for
        provenance parity — coefficients=point_estimates,
        metrics={"objective_sse": objective_value} plus each std_errors
        entry folded in as "<name>_stderr", n_samples=n_observations,
        data_window=data_window. Drops the covariance matrix entirely
        (FitResult has no slot for one); callers who need it use
        `covariance` directly, not this method."""
```

### `estimate_parameters`/`commit_estimate` (`estimation/estimator.py`)

```python
def estimate_parameters(
    unit,
    data: pd.DataFrame,
    tagmap: TagMap,
    parameter_names: list[str],
    *,
    output_names: list[str] | None = None,
    measurement_error: dict[str, float] | None = None,
    cov_method: str | None = "reduced_hessian",
    solver: str = "ipopt",
) -> EstimationResult: ...

def commit_estimate(unit, result: EstimationResult) -> None: ...
```

- `parameter_names` is **required and explicit** — a list of names, each
  matched against `registry.parameters` by `ParameterRecord.name` (e.g.
  `"efficiency"` for `Pump`, `"energy_intensity"` for a constant-intensity
  unit). No auto-discovery: unlike `apply_to_model`'s "every unit with any
  regressable parameter," a nonlinear fit has no cheap zero-degree-of-freedom
  check that determines *which* parameters the data can identify — the
  caller states the hypothesis.
- **Gate strictly on `ParameterRecord.regressable is True`.** A name that
  isn't a registered, regressable parameter raises `FlexConfigError` naming
  what *is* registered — mirror `_attach_surrogate`'s existing message shape
  (`apply.py` lines ~218–227) verbatim in tone. Do **not** accept an
  arbitrary `unit.find_component(dotted_name)` path with no registry check —
  the registry is exactly how a unit author says "this Var is a
  design/regression coefficient, not structural state" (e.g.
  `Tank.initial_volume`/`Battery.capacity` are deliberately
  `regressable=False`); bypassing it would let a caller "estimate" a Var that
  was never meant to be fit.
- `output_names` defaults to every `registry.io_variables` entry with
  `role="output"` whose `model_alias` is present in `data`'s columns (after
  `tagmap.apply`) — mirror `check_sufficiency`'s IO-pair walk, for the
  residual side rather than the fixed side.
- **`estimate_parameters` is pure — it never mutates `unit`.** There is no
  a-priori correctness gate here analogous to `check_sufficiency`'s hard
  up-front raise (IPOPT can converge to a locally-optimal-but-physically-
  wrong point, e.g. an `efficiency` pinned near a bound); defaulting to
  mutate-in-place would risk silently corrupting a live model on a bad fit.
  Before solving, raise `FlexDataError` if the total observation count
  (summed across all `output_names` and all `t`, after dropping nulls) is
  `<= sum of len(parameter_names)` — the same spirit as `LinearRegressor`'s
  row-count guard (`regression/linear.py`), not a full identifiability check
  (there is no cheap one for a nonlinear fit; this is a floor, not a
  guarantee — see Pitfall 7).
- `commit_estimate(unit, result)` is the separate, explicit, opt-in
  write-back: `unit.update_parameters(result.point_estimates)` then `.fix()`
  each named Var — literally generalizing `_attach_surrogate`'s
  `constant_intensity` branch (`apply.py` lines ~229–233) off its single
  hard-coded name.
- Construct the `Estimator` as `Estimator([experiment], obj_function="SSE",
  solver_options=...)` — **never** `obj_function=None` (see update note).
- `solver="ipopt"` is passed through to `theta_est`/`cov_est`, not resolved
  via `flexcore.solvers.get_solver` — parmest's own solve calls take a bare
  solver name string, and this repo's idaes/HSL IPOPT is already the only
  NLP-capable option in `flexcore.solvers.facade`'s priority list, so there's
  nothing the facade's classification logic adds here. If no IPOPT is
  available at all, catch whatever `pyo.SolverFactory("ipopt").available()`
  reports and `pytest.skip`/raise `FlexSolverError` accordingly (mirror
  `flexops/testing/harness.py`'s skip pattern for the tests; production
  callers get a clear `FlexSolverError`).
- **v1 precondition, documented loudly in the public docstring:**
  `unit.model()` must contain only the target unit (its `TimeBlock`, its
  `property_package`, itself) — the exact shape
  `tests/helpers.py::build_plant()` builds. `UnitExperiment.get_labeled_model`
  returns `unit.model()` whole; multi-unit-embedded estimation is out of
  scope (see Pitfall 3).

### `UnitExperiment` (`estimation/experiment.py`)

Subclasses `pyomo.contrib.parmest.experiment.Experiment`. `get_labeled_model`:

1. For every `registry.io_variables` entry with `role="input"` whose
   `model_alias` is a column of the (tagmap-applied) data: for every `t` in
   `find_time_block(unit.model()).time_index`, `.fix()` that Var at
   `data.loc[tb.timestamp_of(t), alias]`. An input not present in the data is
   left however the unit was built — do not raise.
2. Do **not** call `.unfix()` on the target parameter(s) —
   `declare_process_parameter`/`register_process_parameter` always leave a
   registered parameter `.fix()`ed on creation, which is exactly the state
   parmest wants on entry (it unfixes internally via
   `utils.convert_params_to_vars`).
3. `model.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)`;
   populate with `{record.param: pyo.value(record.param) for name in
   parameter_names}` — seeds parmest's initial guess from however the unit
   was configured/built.
4. `model.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)` and
   `model.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)` (the
   latter is **required** by `cov_est` even with every value `None` — omit
   it and `cov_est` raises `AttributeError`; `None` lets parmest estimate the
   noise variance itself from the residual sum of squares). One entry per
   `(output alias, t)` pair present and non-null in the data; skip
   missing/NaN rows rather than raising.
5. `return unit.model()` (the whole model — see the v1 precondition above).

Reuse `apply.py`'s alias-resolution helpers (`model_alias`, the
`_registered_alias`/`_basis_aliases` pattern) rather than writing new lookup
logic.

## Pitfalls

1. **`obj_function=None` silently produces no objective at all.** Verified
   by spike (see update note) — always pass `obj_function="SSE"` explicitly
   when constructing the `Estimator`.
2. **`measurement_error` Suffix is mandatory for `cov_est`, even when every
   entry is `None`.** Omitting the Suffix entirely (not just leaving it
   empty) raises `AttributeError` deep inside parmest, not a flex-pse error —
   always create it in `get_labeled_model`.
3. **`get_labeled_model` returns `unit.model()` whole, not a clone of just
   `unit`.** This requires an isolated single-unit model as a v1
   precondition. Cloning just `unit`'s sub-block (for estimation against one
   unit embedded in a larger multi-unit facility) needs a Pyomo
   `Block.clone()` cross-block-`Set`-reference spike this milestone does not
   attempt — document the precondition in the public docstring; do not
   silently attempt it against a multi-unit model.
4. **Do not gate on a free `unit.find_component(dotted_name)` path.** Gate
   strictly on `registry.parameters`' `regressable=True` entries (see
   Specification) — this is the existing "may this be fit" decision unit
   authors already make, and bypassing it would fit Vars deliberately marked
   `regressable=False` (e.g. `Tank.initial_volume`).
5. **Do not default to mutating `unit`.** `estimate_parameters` is pure;
   `commit_estimate` is the separate, explicit write-back — see Specification
   for why (no a-priori correctness gate exists for a nonlinear fit the way
   `check_sufficiency` provides one for the linear/constant pipeline).
6. **`cov_method="automatic_differentiation_kaug"` is not supported.** It
   needs the external `k_aug` executable, which is not in this repo's
   toolchain (`flexcore/solvers/facade.py`'s priority list never mentions
   it) — reject that value explicitly with a clear `FlexConfigError` rather
   than letting parmest fail looking for a missing binary.
7. **No sufficiency guarantee, only a floor.** Unlike
   `check_sufficiency`'s zero-degree-of-freedom guarantee for the
   linear/constant pipeline, there is no cheap pre-solve check that the data
   actually *identifies* `parameter_names` (e.g. data where `delta_pressure`
   barely varies makes `efficiency` numerically unidentifiable from
   `power_eq` alone even with thousands of rows). The
   `n_observations <= n_parameters` guard (Specification) catches the
   degenerate case only; `EstimationResult.std_errors` is the real signal,
   known only *after* paying for the solve. State this limitation in the
   public docstring; do not claim a stronger guarantee than the code
   provides.
8. **Time-index conversion.** Use `TimeBlockData.timestamp_of(i)`/
   `.index_of(timestamp)` (already implemented) for all `Datetime <->
   int` conversions between `data`'s index and `time_index` — do not invent
   a new mapping utility.
9. **Reusing `_attach_surrogate`'s message tone, not its code path.**
   `commit_estimate` generalizes the *pattern* (`update_parameters` +
   `.fix()`), not the function itself — `_attach_surrogate` stays entirely
   inside `apply.py`, untouched, still hard-coded to `COEFFICIENT_NAME` for
   its own (`constant_intensity`) purpose.
10. **Determinism.** Every synthetic-noise test uses
    `numpy.random.default_rng(<fixed seed>)` (conventions §7) — no
    unseeded randomness.

## Tests

`src/flexparameterize/tests/estimation/test_experiment.py` (all `unit`, no
solve):
- `test_get_labeled_model_fixes_known_inputs`
- `test_get_labeled_model_leaves_target_parameter_fixed_but_labeled`
- `test_get_labeled_model_populates_experiment_outputs_per_timepoint`
- `test_get_labeled_model_always_sets_measurement_error_suffix`
- `test_unknown_parameter_name_raises` (`FlexConfigError`)
- `test_missing_measured_output_at_a_timepoint_is_skipped_not_errored`

`src/flexparameterize/tests/estimation/test_estimator.py` (`component`,
`needs_ipopt`):
- `test_estimate_parameters_returns_point_estimate_and_covariance` — refit
  `energy_intensity` on `build_plant()`'s fixture (deliberately linear, so
  ground truth is unambiguous) via the NLP path; compare against
  `helpers.INTENSITY`; assert `covariance` is square/symmetric/finite.
- `test_estimate_parameters_does_not_mutate_the_unit`
- `test_commit_estimate_writes_back_and_refixes`
- `test_cov_method_none_skips_covariance`
- `test_automatic_differentiation_kaug_rejected` (`FlexConfigError`)
- `test_insufficient_solver_skips` — mirror
  `flexops/testing/harness.py`'s `pytest.skip` pattern when no NLP solver.

`src/flexparameterize/tests/estimation/test_pump_hydraulic.py` (`component`,
`needs_ipopt` — the motivating case):
- `test_recovers_efficiency_from_noise_free_hydraulic_data` — direct-evaluate
  `power = delta_pressure * flow / true_efficiency` (no solver, like
  `helpers.evaluate_data`), fit, assert tight-tolerance recovery.
- `test_recovers_efficiency_with_seeded_noise_and_finite_covariance` —
  `numpy.random.default_rng(42)` additive noise; assert the estimate is
  within a few `std_errors` of the truth; assert
  `numpy.linalg.eigvalsh(result.covariance) > 0`.
- `test_cov_method_reduced_hessian_vs_finite_difference_agree` — std errors
  within ~20% of each other on the same noisy data.
- `test_efficiency_out_of_bounds_data_reports_low_confidence_not_a_crash` —
  degenerate/unidentifiable data (e.g. constant zero `delta_pressure`):
  either raises `FlexDataError`/`FlexSolverError` or returns a very large
  `std_errors["efficiency"]`; pin down whichever the implementation does.

## Documentation tasks

- `docs/reference/flexparameterize/index.rst`: new section for
  `estimation.result.EstimationResult`,
  `estimation.estimator.estimate_parameters`/`commit_estimate`,
  cross-referenced against the existing "Regression" and "Applying a fit to
  a live model" sections.
- `docs/how_to/parameterize_from_data.md`: new "§3b. When the model's own
  constraint is already nonlinear in its parameters" subsection, right after
  §3/§4a, contrasting `estimate_parameters`/`commit_estimate` with
  `LinearRegressor`.
- `docs/explanation/config_schema.md`: extend the M11 extension-points
  paragraph — not every parameter fit produces a `SurrogateSpec`; a
  native-constraint NLP fit only fixes an existing
  `register_process_parameter` Var in place, a *third* extension point
  alongside "new Surrogate class" and "new Regressor."
- CHANGELOG "Unreleased": one entry, noting this is **not** wired into
  `apply_to_model`'s automatic per-unit fitting path, and is a sibling
  capability to `regression/`, not a replacement or extension of it.

## Definition of Done

- [ ] `EstimationResult`, `UnitExperiment`, `estimate_parameters`,
      `commit_estimate` exist with the signatures above, in a new
      `flexparameterize.estimation` package; `flexparameterize.regression`
      is untouched (no file moves, no behavioral change, all M10/M11 tests
      still green unmodified).
- [ ] `estimate_parameters` gates strictly on `ParameterRecord.regressable`;
      an unregistered/non-regressable name raises `FlexConfigError` naming
      what is registered.
- [ ] `estimate_parameters` never mutates `unit`; `commit_estimate` is the
      only write-back path, verified by a dedicated purity test.
- [ ] `Estimator` is always constructed with `obj_function="SSE"` (never
      relying on a default); `cov_method="automatic_differentiation_kaug"`
      is rejected explicitly.
- [ ] The Pump-hydraulic recovery tests pass: a known synthetic `efficiency`
      is recovered within tolerance, `cov_method="reduced_hessian"` and
      `"finite_difference"` agree, and the covariance is positive-definite.
- [ ] Every sklearn/solver-dependent test only skips (never fails) when its
      dependency is absent, per the existing `needs_ipopt` marker pattern.
- [ ] `pyproject.toml` needs **no** new dependency (parmest ships inside the
      pinned `pyomo`); confirm no `.importlinter` contract changes.
- [ ] Reference/how-to/explanation docs updated; `sphinx-build -W` passes;
      CHANGELOG updated.
- [ ] plus the generic DoD in `CLAUDE.md`
