# M11 — Regressor protocol + linear regression

**Effort:** 1.5 days · **Depends on:** M10 · **Parallelizable:** with M12

## Update note (pre-implementation revision)

This file is being started fresh against the M10 code actually on `main`, which
went further than the M10 milestone spec in two ways that change what is left
to build here. Read this section before the rest — it supersedes wording
below wherever the two disagree, and M10b is out of scope entirely (dropped
from the roadmap; do not read it for context).

- **`SurrogateSpec` is `surrogate_type: SurrogateType` + an opaque `data`
  mapping, not `functional_form`/bare `coefficients`/`input_variables`/
  `output_variables` fields.** `SurrogateType` (`flexcore.config.schema`) has
  members `constant_intensity`, `multilinear`, `quadratic`, `exponential`,
  `arima`, `neural_network`. There is no `"linear"` type — `multilinear`
  (`flexops.surrogates.multilinear.MultilinearSurrogate`) is a constant plus
  each input's own coefficient plus (optionally) named cross terms, and is
  what this milestone's linear regressor must emit. `quadratic`/`exponential`/
  `arima`/`neural_network` already exist as **stub classes in
  `flexops.surrogates`** that raise `NotImplementedError` at construction —
  this milestone's registry should name the same reserved set, not the
  `"nn"`/`"multiconvex"` placeholders an earlier draft of this file used.
- **The swap machinery this milestone's tests were going to exercise already
  exists and is already tested**, built out during M10's own follow-up work:
  `OpsBlockData.swap_relation`, `flexops.surrogates.surrogate_from_spec`,
  `apply_to_model`'s `surrogates=` argument (a hand-built spec attached as a
  unit's own energy relation, or a `{relation_name: spec}` mapping naming any
  other relation the unit registered via `register_relation`, e.g. a reverse
  osmosis skid's `split_definition`), and the one-fit/two-consumers agreement
  between `apply_to_model` and `emit_model_config`. See
  `src/flexparameterize/tests/test_apply.py`
  (`test_apply_swaps_energy_relation_to_multilinear`,
  `test_apply_swaps_a_named_relation`) — all of it exercised today with a
  **hand-built** `SurrogateSpec`, never a fitted one. **This milestone does not
  rebuild or retest that machinery.** Its job is narrower: give the pipeline a
  second regressor that can *produce* a `multilinear` `SurrogateSpec` from
  data, formalize the shared result shape both regressors already return by
  hand, and a name→class registry. The round-trip/apply tests below are
  intentionally thin re-runs of the existing hand-built-spec tests with a
  fitted spec substituted in — proving the fitted path reaches the same
  swap code, not re-proving the swap.
- Multi-output regression (fitting several output columns in one call) is
  **out of scope** — it was aimed at the M10b multi-dimensional-surrogate work,
  which is not being built. `MultilinearSurrogate` itself only accepts a
  single `output_variables` entry ("a registered relation has one target");
  `LinearRegressor` follows `ConstantIntensityRegressor`'s existing one-output
  shape. Fitting two outputs for one unit (e.g. power and chemical dose) is
  two separate fits, each producing its own spec, attached to two separate
  registered relations via `apply_to_model(..., surrogates={unit: {relation_a:
  spec_a, relation_b: spec_b}})` — already possible with no change here.
- `apply_to_model` does **not** gain a way to pick a regressor per unit for its
  automatic fit-from-data path; it still fits every unit's own energy relation
  with `ConstantIntensityRegressor`, exactly as M10 built it. `get_regressor`
  and `LinearRegressor` are for a caller who fits explicitly (as
  `docs/how_to/parameterize_from_data.md` §3 already shows for
  `ConstantIntensityRegressor`) and hands the result to `emit_model_config`,
  or builds the `SurrogateSpec` and hands it to `apply_to_model(...,
  surrogates=...)`. Wiring automatic regressor *selection* into the
  fit-from-data path is not this milestone's problem and is not implied by
  "config-driven selection is `get_regressor(spec.surrogate_type)`" below —
  that phrase describes a config file naming which regressor *produced* a
  spec (provenance/tooling), not `apply_to_model` choosing one at runtime.
- `fit(...)` keeps returning the fitted regressor itself (`Self`), not a bare
  `FitResult` — every existing call site
  (`flexparameterize/apply.py::_fit_unit`, `flexparameterize/emit.py`,
  `flexparameterize/tests/helpers.py::fit_intensity`, the how-to doc) chains
  `Cls().fit(X, y)` and then calls `.to_surrogate_spec(...)` on the same
  object, or reads `.coefficient`/`.metrics`/`.n_samples`/`.data_window`
  straight off it. Changing `fit`'s return type would break all of those for
  no behavioral gain. `FitResult` is instead a small dataclass every
  conforming regressor can *produce* from its own already-fitted attributes
  via a new `to_fit_result()` method — the "shared result shape" the original
  sketch wanted, without an incompatible return-type change.

## Goal

Formalize the pluggable-regressor seam: a runtime-checkable `Regressor`
Protocol with a shared `FitResult` dataclass, a sklearn-backed
`LinearRegressor` behind the `[parameterize]` extra, and a small
`SurrogateType`-keyed registry for config-driven regressor lookup. After this
milestone, adding a quadratic/ARIMA/neural-network regressor later is
"implement the protocol, register the name" — no pipeline changes.

A fitted regressor's `SurrogateSpec` feeds **both** consumers of the two-way
pipeline (architecture §5, R10/R11): `emit_model_config` (which turns it into
a config that rebuilds the model) and `apply_to_model` (which uses the same
spec to swap a unit's relation Constraint in place on a live model, via
`flexops.surrogates.surrogate_from_spec` — already built). There is one
`SurrogateSpec` per fit and (at least) two consumers of it — the linear
regressor's spec must be equally consumable by both, exactly as a hand-built
one already is.

## Read first

- `plan/01_architecture.md` §5 (`flexparameterize`/`flexops` two-way coupling;
  the same `SurrogateSpec` feeds both `emit_model_config` and
  `apply_to_model` — R10, one fit, two consumers)
- `plan/01_architecture.md` §2.3 (R3: `SurrogateSpec` = `surrogate_type` +
  opaque `data`; `provenance`)
- `src/flexcore/config/schema.py` (`SurrogateType`, `SurrogateSpec` — the
  actual current shape; read this instead of trusting field names in older
  milestone prose, this file's own M10 included)
- `src/flexops/surrogates/` (`Surrogate` ABC, `SURROGATES` registry,
  `surrogate_from_spec`; `multilinear.py` for the exact `data` contract
  `LinearRegressor.to_surrogate_spec()` must produce; `quadratic.py` for the
  shape of a reserved stub)
- `src/flexparameterize/regression/constant.py` (the pattern to follow:
  attributes stored on the instance after `fit`, `to_surrogate_spec` reading
  them, `FlexDataError` if called unfit)
- `src/flexparameterize/apply.py` and `emit.py` (both consumers; `emit.py`'s
  `_surrogate_and_fit_provenance` is the one place that reads a fitted
  regressor's provenance fields — the mechanical hook for `to_fit_result()`)
- `src/flexparameterize/tests/test_apply.py` (`test_apply_swaps_energy_relation_to_multilinear`,
  `test_apply_swaps_a_named_relation`) and `tests/helpers.py` (`build_plant`,
  `evaluate_data`) — the existing fixtures and hand-built-spec tests this
  milestone's tests reuse and lightly extend
- `plan/00_conventions.md` §3 (exceptions: `FlexConfigError`/`FlexDataError`
  with actionable messages), §7 (markers, determinism, fixed seeds)
- `plan/02_testing_and_ci.md` §1 (solver/availability skip pattern — mirror it
  for the optional sklearn dependency)

## Files to create or modify

- `src/flexparameterize/regression/base.py` — `Regressor` Protocol + `FitResult` dataclass (new).
- `src/flexparameterize/regression/constant.py` — add `to_fit_result()` to `ConstantIntensityRegressor`; no other change.
- `src/flexparameterize/regression/linear.py` — `LinearRegressor` (new; sklearn, optional extra).
- `src/flexparameterize/regression/__init__.py` — add `get_regressor(name)` + re-exports.
- `src/flexparameterize/emit.py` — `_surrogate_and_fit_provenance` reads `fit_result.to_fit_result()` instead of the four attributes directly (mechanical).
- `src/flexparameterize/__init__.py` — re-export `Regressor`, `FitResult`, `LinearRegressor`, `get_regressor`.
- `pyproject.toml` — add `scikit-learn` to the existing `[parameterize]` extra (it currently lists only `pandas>=2.0`, which is now redundant with the core dependency — leave it, don't clean it up here).
- `src/flexparameterize/tests/regression/test_base.py`, `test_linear.py`, `test_registry.py`, `test_linear_roundtrip.py` (new directory alongside the existing flat `src/flexparameterize/tests/test_*.py` files; `regression/constant.py` has no dedicated test file today — its behavior is covered indirectly through `test_roundtrip.py`/`test_apply.py`/`test_emit.py`, which must stay green).
- Docs: `docs/how_to/parameterize_from_data.md` §3 (replace the "completed in
  M11" placeholder), `docs/reference/flexparameterize/index.rst`, a short
  extension-points note.

## Specification

### FitResult and Protocol (`src/flexparameterize/regression/base.py`)

```python
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from flexcore.config.schema import SurrogateSpec


@dataclass(frozen=True)
class FitResult:
    """The common shape every regressor's fit reduces to, for provenance."""

    coefficients: dict[str, float]      # coefficient name -> value ("intercept" when present)
    metrics: dict[str, float]           # at minimum: "r2", "rmse"
    n_samples: int
    data_window: tuple                  # (first, last) index value of the fitted rows


@runtime_checkable
class Regressor(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "Regressor": ...
    def to_fit_result(self) -> FitResult: ...
    def to_surrogate_spec(self, **kwargs) -> SurrogateSpec: ...
```

- `typing.Protocol` + `@runtime_checkable`. No ABC, no base-class inheritance
  requirement — conformance is structural (Pitfall 2 below on what that does
  and doesn't check).
- `fit` returns the fitted regressor itself (`Self`, written as `"Regressor"`
  for the Protocol) so `Cls().fit(X, y).to_surrogate_spec(...)` keeps working
  everywhere it already does. It stores its fit on instance attributes,
  exactly as `ConstantIntensityRegressor` does today; calling `to_fit_result()`
  or `to_surrogate_spec()` before `fit` raises `FlexDataError` telling the
  user to fit first (mirror `ConstantIntensityRegressor`'s existing check).
- `to_surrogate_spec` takes `**kwargs` because the two regressors need
  different unit metadata at that boundary (`ConstantIntensityRegressor`
  already takes `input_units`/`output_units` as single strings;
  `LinearRegressor`, having several input columns, needs a per-column units
  mapping — see below). The Protocol does not pin the keyword names; each
  regressor documents its own.
- `data_window` tuple element type follows the index (timestamps for a
  `DatetimeIndex`, as every existing regressor and consumer assumes);
  `emit_model_config` is what serializes it into ISO-8601 strings for
  provenance (already true today — `emit.py`'s `_surrogate_and_fit_provenance`
  does the `.isoformat()` conversion, not the regressor).

### `ConstantIntensityRegressor.to_fit_result()` (`regression/constant.py`)

Purely additive — do not change `fit`'s signature, `coefficient`'s name, the
zero-flow guard, or the mean-ratio fit rule:

```python
def to_fit_result(self) -> FitResult:
    if self.coefficient is None:
        raise FlexDataError(...)  # same message/style as to_surrogate_spec's guard
    return FitResult(
        coefficients={COEFFICIENT_NAME: self.coefficient},
        metrics=self.metrics,
        n_samples=self.n_samples,
        data_window=self.data_window,
    )
```

Then simplify `emit.py::_surrogate_and_fit_provenance` to call
`fit_result.to_fit_result()` once and read the four fields off the returned
`FitResult` instead of reaching into the regressor's own attribute names —
this is what makes the refactor "mechanical" for a second regressor with
differently-named attributes.

### LinearRegressor (`src/flexparameterize/regression/linear.py`)

Ordinary least squares via scikit-learn, one output column (see the update
note above on why), any number of input columns, available only with the
`[parameterize]` extra:

- Import guard: import `sklearn.linear_model` lazily inside `__init__` or
  `fit` (implementer's choice) so `import flexparameterize.regression` and
  `import flexparameterize` succeed with no sklearn installed (Pitfall 1).
  On absence, raise:
  ```python
  raise FlexConfigError(
      "LinearRegressor requires scikit-learn. "
      "Install it with `pip install 'flex-pse[parameterize]'`."
  )
  ```
- `fit(X, y)`: `X` is one or more input columns, `y` is exactly one output
  column (reuse `ConstantIntensityRegressor`'s private `_single_column`
  helper for `y`, or an equivalent — do not fork the "insufficient shape"
  error wording). Drop rows with any NaN across `X`/`y` before fitting; if
  fewer rows remain than `X.shape[1] + 1`, raise `FlexDataError` saying how
  many rows survived and the minimum needed. Fit
  `sklearn.linear_model.LinearRegression(fit_intercept=True)`. Store on the
  instance: `coefficients` (`{column_name: coef, "intercept": intercept}`),
  `metrics` (`{"r2": ..., "rmse": ...}`, sklearn `.score()` and RMSE of the
  fitted line against `y`), `n_samples` (post-drop row count), `data_window`
  (`(index.min(), index.max())` of the surviving rows), `input_variables`
  (list of `X`'s column names, in order), `output_variable` (`y`'s column
  name). This is the same attribute shape as `ConstantIntensityRegressor`,
  just with a dict of coefficients instead of one and a list of inputs instead
  of one.
- `to_fit_result()` — mechanical, same pattern as `ConstantIntensityRegressor`'s.
- `to_surrogate_spec(self, *, input_units: dict[str, str], output_units: str) -> SurrogateSpec`
  returns `SurrogateType.MULTILINEAR` with
  `data={"input_variables": {name: input_units[name] for name in self.input_variables},
  "output_variables": {self.output_variable: output_units}, "coefficients": self.coefficients}`
  — exactly `MultilinearSurrogate`'s validated contract (`flexops/surrogates/multilinear.py`).
  `input_units` must carry an entry for every fitted input column; a missing
  one raises `FlexConfigError` naming it (do not let a `KeyError` escape).
  No cross terms are ever synthesized — a plain OLS fit produces one
  coefficient per input column plus an intercept, which is a valid (degree-1)
  multilinear relationship; a caller wanting a cross term engineers it into
  an `X` column named `"a*b"` before fitting (mention this in the how-to, do
  not build automatic feature engineering here).
- `FitResult.metrics`/`SurrogateSpec.provenance`: identical shape to
  `ConstantIntensityRegressor`'s — flat `{"r2": ..., "rmse": ...}`, since
  there is exactly one output.

### Registry (`src/flexparameterize/regression/__init__.py`)

```python
def get_regressor(name: str) -> type: ...
```

- Keys are `SurrogateType` values, not a separate ad hoc string space —
  `SurrogateType.CONSTANT_INTENSITY.value` (`"constant_intensity"`) →
  `ConstantIntensityRegressor`, `SurrogateType.MULTILINEAR.value`
  (`"multilinear"`) → `LinearRegressor`. Accept either a `SurrogateType`
  member or its string value. Config-driven selection is then literally
  `get_regressor(spec.surrogate_type)`.
- **Reserved names**, matching the reserved stub classes already in
  `flexops.surrogates` — `SurrogateType.QUADRATIC`, `.EXPONENTIAL`, `.ARIMA`,
  `.NEURAL_NETWORK` — raise `NotImplementedError` with a message naming the
  corresponding stub class (e.g. `"...; see flexops.surrogates.QuadraticSurrogate,
  not yet implemented."`), so the two "not yet built" registries (surrogate
  classes and regressors) stay in the same namespace and fail the same way.
- Unknown names raise `FlexConfigError` listing the valid and reserved names
  (iterate `SurrogateType` for both lists rather than hand-maintaining them,
  so a new enum member cannot silently drift out of sync).
- Keep the registry a plain module-level dict (implementer's choice; no
  entry-points plugin machinery in v0).
- Re-export `Regressor`, `FitResult`, `LinearRegressor`, and `get_regressor`
  from `flexparameterize.regression` and from the top-level `flexparameterize`
  package (alongside the existing `ConstantIntensityRegressor` re-export).
  Importing either package must NOT import sklearn (Pitfall 1).

## Pitfalls

1. **Registry import pulls in sklearn.** `get_regressor("multilinear")` may
   import sklearn lazily, but `import flexparameterize` and `import
   flexparameterize.regression` must succeed without sklearn installed — guard
   inside `LinearRegressor.__init__`/`fit`, never at module import time.
2. **`runtime_checkable` checks names, not signatures.** `isinstance(obj,
   Regressor)` only verifies the methods exist. The conformance test must
   also *call* `fit`/`to_fit_result`/`to_surrogate_spec` and check the return
   types.
3. **Monkeypatching the sklearn-absent path.** Simulate absence with
   `monkeypatch.setitem(sys.modules, "sklearn", None)` (plus the submodule
   keys actually imported, e.g. `sklearn.linear_model`) rather than
   uninstalling anything.
4. **Skips leaking red into the extra-less env.** Every test that genuinely
   imports sklearn carries `pytest.importorskip("sklearn")` (module level) so
   a checkout without the extra passes cleanly (skip, not fail) — mirror the
   solver-availability pattern from `02_testing_and_ci.md` §1.
5. **Unseeded noise.** The synthetic-recovery test uses
   `numpy.random.default_rng(42)` (or any fixed seed) — conventions §7
   forbids nondeterminism.
6. **Reintroducing multi-output.** `MultilinearSurrogate` enforces exactly one
   `output_variables` entry; do not widen `LinearRegressor` past one `y`
   column to "support" it — that is out of scope (see the update note above).
7. **Breaking existing call sites by changing `fit`'s return type.**
   `apply.py::_fit_unit`, `emit.py`, and `tests/helpers.py::fit_intensity` all
   chain `Cls().fit(X, y)` expecting the fitted instance back. `fit` keeps
   returning `Self`; `FitResult` is produced by the separate `to_fit_result()`
   method.
8. **Coefficient key drift between `LinearRegressor` and `MultilinearSurrogate`.**
   The surrogate validates that every non-`"intercept"` coefficient key is
   (a product of) names in `input_variables`; `LinearRegressor` must use the
   fitted column names verbatim as both the `input_variables` keys and the
   (single-factor) coefficient keys, or the emitted/applied spec fails
   `MultilinearSurrogate._validate` at build/apply time instead of at fit time.
9. **Provenance key collisions.** `emit_model_config` merges `to_fit_result()`'s
   `n_samples`/metrics/`data_window` with its own `"versions"` and any
   caller-supplied `provenance` (already implemented, `emit.py`); a new
   regressor must not invent a metrics key that collides with those
   (`"versions"` is reserved).
10. **Re-testing the swap machinery instead of the new regressor.** The
    round-trip/apply tests below should be minimal deltas on the existing
    hand-built-spec tests (swap in a fitted spec for the hand-built one,
    assert equal numeric behavior) — do not duplicate
    `test_apply_swaps_energy_relation_to_multilinear`'s already-covered
    assertions about deactivation/ports/arcs.

## Tests

`src/flexparameterize/tests/regression/test_base.py`
- `test_protocol_conformance` (`unit`) — `isinstance(ConstantIntensityRegressor(), Regressor)`
  is True; a class missing `to_surrogate_spec` is False; then fit
  `ConstantIntensityRegressor` on a tiny frame and assert `to_fit_result()`
  returns a `FitResult` with the right fields (see Pitfall 2 — call, don't
  just `isinstance`).
- `test_to_fit_result_before_fit_raises` (`unit`) — `FlexDataError`.

`src/flexparameterize/tests/regression/test_linear.py` (module-level
`pytest.importorskip("sklearn")`, except the absence test)
- `test_recovers_coefficients_noisy_pump` (`unit`) — synthetic pump data:
  `power = 0.4 * flow + 2.0 + noise`, 200 rows, `default_rng(42)`, noise sigma
  0.01; fitted coefficient within `pytest.approx(0.4, rel=1e-2)` and intercept
  within `pytest.approx(2.0, abs=0.05)`.
- `test_two_input_columns` (`unit`) — synthetic data with two input columns
  (e.g. flow and pressure, both linear terms, no cross term); both
  coefficients recovered; `to_surrogate_spec` requires `input_units` for both
  or raises `FlexConfigError` naming the missing one.
- `test_isinstance_regressor` (`unit`) — `isinstance(LinearRegressor(),
  Regressor)` is True (Pitfall 2: also call `fit`/`to_fit_result`).
- `test_provenance_populated` (`unit`) — `to_fit_result()`'s metrics contain
  `r2`/`rmse`; values finite; the resulting `SurrogateSpec.provenance` (as
  `emit_model_config` would assemble it) is JSON-serializable.
- `test_sklearn_absent_raises` (`unit`, NO skipif) — monkeypatch sklearn out
  of `sys.modules`; instantiating/fitting `LinearRegressor` raises
  `FlexConfigError` whose message contains `flex-pse[parameterize]`.

`src/flexparameterize/tests/regression/test_registry.py` (all `unit`)
- `test_get_regressor_known_names` — both `SurrogateType.CONSTANT_INTENSITY`
  and `SurrogateType.MULTILINEAR` (as members and as their string values)
  return the right classes.
- `test_reserved_names_raise_notimplemented` — `SurrogateType.QUADRATIC`,
  `.EXPONENTIAL`, `.ARIMA`, `.NEURAL_NETWORK` → `NotImplementedError`.
- `test_unknown_name_raises` — `FlexConfigError` listing valid + reserved names.
- `test_package_import_without_sklearn` — with sklearn monkeypatched absent,
  `importlib.reload(flexparameterize.regression)` (and
  `importlib.reload(flexparameterize)`) succeed.

`src/flexparameterize/tests/regression/test_linear_roundtrip.py` (skipif
sklearn absent; reuse `flexparameterize.tests.helpers.build_plant`/
`evaluate_data` rather than rebuilding fixtures)
- `test_linear_fit_emit_rebuild_predictions` (`component`) — build a plant
  with `has_pressure=True` (as `test_apply_swaps_energy_relation_to_multilinear`
  does), compute a *linear* (no cross term) synthetic power draw from flow and
  outlet pressure by direct evaluation, fit it with `LinearRegressor`, emit via
  `emit_model_config`, round-trip through `dump_model_config`/
  `load_model_config`, rebuild, and assert the rebuilt relation's constraint
  body matches the regressor's own predictions at 5 probe points
  (`pytest.approx(rel=1e-6)`). No solver needed (constraint-body evaluation,
  as the existing M10 tests already do).
- `test_linear_surrogate_spec_applies_in_place` (`component`) — the same
  fitted spec through `apply_to_model(m, data, tagmap, surrogates={unit.name:
  spec})` instead of emit; assert the swapped-in
  `power_electrical_relation_fitted` Constraint (see
  `test_apply_swaps_energy_relation_to_multilinear` for the exact assertion
  shape) reproduces the same predictions at the same 5 probe points. This is
  the one-fit/two-consumers check (R10) for a *fitted* spec — the swap
  mechanics themselves are already covered by M10's tests and are not
  re-asserted here beyond "it swapped and the numbers match."

## Documentation tasks

- `docs/how_to/parameterize_from_data.md` §3 ("Choosing a regressor"): replace
  the "completed in M11" placeholder with the real section — `get_regressor`,
  `LinearRegressor` next to `ConstantIntensityRegressor`, the `[parameterize]`
  extra and its install line (`pip install 'flex-pse[parameterize]'`), and a
  short note that a `SurrogateSpec` from either regressor works identically in
  §4a/§4b below it (no changes needed to those sections — they are already
  regressor-agnostic).
- `docs/reference/flexparameterize/index.rst`: add a `regression.base`
  subsection (`Regressor`, `FitResult`) and extend the existing "Regression"
  section with `flexparameterize.regression.linear.LinearRegressor` and
  `flexparameterize.regression.get_regressor`.
- A short extension-points note: the Regressor Protocol + reserved
  `SurrogateType` names are the hook for a future quadratic/ARIMA/neural-
  network regressor — one or two sentences is enough; the smallest home is
  probably alongside the existing config-schema explanation page rather than
  a new file (implementer's choice).
- `docs/getting_started/installation.md`: mention the `[parameterize]` extra
  now installs scikit-learn too.
- CHANGELOG entry under "Unreleased".

## Definition of Done

- [ ] `Regressor` Protocol (`runtime_checkable`) + `FitResult` dataclass exist
      with the signatures above; `fit` still returns the fitted regressor
      itself everywhere (no call site broken).
- [ ] `ConstantIntensityRegressor.to_fit_result()` added with no other change
      to its behavior; all of M10's existing round-trip/apply/emit tests stay
      green unmodified.
- [ ] `LinearRegressor` fits one output against one or more input columns,
      recovers seeded synthetic coefficients within tolerance, and its
      `SurrogateSpec` validates against `MultilinearSurrogate`.
- [ ] sklearn-absent path raises `FlexConfigError` with the install
      instruction; `flexparameterize`/`flexparameterize.regression` import
      cleanly with no sklearn installed.
- [ ] `get_regressor` resolves `constant_intensity`/`multilinear` (member or
      string value); the four reserved `SurrogateType` names raise
      `NotImplementedError`; unknown names raise `FlexConfigError`.
- [ ] All sklearn-dependent tests skip (not fail) when the extra is absent.
- [ ] The fitted-`LinearRegressor` round-trip and apply-in-place tests
      reproduce the regressor's own predictions at 5 probe points
      (`rel=1e-6`), confirming the existing swap machinery accepts a fitted
      spec exactly as it already accepts a hand-built one.
- [ ] `pyproject.toml`'s `[parameterize]` extra includes scikit-learn; how-to
      §3 completed; reference docs build with `sphinx-build -W`; CHANGELOG
      updated.
- [ ] plus the generic DoD in CLAUDE.md
