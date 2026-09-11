# Surrogate Coefficients as First-Class Model Variables

## Goal

Refactor surrogate building so that surrogate coefficients become first-class
Pyomo `Var` objects on a dedicated sub-block, rather than Python floats baked
into a constraint expression. This lets `flexparameterize` unfix the
coefficients for a regression solve, then fix them at their fitted values — and
lets the surrogate block be cleanly deactivated when rebuilding the unit model.

## Current State (problem)

`MultilinearSurrogate.build(unit, target)` returns a `body(t)` callable that
produces a Pyomo expression. The expression embeds the fitted coefficients as
Python `float` literals baked in at build time. Once baked in, the coefficients
cannot be changed without calling `swap_relation` again (which deactivates the
old constraint and builds a new one). This has two consequences:

1. **No coefficient-level DOF for regression.** The regression pipeline cannot
   unfix individual coefficients, solve for them, and re-fix — it must instead
   rebuild the whole constraint from a new `SurrogateSpec` each time.
2. **No clean deactivation of the surrogate layer.** When `flexparameterize`
   rebuilds the unit model (e.g. after a parameter change), there is no single
   block to deactivate; the surrogate expression is fused into the constraint
   rule on the unit itself.

## Desired State

```
unit
├── power_electrical[t]              (performance Var, kW)
├── power_electrical_relation        (deactivated original Constraint)
└── surrogate_power                  (Block added by swap_relation)
    ├── coefficients                 (CoefficientRegistry: intercept, flow_out, …)
    ├── body[t]                      (Expression: intercept + Σ coef * product(inputs))
    └── fitted[t]                    (active Constraint on the block:
                                      target[t] == body[t])
```

Key structural change from current: **the fitted constraint lives on the
surrogate block, not on the unit**. Deactivating `surrogate_power` therefore
deactivates everything — coefficients, body, and fitted constraint — in one
step. `swap_relation` does not add a fitted constraint to the unit; it adds it
to the surrogate block.

The `surrogate_power` block is the single unit to deactivate/reactivate. The
coefficient Vars inside it are regressable: call
`unfix_surrogate_coefficients` before fitting, then
`fix_surrogate_coefficients` after.

Block naming: base name is `surrogate_{relation_base}` (e.g. `surrogate_power`
for `power_electrical_relation`). Only if a block with that name already exists
on the unit (from a prior swap) is a unique suffix appended — `swap_relation`
tracks this.

## Coefficient storage

`block.coefficients` is a **`CoefficientRegistry`** instance, created automatically
when the surrogate block is constructed. The registry exposes:

- `register_coefficient(name, var)` — add a single named coefficient Var.
- `register_coefficients(mapping)` — bulk-add from a dict.
- `items()` / `__getitem__()` / `__contains__()` — dict-like access to the
  stored Vars by name.

The registry **dynamically expands** as coefficients are registered, so a
`build()` method can call `register_coefficient` multiple times — useful when
coefficients come from several independent Vars or Var groups (e.g. separate
low-flow and high-flow regimes).

```python
# In a surrogate's build() method:
block = pyo.Block(concrete=True)
registry = block.coefficients  # CoefficientRegistry instance

registry.register_coefficient("intercept", pyo.Var(initialize=5.0))
registry.register_coefficients({
    name: pyo.Var(initialize=float(value))
    for name, value in self.data["coefficients"].items()
    if name != "intercept"
})
# Later, for a piecewise surrogate:
registry.register_coefficient("breakpoint_1", pyo.Var(initialize=10.0))
```

The body expression references Vars through the registry:

```python
def body(t):
    total = block.coefficients["intercept"]
    for key, _ in self.data["coefficients"].items():
        if key == "intercept":
            continue
        term = block.coefficients[key]
        ...
    return total
```

`register_surrogate_coefficients` on the unit iterates the registry's items
directly. No separate normalization step is needed because the registry is
already `dict[str, pyo.Var]`-shaped.

## Proposed API Changes

### `Surrogate.build()` signature change

**Current:**

```python
def build(self, unit, target):
    ...
    return body  # callable: t -> expression
```

**Proposed:**

```python
def build(self, unit, target):
    ...
    return block, body  # block: Pyomo Block (or None), body: callable t -> expression
```

`build()` constructs the block and its coefficient Vars using
`block.coefficients.register_coefficient(...)`, but does **not** add the block to the unit.
`swap_relation` owns all naming and attachment.

Returning `None` for `block` is allowed for surrogates that do not need
auxiliary Vars (e.g. constant intensity, which is not a `Surrogate` class at
all).

### `swap_relation()` signature change

**Current:**

```python
def swap_relation(self, relation_name: str, surrogate: Surrogate) -> None:
```

**Proposed:**

```python
def swap_relation(self, relation_name: str, surrogate: Surrogate) -> Block | None:
```

Returns the surrogate sub-block (or `None`) so callers can hold a reference to
the coefficient Vars without re-walking the component map.

### `RelationRecord` extension

Add a `surrogate_blocks` history list and a `surrogate_block` field:

```python
@dataclass
class RelationRecord:
    ...
    surrogate_block: Any = None        # currently active block; None if no surrogate swap
    surrogate_blocks: list = field(default_factory=list)  # every block ever built for this relation, oldest first
```

`swap_relation` appends each new block to `surrogate_blocks` before making it
active. This gives the record a complete history without the caller having to
track block names.

### `MultilinearSurrogate.build()` change

`build()` constructs an un-added `pyo.Block(concrete=True)` with:

- `block.coefficients` — a `CoefficientRegistry` instance, populated by
  successive `register_coefficient` calls. For the multilinear case this
  yields entries keyed by coefficient name (`"intercept"`, `"flow_out"`,
  `"flow_out*pressure"`, …), each a scalar Var initialized from the
  spec data.
- `block.body[t]` — a time-indexed `pyo.Expression` whose rule evaluates the
  multilinear sum using `block.coefficients[name]` for each term.
- No fitted Constraint is created here; `swap_relation` adds that.

Returns `(block, body)` where `body` is a callable `t -> pyoExpression` that
evaluates `block.body[t]`.

```python
def build(self, unit, target):
    output_units = parse_units(next(iter(self.output_variables.values())))
    declared = { ... }  # resolved input Vars with their declared units

    block = pyo.Block(concrete=True)
    coefficients = CoefficientRegistry()

    coefficients.register_coefficient(
        "intercept", pyo.Var(initialize=float(self.data["coefficients"]["intercept"]))
    )
    for key, value in self.data["coefficients"].items():
        if key == "intercept":
            continue
        coefficients.register_coefficient(key, pyo.Var(initialize=float(value)))

block.coefficients.register_coefficients({
        name: pyo.Var(initialize=float(value))
        for name, value in self.data["coefficients"].items()
        if name != "intercept"
    })

    block.coefficients = coefficients

    for name, var in coefficients.items():
        var.fix()

    def body(t):
        total = block.coefficients["intercept"]
        for key, _ in self.data["coefficients"].items():
            if key == "intercept":
                continue
            term = block.coefficients[key]
            for name in key.split("*"):
                var, units = declared[name]
                converted = pyunits.convert(var[t], units)
                term = term * (converted / units)
            total = total + term
        return total * output_units

    return block, body
```

Key design points:

- Coefficients are fixed at build time to their initialized values, so the
  model is well-defined immediately after `swap_relation`. If the spec carries
  no coefficient for a given name, the Var is initialized to `1.0` and fixed.
  Callers that need to solve for coefficients (regression, round-trip tests)
  call `unfix_surrogate_coefficients()` before solving and
  `fix_surrogate_coefficients()` after.
- The block is returned un-added; `swap_relation` adds it to the unit.

## Multiple surrogates and coefficient switching

A unit can have multiple registered relations (e.g. `power_electrical_relation`,
`split_definition`, `level_definition`), each independently swappable. Each
relation carries its own `RelationRecord` with its own `surrogate_block`. The
system handles three scenarios:

### Scenario A — Multiple surrogates active simultaneously

Each relation has its own block:

```
unit
├── power_electrical_relation        (deactivated original)
├── surrogate_power                  (active block for power relation)
│   ├── coefficients                 (intercept, flow, …)
│   └── fitted[t]
└── surrogate_split                  (active block for split relation)
    ├── coefficients                 (fraction, …)
    └── fitted[t]
```

`register_surrogate_coefficients` is called per-relation. Each call registers
that relation's coefficients as regressable parameters and removes *only that
relation's* performance target from the parameters list. The unit's registry
ends up containing coefficients from both surrogates, each keyed by its unique
name.

The regression pipeline fits **all** registered parameters together — there is
no per-relation scoping. If a developer needs to fit one surrogate but not the
other, they control that at the unit level by selectively calling
`register_surrogate_coefficients` before fitting.

### Scenario B — Switching surrogates for the same relation

Calling `swap_relation` again for the same relation deactivates the old block
(and everything in it, including its fitted constraint) and builds a new one:

```
unit
├── surrogate_power      (deactivated, from first swap)
│   ├── coefficients     (old: intercept, flow)
│   └── fitted[t]       (inactive)
└── surrogate_power_1    (active, from second swap)
    ├── coefficients     (new: intercept, flow, flow*pressure)
    └── fitted[t]       (active)
```

`register_surrogate_coefficients` for the second swap:
- Removes stale parameter records for the first block's coefficients
- Registers the new block's coefficients
- Removes the performance target (already removed the first time, but the
  operation is idempotent)

The regression pipeline then fits the new set of coefficients. The old block's
coefficients are no longer in the registry and are not fitted.

### Scenario C — Piecewise / multi-regime surrogate on a single block

A single surrogate block can hold coefficients for multiple operating regimes
using one `CoefficientRegistry`:

```
unit
└── surrogate_power
    ├── coefficients
    │   ├── intercept_low
    │   ├── slope_low
    │   ├── intercept_high
    │   └── slope_high
    ├── body[t]          (selects low/high based on flow[t])
    └── fitted[t]
```

All four coefficients are registered together via one
`register_surrogate_coefficients` call. The regression pipeline fits all four
simultaneously. The `body(t)` closure selects which coefficients to use at each
time point based on the current input values.

### Design implications

- **No per-relation coefficient scoping in the regression pipeline.** The
  pipeline fits whatever is registered. If a unit has two surrogates and only
  one should be fitted, the caller controls this by not calling
  `register_surrogate_coefficients` for the unwanted one.
- **`register_surrogate_coefficients` is idempotent per relation.** Calling it
  twice for the same relation replaces the old coefficient records with the new
  ones. The old block's Vars remain on the model (deactivated) but are no
  longer in the registry, so they are not fitted.
- **Block deactivation is automatic on the next swap.** Because each block is
  tracked in its `RelationRecord.components`, the next `swap_relation` call
  deactivates the entire old block — no manual cleanup needed.

### Mechanism for explicit switching without rebuilding

If a developer wants to switch between two pre-built surrogate blocks without
calling `swap_relation` again (e.g. toggling between a day and night model,
or reactivating a previously deactivated surrogate to refit it), three helper
methods provide this:

**`list_surrogate_blocks(relation_name)`** — returns the local names of every
surrogate block ever built for this relation, oldest first.

**`switch_surrogate_block(block_name)`**

```python
def switch_surrogate_block(self, block_name: str) -> None:
    """Switch to a previously built surrogate block by name.

    Deactivates whatever is currently active for ``relation_name``, activates
    the named block, updates ``record.surrogate_block`` to point to it, and
    re-registers its coefficients as regressable parameters.

    Args:
        relation_name: The relation whose surrogate history to switch.
        block_name: The local name of the surrogate block to activate
            (e.g. ``"surrogate_power"`` or ``"surrogate_power_1"``).

    Raises:
        FlexConfigError: If ``relation_name`` is not registered, or
            ``block_name`` is not found in the relation's block history.
    """
```

**`current_surrogate_block(relation_name)`** — returns the local name of the
currently active surrogate block, or `None` if no surrogate is active.

```python
def current_surrogate_block(self, relation_name: str) -> str | None:
    """Return the local name of the currently active surrogate block, or None."""
```

With these, the reactivation flow is:

```python
# Step 1: swap twice — builds surrogate_power, then surrogate_power_1
unit.swap_relation("power_electrical_relation", multilinear_spec_1)
unit.swap_relation("power_electrical_relation", multilinear_spec_2)

# Step 2: list available blocks and switch to the one you want
blocks = unit.list_surrogate_blocks("power_electrical_relation")
# → ["surrogate_power", "surrogate_power_1"]
unit.switch_surrogate_block("surrogate_power")

# Step 3: refit as normal — registry now points at surrogate_power's coefficients
unit.register_surrogate_coefficients("power_electrical_relation")
# ... run regression, fix coefficients ...
```

Or via `apply_to_model`:

```python
apply_to_model(
    model, data, tagmap,
    active_surrogates={
        "facility.pump": "surrogate_power",  # activate this block before fitting
    },
)
```

This is an advanced use case; the common path is `swap_relation`, which handles
all steps automatically. The explicit methods exist for workflows that need to
toggle between pre-built surrogates without rebuilding them.

## Implementation Steps

### Step 1 — Extend `RelationRecord` with `surrogate_block`

**File:** `src/flexops/core/registration.py`

Add `surrogate_block: Any = None` to `RelationRecord`. Update the docstring.

**Risk:** Low — additive field with a default, no existing code touches it.

### Step 2 — Add `CoefficientRegistry` and attach it to surrogate blocks

**File:** `src/flexops/core/ops_block.py`

```python
class CoefficientRegistry:
    """A dict-like container for a surrogate block's coefficient Vars.

    The registry dynamically expands as coefficients are registered. It is
    attached to every surrogate block as ``block.coefficients`` before
    ``build()`` returns, so a developer can call ``register_coefficient``
    multiple times in a single ``build()`` method — useful when coefficients
    come from several independent Var groups (e.g. separate low-flow and
    high-flow regimes).

    Attributes:
        _vars: The internal dict mapping coefficient name -> pyo.Var.
    """

    def __init__(self) -> None:
        self._vars: dict[str, pyo.Var] = {}

    def register_coefficient(self, name: str, var: pyo.Var) -> None:
        """Add a single named coefficient Var.

        Args:
            name: The coefficient name used by ``body(t)`` and
                ``register_surrogate_coefficients``.
            var: The Pyomo Var carrying this coefficient's value.

        Raises:
            FlexConfigError: If ``name`` is already registered or ``var``
                is not a ``pyo.Var``.
        """
        if name in self._vars:
            raise FlexConfigError(
                f"Coefficient {name!r} is already registered on this block."
            )
        if not isinstance(var, pyo.Var):
            raise FlexConfigError(
                f"Coefficient {name!r} must be a pyo.Var, got "
                f"{type(var).__name__}."
            )
        self._vars[name] = var

    def register_coefficients(self, mapping: dict[str, pyo.Var]) -> None:
        """Bulk-add coefficients from a name->Var mapping.

        Args:
            mapping: Dict of coefficient name to Pyomo Var.

        Raises:
            FlexConfigError: If any name is already registered or any value
                is not a ``pyo.Var``.
        """
        for name, var in mapping.items():
            self.register_coefficient(name, var)

    def items(self):
        """Return the registered (name, Var) pairs."""
        return self._vars.items()

    def __getitem__(self, name: str) -> pyo.Var:
        return self._vars[name]

    def __contains__(self, name: str) -> bool:
        return name in self._vars

    def __iter__(self):
        return iter(self._vars)

    def __len__(self) -> int:
        return len(self._vars)
```

Every surrogate block is given a `CoefficientRegistry` at construction time,
before `build()` returns. The block's `coefficients` attribute is the registry
itself — not a raw Var or dict. `swap_relation` and
`register_surrogate_coefficients` both know how to work with it.

### Step 3 — Change `Surrogate.build()` return type

**File:** `src/flexops/surrogates/base.py`

Update the abstractmethod signature:

```python
@abstractmethod
def build(self, unit, target) -> tuple[Block | None, Callable]:
    """Return (block, body(t)).

    block: a Pyomo Block the surrogate constructs, or None if this surrogate
        needs no auxiliary Vars. ``swap_relation`` adds the block to ``unit``
        itself; this method must not touch ``unit.add_component``.
    body: a callable taking a time index and returning a units-carrying
        Pyomo expression in this relationship's declared output units.
    """
```

Update every concrete surrogate class:

**Files:**
- `src/flexops/surrogates/multilinear.py`
- `src/flexops/surrogates/quadratic.py`
- `src/flexops/surrogates/exponential.py`
- `src/flexops/surrogates/arima.py`
- `src/flexops/surrogates/neural_network.py`

For `MultilinearSurrogate.build()`:
- Create a `pyo.Block(concrete=True)` (do **not** add it to `unit`).
- Call `block.coefficients.register_coefficient(...)` /
  `block.coefficients.register_coefficients({...})` to populate the registry.
   with scalar Vars initialized from the spec data.
- The body uses the registered coefficient Vars.
- Return `(block, body)`.

For all other surrogate stubs (`QuadraticSurrogate`, etc.) that currently raise
`NotImplementedError`, return `(None, body)` using their existing float-based
body until they are implemented — the `block` is optional.

### Step 4 — Refactor `swap_relation()` to own block attachment and fitted-constraint placement

**File:** `src/flexops/core/ops_block.py`

```python
def swap_relation(self, relation_name: str, surrogate: Surrogate) -> Block | None:
    ...
    record = ...  # find RelationRecord
    ...

    # Deactivate old
    (record.fitted or record.constraint).deactivate()
    for component in record.components:
        getattr(component, "deactivate", lambda: None)()

    # Build surrogate block + body (build() does NOT add the block to unit)
    surrogate_block, body = surrogate.build(self, record.target)

    # Determine unique block name: base is "surrogate_{base}", uniquify only
    # if a block with that name already exists on this unit.
    base_name = f"surrogate_{relation_name.replace('_relation', '')}"
    block_name = base_name
    counter = 1
    while self.find_component(block_name) is not None:
        block_name = f"{base_name}_{counter}"
        counter += 1

    if surrogate_block is not None:
        self.add_component(block_name, surrogate_block)
        record.surrogate_blocks.append(surrogate_block)
        record.surrogate_block = surrogate_block
        block_component = surrogate_block
    else:
        block_component = None
        record.surrogate_block = None

    # Build the fitted constraint ON the surrogate block, not on the unit.
    record.swap_count += 1
    fitted_name = f"fitted" if record.swap_count == 1 else f"fitted_{record.swap_count}"

    target = record.target

    def _rule(b, t, _body=body, _target=target):
        expr = _body(t)
        if expr is pyo.Constraint.Skip:
            return pyo.Constraint.Skip
        try:
            converted = pyunits.convert(expr, pyunits.get_units(_target[t]))
        except UnitsError as exc:
            raise FlexConfigError(
                f"{type(surrogate).__name__}'s output units are "
                f"incompatible with {_target.local_name!r}'s "
                f"({pyunits.get_units(_target[t])!s}).",
                field="output_variables",
            ) from exc
        return _target[t] == converted

    if surrogate_block is not None:
        # Constraint lives on the surrogate block
        surrogate_block.add_component(
            fitted_name,
            pyo.Constraint(
                target.index_set(),
                rule=_rule,
                doc=f"Fitted relationship ({type(surrogate).__name__}), "
                f"replacing the deactivated {relation_name}. The surrogate's "
                f"own output units are converted into {target.local_name!r}'s.",
            ),
        )
        record.fitted = surrogate_block.find_component(fitted_name)
    else:
        # No surrogate block: constraint goes on the unit (legacy path)
        suffix = "" if record.swap_count == 1 else f"_{record.swap_count}"
        fitted_name_full = f"{relation_name}_fitted{suffix}"
        self.add_component(
            fitted_name_full,
            pyo.Constraint(
                target.index_set(),
                rule=_rule,
                doc=f"Fitted relationship ({type(surrogate).__name__}), "
                f"replacing the deactivated {relation_name}.",
            ),
        )
        record.fitted = self.find_component(fitted_name_full)

    new_components = [c for c in [
        block_component,
        record.fitted,
    ] if c is not None]
    record.components = new_components

    assert record.fitted is not None, "Fitted constraint was not added."
    return record.surrogate_block
```

**Why the fitted constraint belongs on the surrogate block:**

`swap_relation` already deactivates everything in `record.components` on the
next swap. By putting the fitted constraint inside the surrogate block and
tracking the block in `record.components`, a single `block.deactivate()` call
turns off the entire surrogate layer — coefficients, body expression, and
fitted constraint together. This is the single deactivation unit
`flexparameterize` needs when rebuilding a unit.

**Naming convention:**

- Block base name: `surrogate_{relation_name_without_suffix}` — e.g.
  `surrogate_power` for `power_electrical_relation`.
- Uniquify only on collision: if `surrogate_power` already exists, try
  `surrogate_power_1`, `surrogate_power_2`, etc. This keeps names readable
  when there is only one swap and only grows when there are many.
- Fitted constraint name on the block: `fitted` (first swap) or
  `fitted_2`, `fitted_3`, … (subsequent swaps). No leading underscore: the
  fitted constraint is part of the block's public surface.

### Step 5 — Add `register_surrogate_coefficients()`

**File:** `src/flexops/core/ops_block.py`

```python
def register_surrogate_coefficients(
    self,
    relation_name: str,
    *,
    remove_target: bool = True,
) -> list[str]:
    """Register a swapped surrogate's coefficient Vars as regressable parameters.

    After :meth:`swap_relation` attaches a surrogate to ``relation_name``, call
    this to (a) register each coefficient Var in the unit's ``IORegistry`` as a
    regressable process parameter, and (b) remove the relation's performance
    target from the parameter list so it is not regressed against alongside the
    coefficients.

    The surrogate block's ``coefficients`` attribute is a
    :class:`CoefficientRegistry` whose ``items()`` yields ``(name, Var)``
    pairs. Any object with an ``items()`` method returning ``(str, pyo.Var)``
    pairs is accepted.

    Args:
        relation_name: The relation whose surrogate coefficients to register.
        remove_target: If True, remove ``record.target`` from the parameters
            list (it is a performance output, not a regressable input).

    Returns:
        The local names of the registered coefficient parameters.
    """
    record = next(
        (r for r in self._io_registry.relations if r.name == relation_name), None
    )
    if record is None:
        raise FlexConfigError(
            f"{relation_name!r} is not a registered relation on "
            f"{self.name!r}.",
            field="relation_name",
            value=relation_name,
        )
    if record.surrogate_block is None:
        raise FlexConfigError(
            f"{relation_name!r} on {self.name!r} has no surrogate block; "
            "call swap_relation first.",
            field="relation_name",
            value=relation_name,
        )

    block = record.surrogate_block
    coefficients = getattr(block, "coefficients", None)
    if coefficients is None or not hasattr(coefficients, "items"):
        raise FlexConfigError(
            f"Surrogate block on {relation_name!r} has no 'coefficients' "
            f"registry (expected a CoefficientRegistry or dict-like).",
            field="relation_name",
            value=relation_name,
        )

    # Replace any stale parameter records for these names, then register
    names = {name for name, _ in coefficients.items()}
    self._io_registry.parameters = [
        p for p in self._io_registry.parameters
        if p.name not in names
    ]

    registered = []
    for name, var in coefficients.items():
        self.register_process_parameter(name, var, regressable=True)
        registered.append(name)

    if remove_target:
        target = record.target
        self._io_registry.parameters = [
            p for p in self._io_registry.parameters
            if id(p.param) != id(target)
        ]

    return registered
```

### Step 6 — Update `flexparameterize.apply._attach_surrogate()`

**File:** `src/flexparameterize/apply.py`

After `unit.swap_relation(...)` returns the surrogate block, call
`unit.register_surrogate_coefficients(relation_name)` so the coefficients are
in the registry and the performance target is removed, then fix the coefficient
Vars at their fitted values.

```python
if surrogate.surrogate_type is not SurrogateType.CONSTANT_INTENSITY:
    surrogate_block = unit.swap_relation(
        POWER_ELECTRICAL_RELATION, surrogate_from_spec(surrogate)
    )
    if surrogate_block is not None:
        unit.register_surrogate_coefficients(POWER_ELECTRICAL_RELATION)
        for coef_name, coef_value in surrogate.data["coefficients"].items():
            var = surrogate_block.coefficients[coef_name]
            var.set_value(coef_value)
            var.fix()
    return True, dict(surrogate.data["coefficients"])
```

The `ApplyReport` should record the fixed coefficients for richer surrogates.

### Step 7 — Update regression pipeline to use coefficient Vars

**Files:**
- `src/flexparameterize/regression/linear.py` (and any other regressor)
- `src/flexparameterize/apply.py`

The regression pipeline already produces a `SurrogateSpec` with fitted
coefficients. The change is purely in application:

- For `CONSTANT_INTENSITY`: fix the intensity Var (unchanged).
- For richer surrogates: `swap_relation` → `register_surrogate_coefficients` →
  fix each coefficient Var at its fitted value.

No new solve step is needed for multilinear surrogates because
`LinearRegressor` computes OLS coefficients analytically. For nonlinear
surrogates (future milestone), a solve step would be inserted between
`register_surrogate_coefficients` and fixing.

### Step 8 — Block deactivation and reactivation support

`swap_relation` already deactivates everything in `record.components` on the
next swap. Because the fitted constraint now lives inside the surrogate block
(and the block is tracked in `record.components`), deactivating the block
automatically deactivates the fitted constraint — no extra code needed.

For explicit deactivation (e.g. when rebuilding a unit), the caller simply
deactivates the block:

```python
unit.find_component("surrogate_power").deactivate()
```

For reactivation of a previously deactivated surrogate (e.g. toggling back to
a day model after fitting a night model), three convenience methods are added:

**`list_surrogate_blocks(relation_name)`** — returns the local names of every
surrogate block ever built for this relation, oldest first.

**`switch_surrogate_block(block_name)`**

```python
def switch_surrogate_block(self, block_name: str) -> None:
    """Switch to a previously built surrogate block by its local name.

    Looks up the block on this unit by name, finds the ``RelationRecord`` it
    belongs to, deactivates whatever is currently active for that relation,
    activates the named block, updates ``record.surrogate_block`` to point to
    it, and re-registers its coefficients as regressable parameters.

    Args:
        block_name: The local name of the surrogate block to activate
            (e.g. ``"surrogate_power"`` or ``"surrogate_power_1"``).

    Raises:
        FlexConfigError: If ``block_name`` is not found on this unit, or is
            not a surrogate block.
    """
    block = self.find_component(block_name)
    if block is None:
        raise FlexConfigError(
            f"Block {block_name!r} not found on {self.name!r}.",
            field="block_name",
            value=block_name,
        )

    record = next(
        (r for r in self._io_registry.relations if block in r.surrogate_blocks),
        None,
    )
    if record is None:
        raise FlexConfigError(
            f"Block {block_name!r} is not a surrogate block on "
            f"{self.name!r}.",
            field="block_name",
            value=block_name,
        )

    # Deactivate whatever is currently active
    (record.fitted or record.constraint).deactivate()
    for component in record.components:
        getattr(component, "deactivate", lambda: None)()

    # Activate the target block
    block.activate()

    # Update record to point at the reactivated block
    record.surrogate_block = block
    record.components = [c for c in [block, record.fitted] if c is not None]

    # Re-register this block's coefficients (replaces any stale records)
    self.register_surrogate_coefficients(record.name)
```

The underlying source of truth is `record.surrogate_blocks` — a list of the
actual Pyomo block objects, oldest first. `switch_surrogate_block` takes only
the block's local name, finds the owning record by scanning the history, and
performs the switch. The relation name is not needed from the caller.

**`current_surrogate_block(relation_name)`** — returns the local name of the
currently active surrogate block for the given relation, or `None`.

```python
def current_surrogate_block(self, relation_name: str) -> str | None:
    """Return the local name of the currently active surrogate block, or None."""
    record = next(
        (r for r in self._io_registry.relations if r.name == relation_name), None
    )
    if record is None or record.surrogate_block is None:
        return None
    return record.surrogate_block.local_name
```

**`unfix_surrogate_coefficients(relation_name)`** — unfix every coefficient
Var in the currently active surrogate block for the given relation, so a
solver can adjust them.

```python
def unfix_surrogate_coefficients(self, relation_name: str) -> None:
    """Unfix all coefficient Vars in the active surrogate block.

    After calling, the coefficients are free variables again and a solve
    can adjust them. Call :meth:`fix_surrogate_coefficients` to re-fix
    them at their current values after the solve.

    Raises:
        FlexConfigError: If ``relation_name`` is not registered or has
            no active surrogate block.
    """
```

**`fix_surrogate_coefficients(relation_name)`** — fix every coefficient Var
in the currently active surrogate block at its current value, dropping the
model's degrees of freedom.

```python
def fix_surrogate_coefficients(self, relation_name: str) -> None:
    """Fix all coefficient Vars in the active surrogate block at their current values.

    This is the counterpart to :meth:`unfix_surrogate_coefficients`. After
    solving, call this to lock the coefficients at their solved values.
    """
```

Typical regression workflow:

```python
unit.unfix_surrogate_coefficients("power_electrical_relation")
solver.solve(unit)
unit.fix_surrogate_coefficients("power_electrical_relation")
```

**Why `block_name` instead of `index`:**

The underlying source of truth is `record.surrogate_blocks` — a list of the
actual Pyomo block objects on the record, oldest first. The named wrapper
exists because `apply_to_model` and similar external callers only have string
keys, not live records. For callers that do have the record, the blocks are
already there — no lookup needed:

```python
record = next(r for r in unit._io_registry.relations if r.name == relation_name)
# record.surrogate_blocks[0] is the actual first block object
```

The name-based API is a convenience for the string-keyed `apply_to_model`
path; it is not the only way to access the blocks.

**Common call patterns:**

```python
# Step 1: see what's available
blocks = unit.list_surrogate_blocks("power_electrical_relation")
# → ["surrogate_power", "surrogate_power_1"]

# Step 2: switch to the one you want
unit.switch_surrogate_block("surrogate_power")

# Step 3: refit as normal — registry now points at surrogate_power's coefficients
unit.register_surrogate_coefficients("power_electrical_relation")
# ... run regression, fix coefficients ...
```

### Step 9 — Update `flexparameterize.apply` to support reactivation

**File:** `src/flexparameterize/apply.py`

`apply_to_model` should accept an optional `active_surrogates` mapping that
names which pre-built surrogate block to activate per unit, before fitting:

```python
def apply_to_model(
    model,
    data: pd.DataFrame,
    tagmap: TagMap,
    surrogates: dict | None = None,
    *,
    active_surrogates: dict[str, str] | None = None,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> ApplyReport:
```

`active_surrogates` maps `unit_name -> block_name`. For any unit listed,
`apply_to_model` calls
`unit.switch_surrogate_block(block_name)` before fitting,
instead of calling `swap_relation`. This lets a workflow:

1. Build two surrogate blocks for the same relation (two `swap_relation` calls)
2. List them: `unit.list_surrogate_blocks("power_electrical_relation")`
3. Fit the first one with `apply_to_model(active_surrogates={"facility.pump":
   "surrogate_power"})`
4. Later, fit the other with `apply_to_model(active_surrogates={"facility.pump":
   "surrogate_power_1"})`

Without this, the user would have to manually call
`unit.switch_surrogate_block(...)` before `apply_to_model`.

## Files Modified

| File | Change |
|------|--------|
| `src/flexops/core/registration.py` | Add `surrogate_block: Any = None` to `RelationRecord` |
| `src/flexops/core/ops_block.py` | Add `CoefficientRegistry` class; update `swap_relation`; add `register_surrogate_coefficients`, `switch_surrogate_block`, `current_surrogate_block`, `unfix_surrogate_coefficients`, `fix_surrogate_coefficients` |
| `src/flexops/surrogates/base.py` | Update `build()` abstractmethod return type |
| `src/flexops/surrogates/multilinear.py` | Return `(block, body)` using `CoefficientRegistry` |
| `src/flexops/surrogates/quadratic.py` | Return `(None, body)` stub |
| `src/flexops/surrogates/exponential.py` | Return `(None, body)` stub |
| `src/flexops/surrogates/arima.py` | Return `(None, body)` stub |
| `src/flexops/surrogates/neural_network.py` | Return `(None, body)` stub |
| `src/flexparameterize/apply.py` | Add `active_surrogates` parameter; call `switch_surrogate_block` when provided; call `register_surrogate_coefficients`; fix coefficient Vars at fitted values |
| `src/flexparameterize/regression/linear.py` | No structural change; coefficients now applied to Vars |
| `src/flexparameterize/regression/base.py` | No change (FitResult already carries coefficients) |

## Pitfalls and Risks

### Pitfall 1 — Block naming uniqueness

The base name `surrogate_{relation_base}` must be uniquified only when a block
with that name already exists on the unit. `swap_relation` handles this with a
simple counter loop. This keeps names readable (single swap → `surrogate_power`)
and only appends `_1`, `_2`, … when needed.

### Pitfall 2 — Coefficient Var units and scaling

The intercept coefficient currently lives in the surrogate's declared output
units. Input-factor coefficients are unitless in the current `build()` (they
are divided by the factor's declared units inside `body(t)`). When the
coefficients become Vars, their Pyomo units must match what the regression
produces. Otherwise unit conversion in the expression body will double-count
or divide incorrectly.

**Decision:** Coefficient Vars carry no Pyomo units (dimensionless). The
expression body handles unit conversion exactly as today. The regression fit
must produce coefficients in the same basis the current `build()` bakes them in.

### Pitfall 3 — `ApplyReport.fixed_parameters` for richer surrogates

Currently `_attach_surrogate` returns `fixed={}` for non-constant-intensity
swaps. After this change, coefficient Vars are fixed at fitted values. The
`ApplyReport` should record them. Updated in Step 6.

### Pitfall 4 — `constant_intensity` is not a `Surrogate` class

`constant_intensity` continues to fix the `energy_intensity` Var directly. No
change to `constant_intensity` handling in `apply.py`. The new
`register_surrogate_coefficients` path is only for swaps that returned a block.

### Pitfall 5 — `iter_swapped_relations` and block discovery

`iter_swapped_relations` yields `(block, RelationRecord)` pairs. The surrogate
block is accessible via `record.surrogate_block`. No change needed to
`iter_swapped_relations` itself, but downstream code (e.g. reports) should
check `record.surrogate_block` is not `None` before accessing it.

### Pitfall 6 — `register_coefficients` called before the block is added to a unit

`register_coefficients` operates on an un-added `pyo.Block`. This is fine
because it only declares `pyo.Var` components on the block object in memory;
`add_component` is not required until `swap_relation` attaches the block to
the unit. No pitfall here, but `build()` must not call `unit.add_component`.

### Pitfall 7 — `CoefficientRegistry` not populated before `body(t)` is called

`body(t)` is evaluated inside the fitted constraint's rule after the block is
attached to the unit. If `build()` forgets to call
`coefficients.register_coefficient(...)` for a name that `body(t)` references,
the constraint rule raises a `KeyError` at solve time. There is no build-time
check because `body(t)` is a closure, not a method on the block. Document this
in the `Surrogate.build()` contract: every name referenced in `body(t)` must be
registered before returning.

### Pitfall 8 — Testing regression fix/unfix cycle

A new component test must verify:
1. `swap_relation` creates a block with coefficient Vars fixed at their
   initialized values, and the fitted constraint lives on that block.
2. `unfix_surrogate_coefficients` unfixes all coefficient Vars in the active
   block.
3. `fix_surrogate_coefficients` fixes them again at their current values.
4. The unfix → solve → fix cycle preserves the solved coefficient values.

### Pitfall 9 — Reactivating a deactivated surrogate without updating the registry

Calling `block.activate()` directly without also calling
`switch_surrogate_block` leaves `record.surrogate_block` pointing at the
previously active block. The registry therefore still points at the wrong
coefficients. Always use `switch_surrogate_block` for reactivation; it
handles deactivation of the current block, activation of the target block,
registry update, and coefficient re-registration in one call.

## Tests

### Unit tests

- `test_swap_relation_returns_block` — `swap_relation` on a multilinear spec
  returns a non-None block; `record.surrogate_block` is set.
- `test_fitted_constraint_on_surrogate_block` — after swap, the fitted constraint
  is a child of the surrogate block, not the unit.
- `test_surrogate_block_has_coefficient_registry` — block has a
  `CoefficientRegistry` at `block.coefficients`; registry contains entries for
  each coefficient name from the spec, all fixed at their initialized values.
- `test_surrogate_block_naming` — first swap gets `surrogate_power`; a second
  swap gets `surrogate_power_1`.
- `test_register_coefficient_adds_entry` — calling
  `block.coefficients.register_coefficient("c", var)` makes
  `block.coefficients["c"]` return the same Var.
- `test_register_coefficients_bulk_add` — `register_coefficients({...})` adds
  multiple entries in one call.
- `test_register_coefficient_duplicate_raises` — registering the same name
  twice raises `FlexConfigError`.
- `test_register_coefficient_rejects_non_var` — passing a non-Var raises
  `FlexConfigError`.
- `test_register_surrogate_coefficients_populates_registry` — after calling the
  new method, `iter_io_registry` yields coefficient Vars as regressable
  parameters and the performance target is absent.
- `test_coefficient_vars_fixed_after_fit` — fix coefficients at known values,
  evaluate the body at several `t`, compare to the original float-baked
  expression (rel=1e-6).
- `test_registry_dynamic_expansion` — calling `register_coefficient` multiple
  times in `build()` accumulates entries; all are accessible via
  `block.coefficients`.
- `test_switch_surrogate_block_switches_active` — after two swaps,
  `switch_surrogate_block("surrogate_power")` reactivates the
  first block and deactivates the second; `record.surrogate_block` points
  to the first.
- `test_switch_surrogate_block_unknown_name_raises` — passing a block name
  not in the history raises `FlexConfigError` with the available names.
- `test_list_surrogate_blocks_returns_names` — after two swaps,
  `list_surrogate_blocks(relation)` returns `["surrogate_power",
  "surrogate_power_1"]` (oldest first).
- `test_list_surrogate_blocks_empty_before_swap` — returns `[]` before any
  swap.
- `test_switch_surrogate_block_reregisters_coefficients` — after reactivation,
  `register_surrogate_coefficients` registers the reactivated block's
  coefficients (not the deactivated block's).
- `test_current_surrogate_block_returns_name` — returns the correct block name
  when active, `None` when no surrogate is active.
- `test_unfix_surrogate_coefficients_unfixes_all` — after swap, all coefficient
  Vars in the active block are fixed; after `unfix_surrogate_coefficients`,
  all are unfixed.
- `test_fix_surrogate_coefficients_fixes_at_current_value` — after unfixing and
  changing a coefficient value, `fix_surrogate_coefficients` locks each Var
  at its current value.
- `test_unfix_surrogate_coefficients_no_active_block_raises` — raises
  `FlexConfigError` when no surrogate is active for the relation.

### Component tests

- `test_apply_swaps_and_fixes_coefficients` — `apply_to_model` with a
  multilinear surrogate: after apply, the surrogate block's coefficient Vars
  are fixed at the fitted values and the fitted constraint is active on the
  block.
- `test_deactivate_surrogate_block_deactivates_fitted_constraint` — deactivate
  the surrogate block, assert the fitted constraint is also inactive.
- `test_swap_again_deactivates_old_block` — call `swap_relation` twice; the
  first surrogate block is deactivated, the second is active.
- `test_complex_coefficients_indexed_var` — build a surrogate whose
  `CoefficientRegistry` is populated by several `register_coefficient` calls
  (e.g. intercept first, then bulk terms); verify
  `register_surrogate_coefficients` registers all of them and fixing them
  reproduces the expected output.
- `test_reactivate_previous_surrogate_and_refit` — swap twice to build two
  surrogate blocks for the same relation; reactivate the first block via
  `switch_surrogate_block("surrogate_power")`, call
  `register_surrogate_coefficients`, fix its coefficients at known values, and
  assert the output matches the first surrogate's expected values (rel=1e-6);
  assert the second block is deactivated.
- `test_apply_to_model_with_active_surrogates` — build a unit with two
  successive surrogate swaps; call `apply_to_model(active_surrogates={unit_name:
  "surrogate_power"})` and assert the first block's coefficients are fixed and
  its fitted constraint is active.

## Definition of Done

- [ ] `CoefficientRegistry` class exists with `register_coefficient(name, var)`,
      `register_coefficients(mapping)`, and dict-like access (`items()`,
      `__getitem__`, `__contains__`). Duplicate names raise `FlexConfigError`.
- [ ] `Surrogate.build()` returns `(block, body)` for all implemented surrogate
      classes; `build()` never calls `unit.add_component`.
- [ ] `swap_relation` uniquifies the block name (`surrogate_{base}` or
      `surrogate_{base}_{counter}`), adds the un-added block to the unit, places
      the fitted constraint **on the surrogate block**, and tracks the block in
      `record.surrogate_block` and `record.components`.
- [ ] `register_surrogate_coefficients` iterates `block.coefficients.items()`,
      registers all Vars as regressable parameters, and removes the performance
      target from the parameters list.
- [ ] `switch_surrogate_block(block_name)` reactivates a
      previously deactivated surrogate block by local name, finds the owning
      ``RelationRecord`` automatically, updates ``record.surrogate_block``,
      and re-registers its coefficients.
- [ ] `list_surrogate_blocks(relation_name)` returns the local names of all
      surrogate blocks ever built for the relation, oldest first.
- [ ] `current_surrogate_block(relation_name)` returns the currently active
      block's local name, or `None`.
- [ ] `unfix_surrogate_coefficients(relation_name)` unfixes all coefficient
      Vars in the active surrogate block for the relation.
- [ ] `fix_surrogate_coefficients(relation_name)` fixes all coefficient Vars
      in the active surrogate block at their current values.
- [ ] `apply_to_model` accepts `active_surrogates: dict[str, str]` mapping
      unit names to block names; for listed units it calls
      `switch_surrogate_block(block_name)` before fitting.
- [ ] `apply_to_model` fixes coefficient Vars at fitted values after
      `swap_relation` / `switch_surrogate_block` + `register_surrogate_coefficients`.
- [ ] All existing tests pass (no behavioral regression for constant-intensity path).
- [ ] New unit and component tests listed above pass.
- [ ] `ruff check . && black --check . && lint-imports && pytest -q` clean.
