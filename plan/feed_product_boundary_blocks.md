# Feed and Product boundary blocks

> **Read this first — this is not a milestone.**
>
> `CLAUDE.md`'s session-start checklist tells you to find the current milestone and build
> exactly that. The current milestone is **M10 — FlexParameterize core**
> (`plan/milestones/future/M10_parameterize_core.md`). **Do not build M10.** This document
> is an off-ladder patch PR, following the precedent of the mixer/splitter work merged in
> `d5b89d5` and the other post-M09 patches on `main`. Build what is specified here, on
> branch `feature/feed_block`, and leave M10 alone.
>
> Everything else in `CLAUDE.md` still applies unchanged — test-first, one tier marker per
> test, the package DAG, never delete Pyomo components, docs and CHANGELOG are part of Done.
> This document plays the role a milestone file would: its Test plan is the behavioral spec,
> and it is the thing to satisfy before opening the PR.

## Context

flex-pse models facilities as `PlantBlock`s of unit models wired by arcs, but it has **no
boundary blocks**. Today a feed is created by fixing an inlet state block's
`flow_vol_phase[t, phase]` by hand, and a product exists only as
`PlantBlockData.register_product(var, name=..., quality=...)` — a bare registration call the
user must remember to make against some other unit's variable. Nothing owns a boundary
stream, nothing meters the total resource crossing it, and nothing prices it.

This change adds two unit models that close that gap:

- **`Feed`** — a source. Zero inlets, N named outlets. Meters total withdrawal, carries the
  boundary state (composition, pressure, temperature — whatever the property package
  exposes), constrains withdrawal over time, and optionally prices the resource into opex.
- **`Product`** — a sink. N named inlets, zero outlets. Aggregates delivery, constrains
  demand over time, and optionally prices it (negative price = revenue).

Both are aggregated at the plant level into `total_feed[resource, t]` / `total_product[resource, t]`,
which the existing `NetworkBlock` machinery then sums across plants and links between them.

**Scope decisions already made** (do not relitigate):
- Composition is whatever the configured **property package** carries. No new quality concept —
  the boundary state block's variables (including IDAES on-demand properties) *are* the
  composition, and arcs propagate them downstream.
- Prices are **flat scalars**, via the existing `FlexCosting.register_scalar_cost`. No
  time-varying price work.
- Plants **discover** boundary flows through the `_io_registry`, the same way they discover
  power and fuel. No explicit user registration call.
- This is an off-milestone patch PR (mixer/splitter precedent), not a new milestone file.
  Current milestone M10 is untouched.

---

## Design

### 0. Where the code goes

| File | Status | Holds |
|---|---|---|
| `src/flexops/unit_models/feed.py` | new | `FeedData` / `Feed` |
| `src/flexops/unit_models/product.py` | new | `ProductData` / `Product` |
| `src/flexops/unit_models/_boundary.py` | new | `add_time_limits`, the one shared helper |
| `src/flexops/core/registration.py` | edit | `BoundaryKind`, `BoundaryRecord`, `IORegistry.boundary` |
| `src/flexops/core/ops_block.py` | edit | `register_boundary_flow` |
| `src/flexops/core/plant_block.py` | edit | `_feed_terms`, `total_feed`, merged `_product_terms` |
| `src/flexops/core/network_block.py` | edit | `_feed_terms` |
| `src/flexcore/nomenclature.py` | edit | `TOTAL_FEED` |

`Feed` and `Product` are **unit models**, not core infrastructure. `unit_models/__init__.py::__all__`
*is* the registry `UnitConfig.unit_model_class` resolves against, so a block outside it cannot
be built from a config file; it is also what the `unit_model.rst` docs template and the
`UnitModelTestHarness` conventions key off. `flexops/core/` holds only the substrate
(TimeBlock, OpsBlock, Plant/Network, registration, build) — nothing that carries ports.

The consequence is one-directional: `unit_models` imports `core`, so **`core/plant_block.py`
cannot import `Feed`/`Product` to discover them.** That is what §1 solves.

### 1. A boundary record on the IO registry

The plant must aggregate feeds/products **without importing `flexops.unit_models`** — that
would be a circular import (`unit_models` already imports `core`). Follow the existing
power/fuel pattern exactly: discovery keys off a duck-typed record on `_io_registry`, not off
a class.

`src/flexops/core/registration.py` — add:

```python
class BoundaryKind(enum.StrEnum):
    FEED = "feed"
    PRODUCT = "product"

@dataclass
class BoundaryRecord:
    var: Any          # time-indexed total flow across the boundary
    name: str         # the var's local name on its unit block
    resource: str     # aggregation key, e.g. "raw_water", "brine"
    kind: BoundaryKind
```

Add `boundary: list[BoundaryRecord] = field(default_factory=list)` to `IORegistry` and
include it in `is_empty()`.

`src/flexops/core/ops_block.py` — add `register_boundary_flow(self, var, *, resource, kind)`,
modelled line-for-line on [`register_fuel_usage`](../src/flexops/core/ops_block.py#L485): raise
`FlexConfigError(field="resource", value=...)` on an empty resource name, else append the
record. Any unit may call it, not just `Feed`/`Product`.

### 2. Plant and network aggregation

`src/flexcore/nomenclature.py` — add `TOTAL_FEED = "total_feed"` next to the existing
`TOTAL_PRODUCT`.

`src/flexops/core/plant_block.py`:
- `_AggregatingFlowsheet._feed_terms()` — abstract, raises `NotImplementedError`, mirroring
  `_fuel_terms`.
- `_AggregatingFlowsheet._build_aggregates()` — build `total_feed[resource, t]` via the
  existing `_refresh_expression`, guarded on a non-empty `_feed_terms()`, in the same style
  as the `total_fuel_usage` block immediately above it.
- `PlantBlockData._feed_terms()` — walk `component_data_objects(pyo.Block, descend_into=True)`
  for `_io_registry`, collect `BoundaryKind.FEED` records into `{resource: [getters]}`.
  Copy the shape of [`PlantBlockData._fuel_terms`](../src/flexops/core/plant_block.py#L256).
- `PlantBlockData._product_terms()` — **override** and merge two sources: `super()._product_terms()`
  (the existing explicit `register_product` registry, which must keep working unchanged) plus
  the discovered `BoundaryKind.PRODUCT` records.

`src/flexops/core/network_block.py`:
- `_feed_terms()` — sum each child plant's `total_feed[resource, t]`, a direct copy of
  [`_fuel_terms`](../src/flexops/core/network_block.py#L180). Never re-walk units (composition
  invariant, R7).

Cross-network hooks need **no new API**: `NetworkBlock.add_link(name, source, destination)`
already exists and its docstring already names "one plant's product against another's feed".
It takes time-indexed quantities, so a plant total is passed as
`pyo.Reference(plant.total_product["potable_water", :])` and a boundary total as
`feed.withdrawal` directly. Document this pattern; add no code for it.

### 2b. Multiple feeds and products per plant, keyed by name

A plant normally has several boundary streams — raw water, citric acid, and antiscalant in;
potable water, brine, and waste out — so **many `Feed`/`Product` blocks per plant is the
expected case, not the edge case.** The design that makes this work:

- **One block, one resource.** A `Feed` carries exactly one `resource_name`. Several inbound
  resources means several `Feed` blocks, the same way N streams means N ports. Do not add a
  multi-resource `Feed`.
- **`resource_name` is the aggregation key, and it is independent of the Pyomo block name.**
  `total_feed` is indexed `(resource, t)`, so two `Feed` blocks with different
  `resource_name`s give two distinct rows, and two blocks sharing a `resource_name` **sum
  into one row** — which is how you model the same resource entering at two points
  (e.g. city water into two separate trains). This mirrors `register_fuel_usage`'s
  `fuel_name` exactly.
- **The default keeps the common case free.** `resource_name=None` falls back to the block's
  `local_name` in `build()`, so `plant.raw_water = Feed(...)` and
  `plant.citric_acid = Feed(...)` are automatically distinct resources with no extra config.
  Set `resource_name` explicitly only to override that — i.e. to *merge* blocks under one key,
  or when the block name is not the name you want in `total_feed` and the cost report.
- **Costing stays per block.** The scalar-cost name is derived from the block's full dotted
  Pyomo name (sanitized), not from `resource_name`, so two blocks sharing a resource still get
  separate, non-colliding opex line items and separate `unit=` attribution — even though their
  flows aggregate together. Prices may legitimately differ between them.
- **The network already composes this.** `NetworkBlockData._feed_terms` sums each child
  plant's `total_feed[resource, t]` per resource key, so a resource appearing in three plants
  becomes one network row without any per-plant bookkeeping.

### 3. `src/flexops/unit_models/feed.py`

`FeedData(OpsBlockData)` under `@declare_process_block_class("Feed")`. Direct `OpsBlockData`
subclass because the port count is a config option — same reasoning as `Mixer`/`Splitter`
(architecture §3.4). Mirror the structure of
[splitter.py](../src/flexops/unit_models/splitter.py) closely; it is the nearest sibling.

**Config** (all `ConfigValue` with a `description=`; validation lives in `build()`, never in a
domain — Pyomo wraps domain exceptions into bare `ValueError` and loses `field`/`value`):

| key | default | meaning |
|---|---|---|
| `outlet_names` | `("a",)` | role names; outlet *i* is port `f"outlet_{name}"`. First is the reference outlet. |
| `resource_name` | `None` | aggregation key; falls back to the block's `local_name` in `build()`. |
| `max_withdrawal` | `None` | units-carrying scalar; builds the upper-limit Param + Constraint. |
| `min_withdrawal` | `None` | units-carrying scalar; builds the lower-limit Param + Constraint. |
| `price` | `None` | flat price per unit withdrawn, in the costing currency basis. Positive = cost. |

**`build()` sequence:**

1. `super().build()`
2. `validate_port_names(self.config.outlet_names, "outlet_names")` and
   `single_flow_phase(self.config.property_package, "Feed")` — reuse
   [`_multiport.py`](../src/flexops/unit_models/_multiport.py) as-is.
3. `self.add_stream_ports(inlet_ports=(), outlet_ports=self._outlet_port_names())` — verify
   an empty `inlet_ports` tuple is accepted; it iterates, so it is, but assert it in a test.
4. `_register_stream_states()` — the reference outlet's non-flow state vars are the
   **boundary conditions**, registered `role="input"`. Every other outlet's are `role="output"`
   (they are pinned by the ties in step 5). Copy the shape of
   [`Splitter._register_stream_states`](../src/flexops/unit_models/splitter.py#L127).
5. `_tie_outlet_states()` — for N > 1, hold every non-reference outlet's non-flow state vars
   equal to the reference outlet's, indexed `(time, other_names)`, named
   `outlet_state_equality_{state_var}`. One source, one condition. This is structurally the
   same as [`Mixer._tie_inlet_states`](../src/flexops/unit_models/mixer.py#L345) — **write it
   fresh in `feed.py` rather than refactoring `Mixer`** (CLAUDE.md forbids touching prior
   work; the repo already accepts this Mixer/Splitter symmetry duplication).
6. `_build_withdrawal()` — per-outlet `flow_out_{name}` References into
   `state.find_component(flow_name)[:, self._phase]`, then:
   - `withdrawal = pyo.Var(tb.time_index, units=<flow basis units>, doc=...)`
   - `eq_withdrawal[t]: withdrawal[t] == Σ_i flow_out_{i}[t]`

   A **Var + `eq_` equality**, not an Expression — it must be boundable, fixable via
   `set_external_dispatch`, and consumable by costing. This matches FlexCosting's own
   "every derived quantity is a Var defined by an `eq_<name>` equality" convention.
7. `add_time_limits(self, self.withdrawal, "withdrawal", lower=..., upper=...)` (see §5).
8. `self.register_boundary_flow(self.withdrawal, resource=self._resource, kind=BoundaryKind.FEED)`
9. `_register_cost()` — if `config.price is not None` **and** `config.costing_package is not None`:
   ```python
   self.config.costing_package.register_scalar_cost(
       name=self.name.replace(".", "_"),   # dotted Pyomo name would break opex.add_component
       quantity=self.withdrawal,
       price=self.config.price,
       quantity_units=<flow basis units>,
       unit=self,
   )
   ```
   `price` set with no `costing_package` must raise `FlexConfigError(field="price")` rather
   than silently dropping the cost.

No power is declared or registered — like `Mixer` and `Splitter`, a boundary block has no
energy relation. The harness's `test_energy_naming` verifies this.

### 4. `src/flexops/unit_models/product.py`

`ProductData(OpsBlockData)` under `@declare_process_block_class("Product")`. The mirror image:
`inlet_names`, ports `f"inlet_{name}"`, `delivery[t]` Var + `eq_delivery[t]`,
`max_demand`/`min_demand`, `price`, `BoundaryKind.PRODUCT`.

Two deliberate asymmetries with `Feed`, both of which belong in the class docstring:

- **No state ties, no blending.** Each inlet's intensive states arrive from its arc, so they
  are registered `role="output"` and left alone. A `Product` aggregates *flow* and does not
  blend composition, temperature, or pressure — blending is `Mixer`'s job and is bilinear.
  Put a `Mixer` upstream if a single blended stream is wanted. Do not duplicate mixing physics.
- **Composition constraints are per-inlet.** `add_time_limits(product, getattr(product.inlet_a_state, "conc_mass_phase_comp")[...], ...)`
  bounds one named inlet's state, not a blend.

Price sign follows `register_scalar_cost` unchanged: **positive = cost, negative = revenue**.
Document both cases in the docstring — brine disposal is a positive price, potable water sold
is a negative one.

### 5. `src/flexops/unit_models/_boundary.py` — the one shared helper

A new private module alongside `_multiport.py`, holding a single module-level function used
by both blocks:

```python
def add_time_limits(block, quantity, name, *, lower=None, upper=None) -> None:
    """Bound a time-indexed quantity with mutable per-period limit Params."""
```

For each of `lower`/`upper` that is not `None`, build on `block`:
- `{name}_min` / `{name}_max` — **mutable** `Param` over `tb.time_index`, initialized to the
  scalar, carrying the scalar's units.
- `{name}_min_limit` / `{name}_max_limit` — `Constraint(tb.time_index)`.

Mutable Params rather than `Var.setlb/setub` because (a) `set_value` is the sanctioned
in-place update path (conventions §9), (b) they survive `set_external_dispatch` fixing the
Var, and (c) they carry duals, so the shadow price of a resource limit is readable after a
solve — a genuinely useful output for a boundary block.

**Time variation** comes from writing the Params, not from config: `feed.withdrawal_max[t].set_value(v)`
in a loop. Config takes a units-carrying scalar only. An *exact* profile uses the inherited
`set_external_dispatch(feed.withdrawal, series, fix=True)`, which already parses
index- or timestamp-keyed series. **Add no series-parsing code and no `set_limit_profile`
convenience method** — document the two idioms instead.

### 6. On-demand properties: use `getattr`, not `find_component`

Both flex-pse property packages declare `flow_mass_phase`, `dens_mass_phase`, `pressure`, and
`temperature` as IDAES on-demand properties (`define_metadata` → `{"method": "_flow_mass_phase"}`),
built lazily through `StateBlockData.__getattr__`. `find_component` will return `None` for an
unbuilt one. So when resolving a **state-block property by name**, use `getattr(state, name)`;
the repo's usual `find_component` preference applies to components on the unit block itself
and still holds everywhere else in these two files. Say so in a comment where it happens.

### 7. Exports and docs

- `src/flexops/unit_models/__init__.py::__all__` += `"Feed"`, `"Product"` — this list **is**
  the registry `UnitConfig.unit_model_class` resolves against, so config-driven builds get
  them for free.
- `src/flexops/__init__.py::__all__` += the same two names (plus `BoundaryKind` if it should
  be user-visible — it should, since `register_boundary_flow` is public).
- `docs/reference/flexops/unit_models/index.rst` — a prose paragraph each plus an
  `autosummary` block with `:template: unit_model.rst`, copying the `Splitter` entry's shape.
- `docs/reference/flexops/core.rst` — `BoundaryKind` / `BoundaryRecord` / `register_boundary_flow`
  where `registration` is documented.
- `docs/how_to/build_a_plant.md` — extend the worked example to bracket the plant with a
  `Feed` and a `Product`, and show one `NetworkBlock.add_link` from one plant's
  `total_product` to another's feed.
- `CHANGELOG.md` under `## [Unreleased] / ### Added` — one prose paragraph per block in the
  house style, plus the registry/aggregation additions.

The frozen-API fixture (`src/flexops/tests/fixtures/api_freeze/`) and
`src/flexops/tests/test_api_freeze.py` are **not touched**;
`test_api_freeze` does not assert `__all__`, so these exports are purely additive.

---

## Test plan (write these first — they are the spec)

Every test carries exactly one tier marker or collection fails. Confirm each new test fails
with `ImportError`/`AttributeError` before implementing.

`src/flexops/tests/unit_models/test_feed.py` and `test_product.py`, following the shape of
[test_splitter.py](../src/flexops/tests/unit_models/test_splitter.py) and the richer
[test_mixer.py](../src/flexops/tests/unit_models/test_mixer.py):

- **Harness classes** (`UnitModelTestHarness`, via `dummy_time_block(3)` / `dummy_gas_time_block(3)`):
  - one-outlet aqueous `Feed` with `set_external_dispatch(unit.withdrawal, {...})` in
    `configure()` → `expected_dof = 0` and a real `expected_solution` for `withdrawal`;
  - two-outlet aqueous `Feed` with no dispatch → non-zero `expected_dof` (count it: for
    n=3 and a flow-only package, `2N` vars minus `N` `eq_withdrawal` rows = 3);
  - a gas `Feed` (pressure + temperature present) exercising the outlet state ties;
  - the mirror set for `Product`.
- **Unit tier** (`@pytest.mark.unit`, no solver):
  - port names are exactly `{"outlet_a", "outlet_b"}` / `{"inlet_a", ...}` and no inlet
    (resp. outlet) port exists;
  - `eq_withdrawal[t].body` evaluates to 0 on a hand-computed trajectory
    (the `test_tank.py::test_mass_balance_by_hand` idiom);
  - `outlet_state_equality_pressure[t, "b"].body` == 0 with the reference set;
  - limit Params/Constraints exist **iff** configured, and their bodies evaluate correctly;
    mutating `withdrawal_max[1].set_value(...)` changes the constraint body;
  - `_io_registry.boundary` holds one record with the right `resource` and `kind`;
    `resource_name=None` falls back to `local_name`, and an explicit `resource_name`
    overrides it while leaving the block name alone;
  - config rejection: empty/duplicate `outlet_names` raises with
    `excinfo.value.field == "outlet_names"`; `price` without `costing_package` raises with
    `field == "price"`.
- **Component tier** (`@pytest.mark.component @pytest.mark.needs_highs`): `Feed → Pump → Product`
  inside a `PlantBlock`, wired with `Arc`s, `plant._build_aggregates()`, then
  `TransformationFactory("network.expand_arcs").apply_to(m)` and solve. Assert
  `total_feed["raw_water", t]` and `total_product["potable_water", t]` match the flows.

`src/flexops/tests/core/test_plant_block.py` — extend with a local dummy block that calls
`register_boundary_flow` (mirroring the existing `DummyFuelUnit`), asserting `total_feed` /
`total_product` aggregation, re-entrancy of `_build_aggregates()`, and that an explicit
`register_product(...)` and a discovered `Product` block **both** land in `total_product`
without double-counting. Cover the multi-boundary cases from §2b explicitly:

- a plant with `raw_water` and `citric_acid` `Feed`s → `total_feed` has exactly those two
  resource keys, each summing only its own block;
- two `Feed` blocks sharing `resource_name="city_water"` → **one** key whose body is the sum
  of both `withdrawal`s (check the Expression body, `unit` tier);
- the same plant carrying feeds *and* products → the two totals stay disjoint, and a `Feed`
  never leaks into `total_product`.

`src/flexops/tests/core/test_network_block.py` — two plants each with a `Feed`/`Product`;
assert the network's `total_feed`/`total_product` equal the sum of plant totals, and that
`add_link` ties one plant's product total to another's `feed.withdrawal`.

`src/flexops/tests/costing/test_flex_costing.py` — a `Feed` with a positive price and a
`Product` with a negative one; `cost_process()`; assert `report_cost(model).operating.scalar`
matches a hand-computed `price × Σ_t flow[t] × dt` to the cent, and that the revenue leg is
negative.

---

## Verification

```bash
conda activate flex-pse
export PATH="$HOME/.idaes/bin:$PATH"        # else needs_ipopt tests silently skip

pytest -m "unit" -x -q                       # inner loop
pytest -m "unit or component" -q             # after each work unit

# before pushing — the pre-push hook runs the same:
ruff check . && black --check .
lint-imports                                 # must stay clean: no core → unit_models import
pytest -q                                    # ALL tiers
sphinx-build -W --keep-going -b html docs docs/_build
```

End-to-end smoke check that the new blocks work through the config path as well as the
imperative one: add a `Feed`/`Product` pair to a copy of
`src/flexops/tests/fixtures/plant_config_demo.json` and build it with
`flexops.build_model(...)`, confirming `unit_model_class: "Feed"` resolves and the model solves.

## Watch-outs

- **Circular import.** The blocks live in `unit_models/` (§0) and `unit_models` imports `core`,
  so the aggregation code in `core/plant_block.py` must never import them back. Discovery goes
  through `_io_registry.boundary`, never `isinstance(block, FeedData)`. `lint-imports` will not
  catch a violation (it only enforces the four-package DAG) — a `pytest -q` `ImportError` will.
- **Never delete Pyomo components.** `_build_aggregates` is re-entrant by design; extend it
  with `_refresh_expression`, never by rebuilding.
- **`declare_process_block_class` names.** The class is `FeedData`; `Feed` is minted at
  runtime. Tests import `Feed` from `flexops.unit_models`.
- **Dotted cost names.** `self.name` is dotted inside a plant; sanitize before handing it to
  `register_scalar_cost`, or `opex.add_component(f"scalar_cost_{name}")` builds an invalid
  component name.
- **No milestone/decision citations in `src/`.** Reference architecture sections (§3.3, §3.4)
  in docstrings, never `M##` or `R##`.
- **`register_product` must keep working.** `PlantBlockData._product_terms` merges the old
  explicit registry with the new discovery; existing tests in `test_network_block.py` cover
  the old path and must stay green untouched.
