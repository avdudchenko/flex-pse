# M14 — Docs completion + example notebooks

**Note:** M12/M13 (FlexSchedule rolling horizon + set-point extraction) and M16
(`flexops.design`) moved to the 0.2 milestone set (`PLAN.md` §4) before this
milestone ran. Every mention of them below is historical context from when this
work order was drafted alongside them — this milestone does **not** document
`flexschedule` or `flexops.design`, since neither exists yet in 0.1.0.

**Note:** Example notebooks no longer live in this repository. They live in the
companion repo `flex-pse-examples` (`github.com/flex-pse/flex-pse-examples`)
and this repo's docs build pulls them in at a pinned ref. See "5. Notebooks"
below.

**Effort:** 2–3 days · **Depends on:** M11b · **Parallelizable:** no

## Goal

Finish the documentation system: final `conf.py`, the `flexdoc` Sphinx extension
that generates unit-model Variables/Constraints/DoF tables from built models (so
docs cannot drift from code), the unit-model autosummary template, a sweep of all
reference pages onto the generated directives, the docs build wired up to pull
two executed example notebooks from the `flex-pse-examples` repo, and the
finalized `docs.yml` workflow. This milestone also lands **reference-page +
flexdoc coverage for every public surface built in M08/M09** — `NetworkBlock`,
the SISO/SIDO/DIDO topology bases, the full physical unit zoo, the customizable
unit-commitment logic modules, the EECO post-hoc `evaluate_cost`/`report_cost`,
and the config-driven `build_model` with its JSON config schema. Exit state:
`sphinx-build -W` is clean in both execution modes, every public unit model
page renders generated tables, and everything a reader sees under `docs/` reads
like a tool manual — no internal project history required to understand it.

## Read first

- `plan/03_documentation.md` — ALL of it; this milestone implements §1–§6 to completion
- `plan/01_architecture.md` §3.2 (IORegistry / registration API — flexdoc's data source), §3.4 (the unit-model table: **every** class needs a reference page — SISO/SIDO/DIDO bases and the physical zoo: Pump, Tank, Separator, Exchanger, ElectrolysisSeparator, ElectrolysisExchanger, ReverseOsmosis, Combustor, BatteryModel, ConstantEnergyIntensityModel)
- `plan/01_architecture.md` §3.3 (`NetworkBlock`/`PlantBlock` composition — reference pages), §3.5 (the customizable unit-commitment logic modules), §3.6 (`report_cost`/post-hoc EECO evaluation; the reported cost is never the objective — R9), §2.3 (config-driven `build_model` + the JSON config schema)
- `plan/02_testing_and_ci.md` §3 (docs.yml and nightly.yml specs), §1 (tier markers for the flexdoc unit test)
- `PLAN.md` §2 (the api_freeze script — notebook 01's source material)
- `plan/00_conventions.md` §8 (docs summary rules)

**A note on citations in this file vs. citations in the docs themselves:** this
file (and the architecture/convention docs it points to) freely cites decision
codes (`R9`), milestone codes (`M08`), and section numbers (`§3.4`) — that's
internal project shorthand for whoever builds the next milestone, and it stays
out of the public docs entirely. See "0. Writing for the public docs" below
before touching any file under `docs/` or in `flex-pse-examples`.

## Files to create or modify

- `docs/conf.py` — final configuration (see spec)
- `docs/_ext/flexdoc.py` — `flexops-unit-tables` and `flexops-config-table` directives
- `docs/_templates/autosummary/unit_model.rst` — per 03 §3
- `docs/reference/**` — sweep every existing reference page onto the directives; remove TODO markers left by M02–M11b; add pages for the SISO/SIDO/DIDO bases, the full physical zoo, `NetworkBlock`, the logic modules, and `FlexCosting.evaluate_cost`/`report_cost`
- `docs/explanation/config_schema.md` — render pydantic models via `flexops-config-table`; document the JSON config schema driving `build_model`
- `docs/explanation/reported_cost.md` (or a section in `energy_nomenclature.md`) — the reported electricity cost is computed after the solve, never read off the solver's internal objective
- `docs/examples/` — the include mechanism that pulls rendered notebooks from the `flex-pse-examples` repo into the doc tree (see "5. Notebooks")
- `.github/workflows/docs.yml` — finalize PR + deploy jobs, including the pinned checkout of `flex-pse-examples`
- `.github/workflows/nightly.yml` — ensure a notebook-execution step exists (02 §3 requires it; add if missing)
- `src/flexops/tests/docs/test_flexdoc_tables.py` — unit test for table generation

## Specification

### 0. Writing for the public docs

Everything under `docs/` and everything in `flex-pse-examples` is read by
people who have never seen this repository's planning files and never should
need to. Two rules apply to every page, docstring, and notebook cell you write
or touch in this milestone:

1. **No internal project shorthand.** Never write a milestone code (`M08`,
   `M14`, …), a decision code (`R6`, `R9`, …), or a `§`-numbered
   architecture/plan cross-reference in anything a reader of the public docs
   sees — that includes reference pages, explanation pages, how-to pages,
   getting-started pages, class/function docstrings (they render via
   napoleon), and notebook markdown/code cells. If a design choice needs
   explaining (e.g. "why is the reported cost computed after the solve, not
   read off the objective"), explain the reasoning itself in plain prose —
   never point the reader at a decision code or plan section to look it up,
   because they have no access to `plan/`. When sweeping `docs/reference/` in
   §4, grep for stray citations left by earlier milestones
   (`grep -rEn '\bR[0-9]+\b|\bM[0-9]{2}\b|§' docs/`) and rewrite every hit in
   plain language.
2. **Write for a domain expert or an undergraduate, not a contributor to this
   repo.** Assume the reader knows process/energy engineering (pumps, tanks,
   electrolyzers, tariffs, load shifting) but not this codebase's internal
   history or design debates. Prefer short sentences and concrete examples
   over abstract framing. Define a term the first time you use it rather than
   assuming the reader has read another page first. This applies to
   `getting_started/`, `how_to/`, `explanation/`, and every notebook; it does
   **not** relax the technical precision required of `reference/` pages
   (those describe the API as it behaves, generated tables included) — keep
   `reference/` accurate and plain, not vague.

### 1. `docs/conf.py` (final)

Per 03 §1:

- Extensions: `sphinx.ext.autodoc`, `sphinx.ext.autosummary`
  (`autosummary_generate = True`), `sphinx.ext.napoleon` (Google style),
  `sphinx.ext.intersphinx`, `myst_nb`, plus the local `flexdoc` (add
  `docs/_ext` to `sys.path` in conf.py).
- Intersphinx mappings to pyomo, idaes, pandas, pydantic (stable/latest docs
  URLs — implementer's choice of exact URLs; verify each resolves during build).
- Theme: `furo` (chosen in M00 — do not revisit).
- `nitpicky = True` with a curated `nitpick_ignore` list; every entry gets a
  one-line comment saying why (typically upstream objects missing from
  intersphinx inventories). Do not blanket-ignore whole modules.
- Notebook execution switch:
  `nb_execution_mode = os.environ.get("NB_EXECUTION_MODE", "cache")` — PR docs
  CI sets `NB_EXECUTION_MODE=off`; local/main builds cache (03 §4, §6).
- Notebooks are pulled from the `flex-pse-examples` repo into `docs/examples/`
  before the build (see "5. Notebooks"); `conf.py` globs that populated
  directory the same way it would glob a local `/examples` folder.

### 2. `docs/_ext/flexdoc.py`

Two directives, per 03 §2.

```rst
.. flexops-unit-tables:: flexops.unit_models.pump.Pump
```

At docs-build time the directive:

1. Imports the class from its dotted path; constructs it on
   `flexops.testing.dummy_time_block(n=3)` with `SimpleAqueousFlow` defaults
   (the helper is provided by `flexops.testing`; if a unit needs a costing
   package to build, the helper provides a stub — implementer's choice,
   mirroring what `UnitModelTestHarness.configure` defaults do).
2. Reads the unit's `IORegistry`, the `doc=` strings on its Vars/
   Constraints, and their units.
3. Emits three `list-table` nodes:
   - **Variables** — name, index sets, units, IO role, description;
   - **Constraints** — name, description;
   - **Degrees of Freedom** — the registered inputs that must be fixed.

Implementation notes:

- Structure the extension as a thin directive over a pure function
  `collect_unit_tables(cls) -> dict[str, list[list[str]]]` returning the three
  tables as rows of strings — the unit test calls this function directly, no
  Sphinx app needed (implementer's choice, but keep the split: it is what makes
  the extension testable).
- A build-time failure (class won't import, model won't construct, registered
  variable missing a `doc=`) must fail the build with a clear error naming the
  class — under `sphinx-build -W` a warning suffices, but prefer raising. An
  empty table is a silent-drift bug, never acceptable output.
- Descriptions come from component `doc=` strings; 03 §2 already obligates every
  public Var/Constraint to carry one (the harness asserts non-empty for
  registered variables). Write those `doc=` strings as plain descriptions of
  what the quantity is (units, physical meaning) — they render straight into
  the public reference page, so §0's rules apply to them too.

```rst
.. flexops-config-table:: flexcore.config.schema.ModelConfig
```

- Renders a pydantic v2 model's fields as one list-table: name, type, default,
  description (from `model_fields` / `FieldInfo.description`). Used by
  `docs/explanation/config_schema.md` and unit-model config sections.

### 3. `docs/_templates/autosummary/unit_model.rst`

Per 03 §3, composes in order: (1) the napoleon-rendered class docstring,
(2) `.. flexops-unit-tables:: {{ fullname }}`, (3) `automethod` entries for
public methods beyond the standard block interface.
`docs/reference/flexops/unit_models/index.rst` autosummary uses this template
for the full v0 unit zoo (architecture §3.4). Coverage is **every public unit
model**, not just the original six:

- **Topology bases** (`flexops/unit_models/base/`): `SISOBlock`, `SIDOBlock`,
  `DIDOBlock` — one reference page each, documenting the ports / per-stream mass
  balance / energy-registration wiring they own.
- **Physical zoo**: `Pump`, `Tank`, `Separator`, `Exchanger`,
  `ElectrolysisSeparator`, `ElectrolysisExchanger`, `ReverseOsmosis`,
  `Combustor`, `BatteryModel`, `ConstantEnergyIntensityModel`.

The old `Electrolyzer` name is gone (it's called `Separator` now); the
electrolysis units are `ElectrolysisSeparator`/`ElectrolysisExchanger`. Do not
reference `Electrolyzer` anywhere in the docs. `Tank`'s page notes that
its unit-commitment logic is disabled — a tank has no on/off status.

### 4. Reference-page sweep

Earlier milestones left hand-written stubs and `TODO(M14)` markers. Sweep every
page under `docs/reference/` so that:

- every unit model renders via the autosummary template (delete hand-written
  variable tables — the generated ones are canonical);
- config-schema pages use `flexops-config-table`;
- `grep -rn "TODO" docs/` returns nothing;
- `grep -rEn '\bR[0-9]+\b|\bM[0-9]{2}\b|§' docs/` returns nothing (§0).

### 4a. Reference coverage for the M08/M09 public surfaces

Beyond the unit zoo, add or complete reference pages (autodoc/autosummary; use
generated directives where a directive fits) for:

- **Composition** (`docs/reference/flexops/core.rst`): `NetworkBlock` (composes
  plants) alongside `PlantBlock` (composes units) — architecture §3.3, R7.
  Document the recursive aggregation (`NetworkBlock` totals = Σ `PlantBlock`
  totals = Σ unit `power_electrical`/`power_thermal`).
- **Logic layer** (`docs/reference/flexops/logic.rst`): the customizable
  unit-commitment modules — `status`, `startup_shutdown`, `dwell`, `delays`,
  `conditional`, `bypass`, and the model-level `degeneracy` pass (architecture
  §3.5). State which pieces are optional (everything except `status`).
- **Costing post-hoc evaluation** (`docs/reference/flexops/costing.rst`):
  document `FlexCosting.evaluate_cost`/`report_cost` — the post-solve cost
  evaluation on the realized aggregate-power numpy array (architecture §3.6).
- **Config-driven `build_model`** (architecture §2.3): document
  `flexops.build_model(config)` and render the **JSON config schema** — the
  `ModelConfig` tree (`TimeConfig`, `CostingConfig`, `NetworkConfig`/`PlantConfig`,
  `UnitConfig`, `IOVariableSpec`, `SurrogateSpec`) via the
  `.. flexops-config-table::` directive on `docs/explanation/config_schema.md`
  (and cross-referenced from the how-to / getting-started build pages).

Remember §0: the reference pages above describe *what the API does*, in plain
language — cite architecture sections and decision codes only in this
milestone file, never on the rendered page itself.

### 4b. Explanation note — the reported cost is computed after the solve

Add a short note (smallest home: a section in `docs/explanation/energy_nomenclature.md`
or a dedicated `docs/explanation/reported_cost.md` — implementer's choice, but it
must be linkable) explaining, in plain language, that the electricity cost a
user sees is always `FlexCosting.report_cost` — it is evaluated *after* the
model solves, on the power values the solve actually produced — and is not the
same number the solver's internal objective was built from, because that
internal objective uses a simplified, easier-to-solve version of the true cost
function. State plainly why this distinction matters to a user (the number in
their report reflects reality; the objective is a solver aid) without citing
any decision code or architecture section on the page itself. The raw solver
objective is never the reported number; it appears only behind an explicit
debug flag.

### 5. Notebooks

Notebooks are **not** authored in this repository. They live in the public
companion repo `flex-pse-examples` (`github.com/flex-pse/flex-pse-examples`),
which this milestone does not create content for — only wires the docs build
to consume it. Rationale: the notebooks are the most example-driven, most
frequently iterated public-facing artifact, and keeping them in a lighter,
docs-only repo lets them be updated (and reviewed by non-contributors) without
touching this repo's release/versioning cadence.

- `docs.yml` and `nightly.yml` check out `flex-pse-examples` at a **pinned
  ref** (a tag or commit SHA recorded in a small manifest, e.g.
  `docs/examples_ref.txt` — implementer's choice of exact mechanism: git
  submodule, `actions/checkout` of the second repo into `docs/examples/`, or a
  script that fetches a tarball of that ref) before `sphinx-build` runs, so the
  built docs are reproducible from this repo's pinned reference and don't break
  when someone pushes to `flex-pse-examples` unreviewed.
- `conf.py`'s notebook glob and `nb_execution_mode` (§1) apply to whatever
  landed in `docs/examples/` after that checkout step — from Sphinx's point of
  view nothing else changes.
- The notebook content itself (variable names, plots, narrative) is out of
  scope for this milestone's diff, but if you touch it while wiring up the
  integration, the same rules apply as any other public doc: §0 (plain
  language, no internal codes) and the existing content rules below.
- Content rules carried over from 03 §4, non-negotiable wherever the
  notebooks are authored: horizons ≤ 2 days at 15 min; fixed seeds; no
  network access; **every notebook ends with an assert cell** on a numeric
  result (so "execution passed" means something).
- The two notebooks this milestone's docs build depends on existing in
  `flex-pse-examples`:
  - `01_build_a_plant.ipynb` — a walkthrough of building a small plant model
    (a time grid, a couple of unit models, a cost signal, wiring them
    together, solving, and plotting the resulting load shift against the
    time-of-use price) using only the public `flexops` API — no internal
    build scripts referenced.
  - `02_parameterize_from_data.ipynb` — a walkthrough of fitting a unit
    model's parameters from a small synthetic dataset and rebuilding the model
    from the fitted config, so a reader sees the full data-to-model path.
- A rolling-horizon notebook returns in 0.2 once M12/M13 land.
- If `flex-pse-examples` does not yet contain notebooks meeting the content
  rules above, creating/fixing them there is in scope for this milestone (it's
  the other half of "finish the docs system") — just do it in that repo, at
  the pinned ref this repo's CI checks out, not under `examples/` here.

### 6. Workflows

`docs.yml` (final, per 02 §3):

- Checkout step: fetch `flex-pse-examples` at its pinned ref into
  `docs/examples/` (see §5) before the Sphinx build step.
- PR job: `sphinx-build -W --keep-going -b html docs docs/_build` **with
  notebook execution on** (myst-nb `cache` mode; cache the jupyter-cache
  directory in CI keyed on the notebook file hashes so only changed notebooks
  re-run). A broken notebook blocks the merge like any other test. This job is
  a required status check. `NB_EXECUTION_MODE=off` remains for fast local
  iteration only.
- main job: full build with cached notebook execution, then deploy to
  **GitHub Pages** via `actions/upload-pages-artifact` + `actions/deploy-pages`
  (implementer's choice — 02 §3 allows Pages or RTD; record the choice in the
  PR description).

`nightly.yml` (safety net, never a gate — 02 §3): check out `flex-pse-examples`
at the same pinned ref, then force-execute the notebooks cache-free (e.g.
`NB_EXECUTION_MODE=force sphinx-build ...` or `jupyter nbconvert --execute`
over `docs/examples/*.ipynb` — implementer's choice; the requirement is that a
stale-cache or environment-drift breakage fails nightly within a day even when
PR builds hit warm caches).

## Pitfalls

1. **The docs build passing while tables are empty.** If `flexdoc` swallows a
   construction error and emits nothing, docs drift silently forever — the
   whole point of the extension dies. Fail loudly; keep the unit test that
   asserts real rows.
2. **Notebook horizons creeping up.** A 30-day notebook makes the PR docs gate
   slow and flaky. ≤ 2 days at 15 min = ≤ 192 steps; check before committing
   (in `flex-pse-examples`, at the pinned ref).
3. **PR docs CI re-executing unchanged notebooks.** The PR build executes
   notebooks (they gate the merge), but with a warm jupyter-cache only changed
   notebooks should re-run; if every PR re-executes both, the CI cache key
   is wrong.
4. **`nitpick_ignore` as a dumping ground.** Every ignore is curated + commented.
   If you're ignoring your own project's references, fix the docstring instead.
5. **Building the docs model at import time.** `flexdoc` must construct models
   inside directive `run()`, not at module import — Sphinx imports extensions
   before the environment is ready, and the unit test imports it too.
6. **Missing `doc=` strings discovered late.** Run the full `-W` build early;
   each missing description on a registered variable is a build failure to fix
   in the owning module (tiny diffs, but they touch earlier milestones' files —
   allowed here, this milestone says to sweep).
7. **Notebook outputs committed stale.** With `cache` mode, stale caches hide
   breakage locally; nightly's forced execution is the safety net — make sure
   the step actually runs both notebooks.
8. **Intersphinx flakiness.** Network fetch of inventories can fail in CI; pin
   the URLs and, if flakiness appears, commit local inventory fallbacks
   (implementer's choice — note it in the PR).
9. **An un-pinned `flex-pse-examples` checkout.** Tracking that repo's default
   branch means an unrelated push there can break this repo's docs build (or
   silently change what "the docs" show) with no corresponding PR here. Always
   pin to a recorded ref and bump it deliberately.
10. **Internal shorthand leaking into public docs.** A decision code, milestone
    code, or `§` reference copy-pasted from this file (or from a docstring
    written while thinking in those terms) into a reference/explanation page
    or a notebook is a regression against §0 — the `grep` in §4 is the
    mechanical check, run it as part of the sweep, not just once at the end.

## Tests

The docs build IS the test suite for this milestone:

- `NB_EXECUTION_MODE=off sphinx-build -W --keep-going -b html docs docs/_build` — clean (PR mode).
- `sphinx-build -W -b html docs docs/_build` — clean with executed/cached notebooks, after the `flex-pse-examples` checkout step has populated `docs/examples/` (main mode; run locally before merging).

Plus one real test file, `src/flexops/tests/docs/test_flexdoc_tables.py`
(location is implementer's choice — it lives under `flexops` because it
exercises flexops models; load `docs/_ext/flexdoc.py` via
`importlib.util.spec_from_file_location` with a path resolved from the repo
root, and `pytest.skip` with a clear reason if `docs/` is absent, e.g. in an
installed-package run):

- `test_unit_tables_pump` — `@pytest.mark.unit`: call `collect_unit_tables(Pump)`;
  assert the Variables table rows contain the registered variable names
  (`flow_vol_phase`, `power_electrical` at minimum), every row has a non-empty
  description and units string, and the DoF table is non-empty. This is the
  guard against the extension silently emitting empty tables.
- `test_config_table_model_config` — `@pytest.mark.unit`: field rows for
  `flexcore.config.schema.ModelConfig` include `schema_version` with its
  description.

The PR docs build executes both notebooks pulled from `flex-pse-examples`
(merge gate); nightly re-executes them cache-free as the drift safety net.
Their final assert cells are the pass/fail criterion in both.

## Documentation tasks

This whole milestone is documentation; specifically also:

- `docs/getting_started/ten_minutes.md` — verify it matches the final
  api_freeze walkthrough and links notebook 01 (in `flex-pse-examples`).
- `docs/how_to/build_a_plant.md`, `parameterize_from_data.md` — each becomes a
  thin wrapper pointing at its executed notebook (03 §1). (`schedule_rolling_horizon.md`
  returns in 0.2 alongside M12/M13.)
- `docs/explanation/config_schema.md` — rendered via `flexops-config-table`
  for the whole JSON config tree that drives `build_model` (§2.3): `ModelConfig`,
  `TimeConfig`, `CostingConfig`, `NetworkConfig`/`PlantConfig`, `UnitConfig`,
  `IOVariableSpec`, `SurrogateSpec`.
- `docs/explanation/reported_cost.md` (or the `energy_nomenclature.md` section):
  the reported cost is computed after the solve, never read off the solver's
  objective (see §4b — write it in plain language, no citations on the page).
- Run the `grep -rEn '\bR[0-9]+\b|\bM[0-9]{2}\b|§' docs/` sweep (§0/§4) and fix
  every hit.
- CHANGELOG entry under "Unreleased" (docs system + notebook integration are
  user-visible).

## Definition of Done

- [ ] `docs/conf.py` final: napoleon, autosummary generate, myst_nb, intersphinx (pyomo/idaes/pandas/pydantic), furo, nitpicky + curated ignore list, `NB_EXECUTION_MODE` switch
- [ ] `flexdoc.py` provides `flexops-unit-tables` and `flexops-config-table`; failures are loud, never empty tables
- [ ] `_templates/autosummary/unit_model.rst` in place; **every** public unit model renders generated Variables/Constraints/DoF tables — the SISO/SIDO/DIDO bases and the full physical zoo (Pump, Tank, Separator, Exchanger, ElectrolysisSeparator, ElectrolysisExchanger, ReverseOsmosis, Combustor, BatteryModel, ConstantEnergyIntensityModel); no `Electrolyzer` reference anywhere
- [ ] Reference pages exist for `NetworkBlock` (§3.3), the unit-commitment logic modules (§3.5), and `FlexCosting.evaluate_cost`/`report_cost` (§3.6)
- [ ] `build_model` documented and the JSON config schema (`ModelConfig` tree) rendered via `flexops-config-table` (§2.3)
- [ ] Explanation note: reported cost is computed after the solve, never read off the objective, linkable, and free of internal citations
- [ ] Reference sweep complete: no hand-written variable tables, `grep -rn TODO docs/` is empty, `grep -rEn '\bR[0-9]+\b|\bM[0-9]{2}\b|§' docs/` is empty
- [ ] `docs.yml`/`nightly.yml` check out `flex-pse-examples` at a pinned ref into `docs/examples/` before the Sphinx build
- [ ] Two notebooks present in `flex-pse-examples` at the pinned ref, each ≤ 2-day horizon, fixed seeds, no network, ending in an assert cell; both execute clean as part of this repo's docs build
- [ ] `docs.yml` finalized (PR: cached notebook execution ON, required check; main: cached execution + GitHub Pages deploy); nightly cache-free notebook-execution step present
- [ ] `sphinx-build -W --keep-going` clean in BOTH execution modes — zero warnings
- [ ] `test_flexdoc_tables.py` unit tests pass
- [ ] CHANGELOG updated; PR records implementer's-choice decisions (Pages vs RTD, exact `flex-pse-examples` checkout mechanism)
- [ ] plus the generic DoD in CLAUDE.md
