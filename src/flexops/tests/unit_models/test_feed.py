"""Feed(OpsBlockData): a boundary source -- zero inlets, N outlets (§3.2, §3.4)."""

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits
from pyomo.network import Port

from flexcore.exceptions import FlexConfigError
from flexops.core.registration import BoundaryKind
from flexops.testing import (
    UnitModelTestHarness,
    dummy_gas_time_block,
    dummy_time_block,
)
from flexops.unit_models import Feed

_DISPATCH = {0: 1.0, 1: 2.0, 2: 3.0}


def _feed(n: int = 3, **kwargs):
    """Build a Feed on an ``n``-point aqueous ``dummy_time_block``."""
    m = dummy_time_block(n)
    m.raw_water = Feed(property_package=m.properties, **kwargs)
    return m, m.raw_water


def _gas_feed(n: int = 3, **kwargs):
    """Build a Feed on an ``n``-point ``dummy_gas_time_block``."""
    m = dummy_gas_time_block(n)
    m.raw_gas = Feed(property_package=m.properties, **kwargs)
    return m, m.raw_gas


def _registered(unit, role: str) -> set[str]:
    """Return ``{"<state block>.<var>"}`` for IO variables in ``role``."""
    return {
        f"{rec.var.parent_block().local_name}.{rec.name}"
        for rec in unit._io_registry.io_variables
        if rec.role == role
    }


# -- harness ----------------------------------------------------------------


class TestFeedAqueousDispatched(UnitModelTestHarness):
    """One aqueous outlet whose withdrawal is driven by an external series.

    Fixing ``withdrawal`` closes every degree of freedom: the single outlet
    flow is then pinned by ``eq_withdrawal``.
    """

    expected_dof = 0
    expected_solution = {
        "withdrawal[0]": 1.0,
        "withdrawal[2]": 3.0,
        "flow_out_a[0]": 1.0,
        "flow_out_a[2]": 3.0,
    }

    def configure(self):
        m, unit = _feed(3, outlet_names=("a",))
        unit.set_external_dispatch(unit.withdrawal, _DISPATCH)
        return m, unit


class TestFeedAqueousTwoOutlets(UnitModelTestHarness):
    """Two aqueous outlets, undispatched.

    The flow-only aqueous package registers no ``role="input"`` state, so
    nothing is fixed: ``eq_withdrawal`` relates ``withdrawal`` and the two
    outlet flows over 3 time points -- ``9 - 3 == 6`` degrees of freedom.
    """

    expected_dof = 6

    def configure(self):
        return _feed(3, outlet_names=("a", "b"))


class TestFeedGas(UnitModelTestHarness):
    """Two gas outlets: the outlet-state ties hold outlet ``b`` at outlet ``a``.

    The reference outlet's pressure and temperature are registered inputs and
    get fixed; outlet ``b``'s are pinned by the ties. Unfixed variables in the
    9 active constraints are ``withdrawal`` plus two outlet flows plus outlet
    ``b``'s pressure/temperature -- ``15 - 9 == 6``.
    """

    expected_dof = 6

    def configure(self):
        return _gas_feed(3, outlet_names=("a", "b"))


class TestFeedHorizonWithdrawal(UnitModelTestHarness):
    """A horizon-basis limit is degree-of-freedom neutral.

    ``withdrawal_total`` is one free scalar Var and ``eq_withdrawal_total`` one
    scalar equality determining it, so the accounting is unchanged from
    :class:`TestFeedAqueousDispatched`. The total is
    ``(1 + 2 + 3) m**3/hr * 15 min``.
    """

    expected_dof = 0
    expected_solution = {"withdrawal[2]": 3.0, "withdrawal_total": 1.5}

    def configure(self):
        m, unit = _feed(
            3,
            outlet_names=("a",),
            max_withdrawal=100 * pyunits.m**3,
            withdrawal_basis="horizon",
        )
        unit.set_external_dispatch(unit.withdrawal, _DISPATCH)
        return m, unit


# -- topology ---------------------------------------------------------------


@pytest.mark.unit
def test_feed_builds_one_port_per_outlet_name_and_no_inlet():
    """A source has ``outlet_<name>`` ports only -- never an inlet."""
    _, unit = _feed(3, outlet_names=("north", "south"))

    ports = {p.local_name for p in unit.component_objects(Port, descend_into=False)}
    assert ports == {"outlet_north", "outlet_south"}
    for port in ports:
        assert len(list(unit.find_component(port).values())) > 0
    assert unit.find_component("inlet") is None
    assert unit.find_component("inlet_state") is None


@pytest.mark.unit
def test_feed_declares_no_power():
    """A boundary block has no energy relation -- it draws and exports nothing."""
    _, unit = _feed(3)
    assert unit._io_registry.power == []
    assert unit._io_registry.fuel == []
    assert unit.find_component("power_electrical") is None
    assert unit.find_component("power_thermal") is None


@pytest.mark.unit
def test_feed_registers_reference_outlet_states_as_the_boundary_condition():
    """The reference outlet's non-flow states are inputs; the others' are outputs."""
    _, unit = _gas_feed(3, outlet_names=("a", "b"))

    inputs = _registered(unit, "input")
    for state_var in ("pressure", "temperature"):
        assert f"outlet_a_state.{state_var}" in inputs
        assert f"outlet_b_state.{state_var}" not in inputs

    outputs = _registered(unit, "output")
    for state_var in ("pressure", "temperature"):
        assert f"outlet_b_state.{state_var}" in outputs
    # add_stream_ports registers every outlet's flow as a result, not an input.
    assert "outlet_a_state.flow_vol_phase" in outputs


# -- withdrawal balance -----------------------------------------------------


@pytest.mark.unit
def test_feed_withdrawal_balance_body():
    """``withdrawal[t]`` equals the sum of the outlet flows -- nothing else."""
    _, unit = _feed(3, outlet_names=("a", "b", "c"))
    for t in range(3):
        unit.withdrawal[t].fix(6.0)
        unit.flow_out_a[t].fix(1.0)
        unit.flow_out_b[t].fix(2.0)
        unit.flow_out_c[t].fix(3.0)
        assert pyo.value(unit.eq_withdrawal[t].body) == pytest.approx(0.0, abs=1e-9)
        unit.flow_out_c[t].fix(4.0)
        assert pyo.value(unit.eq_withdrawal[t].body) == pytest.approx(-1.0, abs=1e-9)


@pytest.mark.unit
def test_feed_withdrawal_is_a_var_so_it_can_be_dispatched():
    """``withdrawal`` is a fixable Var, not an Expression."""
    m, unit = _feed(3)
    assert isinstance(unit.withdrawal, pyo.Var)
    unit.set_external_dispatch(unit.withdrawal, _DISPATCH)
    for t in m.time_block.time_index:
        assert unit.withdrawal[t].fixed
        assert pyo.value(unit.withdrawal[t]) == pytest.approx(_DISPATCH[t])


@pytest.mark.unit
def test_feed_ties_every_other_outlet_to_the_reference_outlet():
    """One source, one condition: outlet ``b``'s states equal outlet ``a``'s."""
    _, unit = _gas_feed(3, outlet_names=("a", "b"))
    for state_var in ("pressure", "temperature"):
        constraint = unit.find_component(f"outlet_state_equality_{state_var}")
        assert constraint is not None, state_var
        assert len(constraint) == 3

    for t in range(3):
        unit.outlet_a_state.pressure[t].fix(101325.0)
        unit.outlet_b_state.pressure[t].fix(101325.0)
        assert pyo.value(
            unit.outlet_state_equality_pressure[t, "b"].body
        ) == pytest.approx(0.0, abs=1e-9)
        unit.outlet_b_state.pressure[t].fix(150000.0)
        assert pyo.value(
            unit.outlet_state_equality_pressure[t, "b"].body
        ) == pytest.approx(150000.0 - 101325.0, abs=1e-9)


@pytest.mark.unit
def test_feed_builds_no_state_ties_for_a_single_outlet():
    """With one outlet there is nothing to tie."""
    _, unit = _gas_feed(3, outlet_names=("a",))
    assert unit.find_component("outlet_state_equality_pressure") is None


# -- withdrawal limits ------------------------------------------------------


@pytest.mark.unit
def test_feed_limits_are_built_only_when_configured():
    """Limit Params/Constraints exist iff their config option was given."""
    _, bare = _feed(3)
    for name in (
        "withdrawal_min",
        "withdrawal_max",
        "withdrawal_min_limit",
        "withdrawal_max_limit",
    ):
        assert bare.find_component(name) is None

    _, bounded = _feed(
        3,
        min_withdrawal=1 * pyunits.m**3 / pyunits.hr,
        max_withdrawal=9 * pyunits.m**3 / pyunits.hr,
    )
    for name in (
        "withdrawal_min",
        "withdrawal_max",
        "withdrawal_min_limit",
        "withdrawal_max_limit",
    ):
        assert bounded.find_component(name) is not None


@pytest.mark.unit
def test_feed_limit_bodies_and_mutable_params():
    """The limits bound ``withdrawal``, and rewriting a Param moves the bound."""
    _, unit = _feed(3, max_withdrawal=9 * pyunits.m**3 / pyunits.hr)
    for t in range(3):
        assert pyo.value(unit.withdrawal_max[t]) == pytest.approx(9.0)
        unit.withdrawal[t].fix(4.0)
        limit = unit.withdrawal_max_limit[t]
        assert pyo.value(limit.body) == pytest.approx(4.0)
        assert pyo.value(limit.upper) == pytest.approx(9.0)

    # The Param is mutable, so rewriting it moves the bound in place.
    unit.withdrawal_max[1].set_value(2.0)
    assert pyo.value(unit.withdrawal_max_limit[1].upper) == pytest.approx(2.0)
    assert pyo.value(unit.withdrawal_max_limit[0].upper) == pytest.approx(9.0)


@pytest.mark.unit
def test_feed_limit_params_are_converted_to_the_flow_basis():
    """A limit given in other units is converted onto the withdrawal's basis."""
    from pyomo.util.check_units import assert_units_consistent

    _, unit = _feed(3, max_withdrawal=1000 * pyunits.L / pyunits.hr)
    assert_units_consistent(unit)
    for t in range(3):
        assert pyo.value(unit.withdrawal_max[t]) == pytest.approx(1.0)


@pytest.mark.unit
def test_feed_period_limit_accepts_one_value_per_period():
    """A length-T series sets each period's limit, converted onto the flow basis."""
    rate = pyunits.m**3 / pyunits.hr
    _, unit = _feed(
        3, max_withdrawal=[1 * rate, 2 * rate, 3000 * pyunits.L / pyunits.hr]
    )
    values = [pyo.value(unit.withdrawal_max[t]) for t in range(3)]
    assert values == pytest.approx([1.0, 2.0, 3.0])


# -- horizon-basis withdrawal limits ----------------------------------------


@pytest.mark.unit
def test_feed_horizon_basis_builds_a_scalar_total():
    """``withdrawal_basis="horizon"`` bounds one scalar total, not each period."""
    _, unit = _feed(3, max_withdrawal=100 * pyunits.m**3, withdrawal_basis="horizon")

    assert isinstance(unit.withdrawal_total, pyo.Var)
    assert not unit.withdrawal_total.is_indexed()
    assert unit.find_component("eq_withdrawal_total") is not None
    assert not unit.withdrawal_max.is_indexed()
    assert not unit.withdrawal_max_limit.is_indexed()


@pytest.mark.unit
def test_feed_period_basis_is_the_default_and_builds_no_total():
    """Without ``withdrawal_basis`` the limits stay per-period, as before."""
    _, unit = _feed(3, max_withdrawal=9 * pyunits.m**3 / pyunits.hr)

    assert unit.find_component("withdrawal_total") is None
    assert unit.find_component("eq_withdrawal_total") is None
    assert unit.withdrawal_max.is_indexed()


@pytest.mark.unit
def test_feed_horizon_total_is_the_time_integral_of_withdrawal():
    """``withdrawal_total`` equals ``sum_t withdrawal[t] * dt`` -- 15-min steps."""
    _, unit = _feed(3, max_withdrawal=100 * pyunits.m**3, withdrawal_basis="horizon")
    for t, value in _DISPATCH.items():
        unit.withdrawal[t].fix(value)

    # (1 + 2 + 3) m**3/hr * 0.25 hr == 1.5 m**3.
    unit.withdrawal_total.set_value(1.5)
    assert pyo.value(unit.eq_withdrawal_total.body) == pytest.approx(0.0, abs=1e-9)
    unit.withdrawal_total.set_value(2.0)
    assert pyo.value(unit.eq_withdrawal_total.body) == pytest.approx(0.5, abs=1e-9)


@pytest.mark.unit
def test_feed_horizon_limit_params_are_scalar_and_mutable():
    """The horizon limits bound the total, and rewriting a Param moves them."""
    _, unit = _feed(
        3,
        min_withdrawal=1 * pyunits.m**3,
        max_withdrawal=9 * pyunits.m**3,
        withdrawal_basis="horizon",
    )
    unit.withdrawal_total.set_value(5.0)
    assert pyo.value(unit.withdrawal_max_limit.body) == pytest.approx(5.0)
    assert pyo.value(unit.withdrawal_max_limit.upper) == pytest.approx(9.0)
    assert pyo.value(unit.withdrawal_min_limit.lower) == pytest.approx(1.0)

    # The Param is mutable, so rewriting it moves the bound in place.
    unit.withdrawal_max.set_value(6.0)
    assert pyo.value(unit.withdrawal_max_limit.upper) == pytest.approx(6.0)


@pytest.mark.unit
def test_feed_horizon_limits_are_converted_to_one_basis():
    """Limits given in mixed units land on the upper limit's basis."""
    from pyomo.util.check_units import assert_units_consistent

    _, unit = _feed(
        3,
        min_withdrawal=1000 * pyunits.L,
        max_withdrawal=5 * pyunits.m**3,
        withdrawal_basis="horizon",
    )
    assert_units_consistent(unit)
    assert pyo.value(unit.withdrawal_min) == pytest.approx(1.0)
    assert pyo.value(unit.withdrawal_max) == pytest.approx(5.0)


@pytest.mark.unit
def test_feed_horizon_basis_leaves_withdrawal_time_indexed():
    """The metered Var is untouched: costing, aggregation and dispatch still work."""
    m, unit = _feed(3, max_withdrawal=100 * pyunits.m**3, withdrawal_basis="horizon")

    assert unit.withdrawal.is_indexed()
    assert unit.withdrawal.index_set() is m.time_block.time_index
    (record,) = unit._io_registry.boundary
    assert record.kind is BoundaryKind.FEED
    assert record.var is unit.withdrawal

    unit.set_external_dispatch(unit.withdrawal, _DISPATCH)
    for t in m.time_block.time_index:
        assert unit.withdrawal[t].fixed


@pytest.mark.unit
def test_feed_withdrawal_basis_alone_builds_nothing():
    """A basis with no limit configured is not a limit -- nothing is built."""
    _, unit = _feed(3, withdrawal_basis="horizon")
    for name in (
        "withdrawal_total",
        "eq_withdrawal_total",
        "withdrawal_min",
        "withdrawal_max",
    ):
        assert unit.find_component(name) is None


# -- boundary registration --------------------------------------------------


@pytest.mark.unit
def test_feed_registers_its_withdrawal_as_a_feed_boundary_flow():
    """One boundary record, keyed FEED, carrying the withdrawal Var."""
    _, unit = _feed(3)
    (record,) = unit._io_registry.boundary
    assert record.kind is BoundaryKind.FEED
    assert record.var is unit.withdrawal
    assert record.name == "withdrawal"
    assert not unit._io_registry.is_empty()


@pytest.mark.unit
def test_feed_resource_name_defaults_to_the_block_local_name():
    """``resource_name=None`` falls back to the Pyomo block name."""
    _, unit = _feed(3)
    (record,) = unit._io_registry.boundary
    assert record.resource == "raw_water"
    assert unit.local_name == "raw_water"


@pytest.mark.unit
def test_feed_explicit_resource_name_overrides_without_renaming_the_block():
    """The aggregation key is independent of the block name."""
    _, unit = _feed(3, resource_name="city_water")
    (record,) = unit._io_registry.boundary
    assert record.resource == "city_water"
    assert unit.local_name == "raw_water"


@pytest.mark.unit
def test_feed_rejects_an_empty_resource_name():
    """An explicit empty ``resource_name`` is an error, not a silent fallback."""
    with pytest.raises(FlexConfigError) as excinfo:
        _feed(3, resource_name="")
    assert excinfo.value.field == "resource"


# -- config rejection -------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("outlet_names", [(), ("a", "a"), ("a", ""), ("a", 1)])
def test_feed_rejects_bad_outlet_names(outlet_names):
    """Empty, duplicated, or non-string ``outlet_names`` raise, naming the field."""
    with pytest.raises(FlexConfigError) as excinfo:
        _feed(3, outlet_names=outlet_names)
    assert excinfo.value.field == "outlet_names"


@pytest.mark.unit
def test_feed_rejects_a_rate_limit_on_the_horizon_basis():
    """A rate given on the horizon basis is a units error, not a silent rescale."""
    with pytest.raises(FlexConfigError) as excinfo:
        _feed(
            3,
            max_withdrawal=100 * pyunits.m**3 / pyunits.hr,
            withdrawal_basis="horizon",
        )
    assert excinfo.value.field == "max_withdrawal"


@pytest.mark.unit
def test_feed_rejects_a_quantity_limit_on_the_period_basis():
    """The mirror slip: a horizon quantity left on the default period basis."""
    with pytest.raises(FlexConfigError) as excinfo:
        _feed(3, max_withdrawal=2400 * pyunits.m**3)
    assert excinfo.value.field == "max_withdrawal"


@pytest.mark.unit
def test_feed_rejects_an_unknown_withdrawal_basis():
    """``withdrawal_basis`` is one of the two declared bases, naming the field."""
    with pytest.raises(FlexConfigError) as excinfo:
        _feed(3, withdrawal_basis="daily")
    assert excinfo.value.field == "withdrawal_basis"


@pytest.mark.unit
def test_feed_price_without_a_costing_package_raises():
    """A price that could not be billed is rejected, never silently dropped."""
    with pytest.raises(FlexConfigError) as excinfo:
        _feed(3, price=0.5)
    assert excinfo.value.field == "price"


# -- integration with the rest of the library -------------------------------


@pytest.mark.unit
def test_feed_is_in_the_unit_model_registry():
    """``Feed`` is reachable from ``flexops`` and its unit-model registry."""
    import flexops
    from flexops import unit_models

    assert "Feed" in unit_models.__all__
    assert flexops.Feed is Feed


@pytest.mark.component
@pytest.mark.needs_highs
def test_feed_pump_product_plant_aggregates_both_boundaries():
    """A plant bracketed by a Feed and a Product totals both boundary streams."""
    from pyomo.network import Arc

    from flexcore.exceptions import FlexSolverError
    from flexcore.solvers import get_solver
    from flexops import PlantBlock
    from flexops.unit_models import Product, Pump

    m = dummy_time_block(3)
    m.plant = PlantBlock(time_block=m.time_block)
    m.plant.raw_water = Feed(property_package=m.properties, resource_name="raw_water")
    m.plant.pump = Pump(
        property_package=m.properties,
        energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
    )
    m.plant.potable = Product(
        property_package=m.properties, resource_name="potable_water"
    )
    m.plant.feed_to_pump = Arc(
        source=m.plant.raw_water.outlet_a, destination=m.plant.pump.inlet
    )
    m.plant.pump_to_product = Arc(
        source=m.plant.pump.outlet, destination=m.plant.potable.inlet_a
    )
    m.plant._build_aggregates()
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    m.plant.raw_water.set_external_dispatch(
        m.plant.raw_water.withdrawal, {0: 4.0, 1: 5.0, 2: 6.0}
    )

    try:
        solver = get_solver(model=m)
    except FlexSolverError as exc:
        pytest.skip(f"flexcore.solvers.get_solver not available: {exc}")
    pyo.assert_optimal_termination(solver.solve(m))

    for t in m.time_block.time_index:
        expected = {0: 4.0, 1: 5.0, 2: 6.0}[t]
        assert pyo.value(m.plant.total_feed["raw_water", t]) == pytest.approx(
            expected, rel=1e-6
        )
        assert pyo.value(m.plant.total_product["potable_water", t]) == pytest.approx(
            expected, rel=1e-6
        )
