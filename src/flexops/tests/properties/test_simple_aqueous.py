"""Tests for SimpleAqueousFlow: the minimal flow-carrying property package.

Mirrors the test skeleton of ``test_simple_gas.py``; package-specific extras
(the opt-in pressure/temperature) come last.

State variables are indexed over time directly (``flow_vol_phase[t, phase]``);
the owning unit passes a ``time_index`` Set and gets a single scalar state block
rather than one block per time point.
"""

import pyomo.environ as pyo
import pytest
from idaes.core import Component, LiquidPhase
from pyomo.environ import units as pyunits
from pyomo.environ import value
from pyomo.util.check_units import assert_units_consistent, assert_units_equivalent

from flexcore.exceptions import FlexConfigError
from flexops.properties.simple_aqueous import SimpleAqueousFlow

TIMES = [0, 1, 2]


@pytest.fixture
def model():
    """A ConcreteModel with a flow-only package and a 3-point time set."""
    m = pyo.ConcreteModel()
    m.props = SimpleAqueousFlow()
    m.time = pyo.Set(initialize=TIMES, ordered=True)
    return m


@pytest.mark.unit
def test_build_parameter_and_state_block(model):
    """A state block builds flow_vol_phase indexed by (time, phase) in m^3/hr."""
    model.state = model.props.build_state_block(time_index=model.time)
    flow = model.state.flow_vol_phase
    assert set(flow.index_set()) == {(t, "Liq") for t in TIMES}
    assert_units_equivalent(flow[0, "Liq"], pyunits.m**3 / pyunits.hr)


@pytest.mark.unit
def test_missing_time_index_raises(model):
    """Building a state block without a time_index is a config error."""
    with pytest.raises(FlexConfigError, match="time_index"):
        model.state = model.props.build_state_block()


@pytest.mark.unit
def test_phase_and_component(model):
    """The package auto-builds the Liq phase and the H2O component."""
    assert isinstance(model.props.Liq, LiquidPhase)
    assert isinstance(model.props.H2O, Component)
    assert list(model.props.phase_list) == ["Liq"]
    assert list(model.props.component_list) == ["H2O"]


@pytest.mark.unit
def test_define_state_vars(model):
    """define_state_vars exposes only flow_vol_phase by default."""
    model.state = model.props.build_state_block(time_index=model.time)
    assert set(model.state.define_state_vars().keys()) == {"flow_vol_phase"}


@pytest.mark.unit
def test_state_var_units(model):
    """Flow carries the expected units."""
    model.state = model.props.build_state_block(time_index=model.time)
    assert_units_equivalent(
        model.state.flow_vol_phase[0, "Liq"], pyunits.m**3 / pyunits.hr
    )


@pytest.mark.unit
def test_state_var_domains(model):
    """flow_vol_phase is non-negative."""
    model.state = model.props.build_state_block(time_index=model.time)
    assert model.state.flow_vol_phase[0, "Liq"].domain is pyo.NonNegativeReals


@pytest.mark.unit
def test_metadata_properties_and_units(model):
    """Metadata declares the three properties and the five base default units."""
    meta = model.props.get_metadata()
    supported = {p.name for p in meta.properties.list_supported_properties()}
    assert {"flow_vol_phase", "pressure", "temperature"} <= supported
    assert "dens_mass" not in supported
    assert meta.default_units["time"] == pyunits.hr
    assert meta.default_units["length"] == pyunits.m
    assert meta.default_units["mass"] == pyunits.kg
    assert meta.default_units["amount"] == pyunits.mol
    assert meta.default_units["temperature"] == pyunits.K


@pytest.mark.unit
def test_define_display_vars(model):
    """Display vars mirror the state vars."""
    model.state = model.props.build_state_block(time_index=model.time)
    state_block = model.state
    assert state_block.define_display_vars() == state_block.define_state_vars()


@pytest.mark.unit
def test_units_consistent(model):
    """The state block is dimensionally consistent."""
    model.state = model.props.build_state_block(time_index=model.time)
    assert_units_consistent(model.state)


@pytest.mark.unit
def test_initialize_fixes_and_releases():
    """initialize with hold_state fixes the free state vars; release unfixes."""
    m = pyo.ConcreteModel()
    m.props = SimpleAqueousFlow(has_pressure=True)
    m.time = pyo.Set(initialize=TIMES, ordered=True)
    m.state = m.props.build_state_block(time_index=m.time)
    state_data = [
        vd for var in m.state.define_state_vars().values() for vd in var.values()
    ]
    flags = m.state.initialize(hold_state=True)
    assert all(vd.fixed for vd in state_data)
    m.state.release_state(flags)
    assert not any(vd.fixed for vd in state_data)


@pytest.mark.unit
def test_fix_initialization_states(model):
    """fix_initialization_states fixes every state variable."""
    model.state = model.props.build_state_block(time_index=model.time)
    model.state.fix_initialization_states()
    for var in model.state.define_state_vars().values():
        for vd in var.values():
            assert vd.fixed is True


# -- package-specific extras: the opt-in intensive states --------------------
@pytest.mark.unit
def test_get_flow_basis_var_name(model):
    """get_flow_basis_var_name names the extensive flow state variable."""
    assert model.props.get_flow_basis_var_name() == "flow_vol_phase"


@pytest.mark.unit
def test_optional_pressure_temperature():
    """Pressure/temperature are opt-in intensive state vars with correct units."""
    m = pyo.ConcreteModel()
    m.props = SimpleAqueousFlow(has_pressure=True, has_temperature=True)
    m.time = pyo.Set(initialize=TIMES, ordered=True)
    m.state = m.props.build_state_block(time_index=m.time)
    state_block = m.state
    assert set(state_block.define_state_vars().keys()) == {
        "flow_vol_phase",
        "pressure",
        "temperature",
    }
    assert state_block.pressure[0].domain is pyo.PositiveReals
    assert state_block.temperature[0].domain is pyo.PositiveReals
    assert_units_equivalent(state_block.pressure[0], pyunits.Pa)
    assert_units_equivalent(state_block.temperature[0], pyunits.K)
    assert_units_consistent(state_block)


# --- test on-demand properties
@pytest.mark.unit
def test_on_demand_properties(model):
    """A state block builds flow_vol_phase indexed by (time, phase) in m^3/hr."""
    model.state = model.props.build_state_block(time_index=model.time)
    flow = model.state.flow_vol_phase
    flow[0, "Liq"].fix(1)
    mass_flow = model.state.flow_mass_phase
    dens_mass_phase = model.state.dens_mass_phase
    temp = model.state.temperature
    pressure = model.state.pressure
    assert set(mass_flow.index_set()) == {(t, "Liq") for t in TIMES}
    assert set(dens_mass_phase.index_set()) == {(t, "Liq") for t in TIMES}

    assert set(temp.index_set()) == {t for t in TIMES}
    assert set(pressure.index_set()) == {t for t in TIMES}

    assert_units_equivalent(mass_flow[0, "Liq"], pyunits.kg / pyunits.hr)
    assert_units_equivalent(dens_mass_phase[0, "Liq"], pyunits.kg / pyunits.m**3)
    assert pytest.approx(dens_mass_phase[0, "Liq"].value, rel=1e-6) == 1000
    assert pytest.approx(value(mass_flow[0, "Liq"]), rel=1e-6) == 1000


@pytest.mark.unit
def test_wrong_density_units():
    """A state block builds flow_vol_phase indexed by (time, phase) in m^3/hr."""
    m = pyo.ConcreteModel()
    m.props = SimpleAqueousFlow()
    with pytest.raises(TypeError):
        m.props = SimpleAqueousFlow(density=1000)
    with pytest.raises(TypeError):
        m.props = SimpleAqueousFlow(density=1000 * pyunits.kg / pyunits.m)
