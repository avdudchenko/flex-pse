"""Ramp-rate limiting constraint-body tests (M08, §3.5)."""

import logging

import pyomo.environ as pyo
import pytest
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError
from flexops.logic import add_ramp_rate
from flexops.testing import dummy_time_block
from flexops.unit_models import Pump

_N = 6
_RATE_UNITS = pyunits.kW / pyunits.hr


def _con_satisfied(condata, tol: float = 1e-9) -> bool:
    """Whether a constraint body lies within its (lower, upper) bounds."""
    body = pyo.value(condata.body)
    lower, upper = condata.lower, condata.upper
    ok = True
    if lower is not None:
        ok = ok and pyo.value(lower) <= body + tol
    if upper is not None:
        ok = ok and body <= pyo.value(upper) + tol
    return ok


def _unit_with_var():
    m = dummy_time_block(_N)
    m.unit = Pump(property_package=m.properties)
    return m, m.unit, m.unit.power_electrical


def _set_values(var, values):
    for t, value in enumerate(values):
        var[t].set_value(value)


@pytest.mark.unit
def test_ramp_def_ties_tracked_to_var():
    """<name>_def holds tracked[t] == var[t]; breaks when perturbed independently."""
    m, unit, var = _unit_with_var()
    tracked, _, _ = add_ramp_rate(var, ramp_up=4.0, ramp_rate_units=_RATE_UNITS)
    def_con = unit.find_component("power_electrical_ramp_def")

    _set_values(var, range(_N))
    _set_values(tracked, range(_N))
    assert all(_con_satisfied(c) for c in def_con.values())

    tracked[2].set_value(99.0)
    assert not _con_satisfied(def_con[2])


@pytest.mark.unit
def test_ramp_up_only_bounds_increase():
    """window=1: an increase beyond ramp_up*dt violates; ramp_down is absent."""
    m, unit, var = _unit_with_var()
    tracked, up, down = add_ramp_rate(var, ramp_up=4.0, ramp_rate_units=_RATE_UNITS)

    assert down is None
    assert unit.find_component("power_electrical_ramp_ramp_down") is None
    assert set(up.index_set()) == {t for t in range(_N) if t >= 1}

    # dt = 0.25 hr, so the per-step limit is 4 kW/hr * 0.25 hr = 1 kW.
    _set_values(tracked, [0.0, 1.0, 2.0, 3.0, 3.0, 10.0])
    assert all(_con_satisfied(up[t]) for t in range(1, 5))
    assert not _con_satisfied(up[5])


@pytest.mark.unit
def test_ramp_down_only_bounds_decrease():
    """window=1: a decrease beyond ramp_down*dt violates; ramp_up is absent."""
    m, unit, var = _unit_with_var()
    tracked, up, down = add_ramp_rate(var, ramp_down=4.0, ramp_rate_units=_RATE_UNITS)

    assert up is None
    assert unit.find_component("power_electrical_ramp_ramp_up") is None

    _set_values(tracked, [10.0, 9.0, 8.0, 7.0, 7.0, 0.0])
    assert all(_con_satisfied(down[t]) for t in range(1, 5))
    assert not _con_satisfied(down[5])


@pytest.mark.unit
def test_ramp_both_directions():
    """Both directions active simultaneously: each catches its own violation."""
    m, unit, var = _unit_with_var()
    tracked, up, down = add_ramp_rate(
        var, ramp_up=4.0, ramp_down=4.0, ramp_rate_units=_RATE_UNITS
    )

    _set_values(tracked, [0.0, 1.0, 2.0, 1.0, 2.0, 1.0])
    assert all(_con_satisfied(c) for c in up.values())
    assert all(_con_satisfied(c) for c in down.values())

    tracked[3].set_value(5.0)  # +3 from tracked[2]=2, exceeds the 1 kW limit
    assert not _con_satisfied(up[3])

    tracked[3].set_value(1.0)
    tracked[1].set_value(-5.0)  # -5 from tracked[0]=0, exceeds the 1 kW limit
    assert not _con_satisfied(down[1])


@pytest.mark.unit
def test_ramp_excludes_t0():
    """window=1: t=0 has no predecessor and is left out of the ramp index."""
    m, unit, var = _unit_with_var()
    _, up, _ = add_ramp_rate(var, ramp_up=4.0, ramp_rate_units=_RATE_UNITS)
    assert 0 not in set(up.index_set())


@pytest.mark.unit
def test_ramp_window_excludes_points_without_full_history():
    """window=w>1: the ramp index excludes every t < w."""
    m, unit, var = _unit_with_var()
    w = 3
    _, up, _ = add_ramp_rate(var, ramp_up=4.0, ramp_rate_units=_RATE_UNITS, window=w)
    assert set(up.index_set()) == {t for t in range(_N) if t >= w}


@pytest.mark.unit
def test_ramp_window_is_a_net_change_check_not_a_smoothed_step_check():
    """window=w>1 compares endpoints w apart; a mid-window spike that nets to
    zero at the endpoints does not violate, but a genuine net violation does.
    """
    m, unit, var = _unit_with_var()
    w = 3
    tracked, up, _ = add_ramp_rate(
        var, ramp_up=4.0, ramp_rate_units=_RATE_UNITS, window=w
    )
    # dt = 0.25 hr, so the window limit is 4 kW/hr * 3 * 0.25 hr = 3 kW.
    _set_values(tracked, [0.0, 0.0, 0.0, 10.0, 10.0, 0.0])

    assert not _con_satisfied(up[3])  # tracked[3]-tracked[0] = 10 kW > 3 kW
    assert _con_satisfied(up[5])  # tracked[5]-tracked[2] = 0 kW, despite the spike


@pytest.mark.unit
def test_ramp_requires_at_least_one_direction():
    """Neither ramp_up nor ramp_down given raises FlexConfigError."""
    m, unit, var = _unit_with_var()
    with pytest.raises(FlexConfigError):
        add_ramp_rate(var)


@pytest.mark.unit
@pytest.mark.parametrize("kwargs", [{"ramp_up": -1.0}, {"ramp_down": -1.0}])
def test_ramp_negative_limit_raises(kwargs):
    """A negative ramp_up/ramp_down raises FlexConfigError."""
    m, unit, var = _unit_with_var()
    with pytest.raises(FlexConfigError):
        add_ramp_rate(var, ramp_rate_units=_RATE_UNITS, **kwargs)


@pytest.mark.unit
def test_ramp_window_below_one_raises():
    """window < 1 raises FlexConfigError."""
    m, unit, var = _unit_with_var()
    with pytest.raises(FlexConfigError):
        add_ramp_rate(var, ramp_up=4.0, ramp_rate_units=_RATE_UNITS, window=0)


@pytest.mark.unit
def test_ramp_up_without_units_raises():
    """ramp_up given without ramp_rate_units raises FlexConfigError."""
    m, unit, var = _unit_with_var()
    with pytest.raises(FlexConfigError):
        add_ramp_rate(var, ramp_up=4.0)


@pytest.mark.unit
def test_ramp_reuse_on_same_var_logs_warning(caplog):
    """Attaching a second ramp to the same Var is allowed but logs a warning."""
    m, unit, var = _unit_with_var()
    add_ramp_rate(var, ramp_up=4.0, ramp_rate_units=_RATE_UNITS)

    with caplog.at_level(logging.WARNING, logger="flexops.logic.ramp"):
        add_ramp_rate(
            var,
            ramp_up=2.0,
            ramp_rate_units=_RATE_UNITS,
            name="power_electrical_slow_ramp",
            window=3,
        )

    assert any(
        "already has a ramp rate attached" in record.message
        for record in caplog.records
    )
