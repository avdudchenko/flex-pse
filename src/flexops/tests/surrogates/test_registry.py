"""Tests for the flexops.surrogates registry: SURROGATES and surrogate_from_spec."""

import pyomo.environ as pyo
import pytest

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexConfigError
from flexops.core.registration import CoefficientRegistry
from flexops.surrogates import MultilinearSurrogate, surrogate_from_spec

_MULTILINEAR_DATA = {
    "input_variables": {"flow_out": "m^3/hr"},
    "output_variables": {"power_electrical": "kW"},
    "coefficients": {"intercept": 1.0, "flow_out": 0.5},
}


@pytest.mark.unit
def test_surrogate_from_spec_builds_a_multilinear_surrogate():
    """The one implemented class is resolved and validated in one step."""
    spec = SurrogateSpec(
        surrogate_type=SurrogateType.MULTILINEAR, data=_MULTILINEAR_DATA
    )

    surrogate = surrogate_from_spec(spec)

    assert isinstance(surrogate, MultilinearSurrogate)
    assert surrogate.data == _MULTILINEAR_DATA


@pytest.mark.unit
def test_surrogate_from_spec_rejects_constant_intensity():
    """constant_intensity has no class: it fixes a parameter, not a Constraint."""
    spec = SurrogateSpec(surrogate_type=SurrogateType.CONSTANT_INTENSITY, data={})
    with pytest.raises(FlexConfigError, match="constant_intensity"):
        surrogate_from_spec(spec)


@pytest.mark.unit
@pytest.mark.parametrize(
    "surrogate_type",
    [
        SurrogateType.QUADRATIC,
        SurrogateType.EXPONENTIAL,
        SurrogateType.ARIMA,
        SurrogateType.NEURAL_NETWORK,
    ],
)
def test_surrogate_from_spec_stubs_raise_not_implemented(surrogate_type):
    """Every reserved type is registered, but not yet implemented."""
    spec = SurrogateSpec(surrogate_type=surrogate_type, data={})
    with pytest.raises(NotImplementedError, match="MultilinearSurrogate"):
        surrogate_from_spec(spec)


@pytest.mark.unit
def test_coefficient_registry_register_coefficient_success():
    """A valid name/Var pair is stored and retrievable."""
    registry = CoefficientRegistry()
    var = pyo.Var(initialize=0.0)
    registry.register_coefficient("intercept", var)
    assert registry["intercept"] is var
    assert "intercept" in registry
    assert len(registry) == 1
    assert list(registry.items()) == [("intercept", var)]


@pytest.mark.unit
def test_coefficient_registry_register_coefficient_duplicate_raises():
    """Registering the same name twice raises FlexConfigError."""
    registry = CoefficientRegistry()
    registry.register_coefficient("a", pyo.Var(initialize=0.0))
    with pytest.raises(FlexConfigError, match="already registered"):
        registry.register_coefficient("a", pyo.Var(initialize=1.0))


@pytest.mark.unit
def test_coefficient_registry_register_coefficient_non_var_raises():
    """Registering a non-Var value raises FlexConfigError."""
    registry = CoefficientRegistry()
    with pytest.raises(FlexConfigError, match="must be a pyo.Var"):
        registry.register_coefficient("bad", 42)


@pytest.mark.unit
def test_coefficient_registry_register_coefficients_bulk():
    """Bulk registration adds every valid entry."""
    registry = CoefficientRegistry()
    a = pyo.Var(initialize=0.0)
    b = pyo.Var(initialize=1.0)
    registry.register_coefficients({"a": a, "b": b})
    assert len(registry) == 2
    assert registry["a"] is a
    assert registry["b"] is b


@pytest.mark.unit
def test_coefficient_registry_register_coefficients_bulk_raises_on_bad_entry():
    """Bulk registration stops at the first invalid entry."""
    registry = CoefficientRegistry()
    registry.register_coefficients({"a": pyo.Var(initialize=0.0)})
    with pytest.raises(FlexConfigError, match="already registered"):
        registry.register_coefficients(
            {"a": pyo.Var(initialize=1.0), "b": pyo.Var(initialize=2.0)}
        )
    assert len(registry) == 1


@pytest.mark.unit
def test_coefficient_registry_fix_unfix():
    """fix() locks every Var; unfix() releases them."""
    m = pyo.ConcreteModel()
    registry = CoefficientRegistry()
    a = pyo.Var(initialize=0.0)
    b = pyo.Var(initialize=1.0)
    m.add_component("a", a)
    m.add_component("b", b)
    registry.register_coefficients({"a": a, "b": b})

    a.fix()
    registry.fix()
    assert a.is_fixed() and b.is_fixed()

    b.unfix()
    registry.unfix()
    assert not a.is_fixed() and not b.is_fixed()
