"""Tests for the flexops.surrogates registry: SURROGATES and surrogate_from_spec."""

import pytest

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexConfigError
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
