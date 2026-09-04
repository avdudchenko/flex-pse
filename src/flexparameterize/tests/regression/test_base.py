"""Tests for the Regressor Protocol and FitResult dataclass."""

import pandas as pd
import pytest

from flexcore.exceptions import FlexDataError
from flexparameterize.regression import ConstantIntensityRegressor
from flexparameterize.regression.base import FitResult, Regressor


class _NotARegressor:
    """Missing to_surrogate_spec -- should fail the structural check."""

    def fit(self, X, y):
        return self

    def to_fit_result(self):
        return None


@pytest.mark.unit
def test_protocol_conformance():
    """A conforming class isinstance-checks true; calling it returns real types."""
    assert isinstance(ConstantIntensityRegressor(), Regressor)
    assert not isinstance(_NotARegressor(), Regressor)

    X = pd.DataFrame({"flow": [1.0, 2.0, 4.0]})
    y = pd.DataFrame({"power": [0.5, 1.0, 2.0]})
    regressor = ConstantIntensityRegressor().fit(X, y)

    result = regressor.to_fit_result()
    assert isinstance(result, FitResult)
    assert result.coefficients == {"energy_intensity": pytest.approx(0.5)}
    assert result.metrics["r2"] == pytest.approx(1.0)
    assert result.n_samples == 3
    assert result.data_window == (0, 2)


@pytest.mark.unit
def test_to_fit_result_before_fit_raises():
    """Calling to_fit_result before fit raises FlexDataError."""
    with pytest.raises(FlexDataError):
        ConstantIntensityRegressor().to_fit_result()
