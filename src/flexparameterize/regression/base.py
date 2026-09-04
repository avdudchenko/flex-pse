"""The pluggable-regressor seam: ``Regressor`` Protocol and ``FitResult``.

Every regressor fits on instance attributes and reduces its own fit to a
:class:`FitResult` for provenance, so a new regressor is "implement the
protocol, register the name" -- no pipeline changes (see
:func:`flexparameterize.regression.get_regressor`).
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from flexcore.config.schema import SurrogateSpec


@dataclass(frozen=True)
class FitResult:
    """The common shape every regressor's fit reduces to, for provenance.

    Attributes:
        coefficients: Coefficient name -> value (``"intercept"`` when present).
        metrics: At minimum ``{"r2": ..., "rmse": ...}``.
        n_samples: Number of rows the fit used.
        data_window: ``(first, last)`` index value of the fitted rows.
    """

    coefficients: dict[str, float]
    metrics: dict[str, float]
    n_samples: int
    data_window: tuple


@runtime_checkable
class Regressor(Protocol):
    """Structural contract every pluggable regressor conforms to.

    Conformance is structural (``isinstance`` only checks the methods exist —
    call them to verify behavior, not just presence). ``to_surrogate_spec``
    takes ``**kwargs`` because different regressors need different unit
    metadata at that boundary; each regressor documents its own keywords.
    """

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "Regressor":
        """Fit this regressor on ``X``/``y`` and return ``self``."""
        ...

    def to_fit_result(self) -> FitResult:
        """Return this fit's :class:`FitResult`, once fitted."""
        ...

    def to_surrogate_spec(self, **kwargs) -> SurrogateSpec:
        """Return this fit as a persistable ``SurrogateSpec``, once fitted."""
        ...
