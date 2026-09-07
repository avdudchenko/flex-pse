"""``EstimationResult`` — the result shape of a native-constraint NLP fit.

Not yet implemented. This is the intended field-by-field shape that
:func:`~flexparameterize.estimation.estimator.estimate_parameters` will return.
"""

from dataclasses import dataclass

import pandas as pd

from flexparameterize.regression.base import FitResult


@dataclass(frozen=True)
class EstimationResult:
    """The result of a native-constraint NLP parameter fit.

    Attributes:
        point_estimates: Parameter name -> estimated value.
        std_errors: Parameter name -> standard error; empty if
            ``cov_method=None`` was passed to
            :func:`~flexparameterize.estimation.estimate_parameters`.
        covariance: Parameter covariance matrix (index/columns = parameter
            names), or ``None`` if ``cov_method=None``.
        objective_value: Sum-of-squared-errors objective at the optimum.
        n_observations: Total (output, time) pairs used in the fit.
        n_parameters: Number of parameters estimated.
        data_window: ``(first, last)`` timestamp used.
        solver_status: The solver's termination condition, as a string.
    """

    point_estimates: dict[str, float]
    std_errors: dict[str, float]
    covariance: pd.DataFrame | None
    objective_value: float
    n_observations: int
    n_parameters: int
    data_window: tuple
    solver_status: str

    def to_fit_result(self) -> FitResult:
        """Lossy down-cast to :class:`~flexparameterize.regression.base.FitResult`.

        Not yet implemented.
        """
        raise NotImplementedError("M11b is not yet implemented.")
