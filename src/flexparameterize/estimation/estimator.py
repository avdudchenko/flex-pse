"""``estimate_parameters``/``commit_estimate`` — the public entry points.

Not yet implemented. This is the intended shape of native-constraint NLP
parameter estimation (fitting parameters against a unit's own nonlinear
constraint, e.g. a pump's hydraulic law, rather than a regression against a
constant or linear surrogate): ``estimate_parameters`` never mutates the unit
it fits against, and ``commit_estimate`` is the separate, explicit step that
writes a result back onto a live model.
"""

import pandas as pd

from flexparameterize.estimation.result import EstimationResult
from flexparameterize.tags import TagMap


def estimate_parameters(
    unit,
    data: pd.DataFrame,
    tagmap: TagMap,
    parameter_names: list[str],
    *,
    output_names: list[str] | None = None,
    measurement_error: dict[str, float] | None = None,
    cov_method: str | None = "reduced_hessian",
    solver: str = "ipopt",
) -> EstimationResult:
    """Fit ``parameter_names`` against ``unit``'s own native constraint.

    Never mutates ``unit`` — see :func:`commit_estimate` for the write-back.

    Not yet implemented.
    """
    raise NotImplementedError("M11b is not yet implemented.")


def commit_estimate(unit, result: EstimationResult) -> None:
    """Write ``result.point_estimates`` into ``unit`` and re-fix each Var.

    Not yet implemented.
    """
    raise NotImplementedError("M11b is not yet implemented.")
