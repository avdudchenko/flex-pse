"""Native-constraint NLP parameter estimation.

Not yet implemented; this package is a reserved API surface.

Unlike :mod:`flexparameterize.regression` (the ``Regressor`` Protocol, which
fits bare ``pandas`` data columns decoupled from any live Pyomo model), this
package fits a unit's own already-built, nonlinear-in-parameter constraint
(e.g. ``Pump``'s hydraulic power law) directly, via
``pyomo.contrib.parmest``. A sibling capability to ``regression/``, not a
replacement or an extension of it — neither package imports the other.
"""

from flexparameterize.estimation.estimator import commit_estimate, estimate_parameters
from flexparameterize.estimation.experiment import UnitExperiment
from flexparameterize.estimation.result import EstimationResult

__all__ = [
    "EstimationResult",
    "UnitExperiment",
    "commit_estimate",
    "estimate_parameters",
]
