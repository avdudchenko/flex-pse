"""``UnitExperiment`` — the ``pyomo.contrib.parmest`` experiment wiring.

Not yet implemented. This is the intended shape wrapping a built unit as one
``parmest`` ``Experiment`` over its whole horizon.
"""

import pandas as pd
from pyomo.contrib.parmest.experiment import Experiment

from flexparameterize.tags import TagMap


class UnitExperiment(Experiment):
    """Wraps a built unit as one ``parmest`` ``Experiment`` over its whole horizon.

    Not yet implemented.
    """

    def __init__(
        self,
        unit,
        data: pd.DataFrame,
        tagmap: TagMap,
        parameter_names: list[str],
        output_names: list[str],
        measurement_error: dict[str, float] | None,
    ) -> None:
        raise NotImplementedError("M11b is not yet implemented.")

    def get_labeled_model(self):
        """Return ``unit.model()``, annotated with parmest's Suffixes.

        Not yet implemented.
        """
        raise NotImplementedError("M11b is not yet implemented.")
