"""ArimaSurrogate: not yet implemented."""

from typing import ClassVar

from flexcore.config.schema import SurrogateType
from flexops.surrogates.base import Surrogate


class ArimaSurrogate(Surrogate):
    """An ARIMA (lag-polynomial, time-series) relationship.

    Not yet implemented; use
    :class:`~flexops.surrogates.multilinear.MultilinearSurrogate`
    for a relationship with no time-lagged terms.
    """

    surrogate_type: ClassVar[SurrogateType] = SurrogateType.ARIMA

    def _validate(self) -> None:
        """Raise: ARIMA surrogates are not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "ArimaSurrogate is not yet implemented. Use MultilinearSurrogate "
            "(surrogate_type='multilinear') for a relationship with no "
            "time-lagged terms."
        )

    @property
    def input_variables(self) -> dict[str, str]:
        """Never reached -- ``_validate`` always raises first."""
        raise NotImplementedError

    @property
    def output_variables(self) -> dict[str, str]:
        """Never reached -- ``_validate`` always raises first."""
        raise NotImplementedError

    def build(self, unit, target):
        """Never reached -- ``_validate`` always raises first."""
        raise NotImplementedError
