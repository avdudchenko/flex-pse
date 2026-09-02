"""QuadraticSurrogate: not yet implemented."""

from typing import ClassVar

from flexcore.config.schema import SurrogateType
from flexops.surrogates.base import Surrogate


class QuadraticSurrogate(Surrogate):
    """A quadratic (up to degree-2, including squared terms) relationship.

    Not yet implemented; use
    :class:`~flexops.surrogates.multilinear.MultilinearSurrogate`
    for a relationship with no squared term.
    """

    surrogate_type: ClassVar[SurrogateType] = SurrogateType.QUADRATIC

    def _validate(self) -> None:
        """Raise: quadratic surrogates are not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "QuadraticSurrogate is not yet implemented. Use "
            "MultilinearSurrogate (surrogate_type='multilinear') for a "
            "relationship with no squared term."
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
