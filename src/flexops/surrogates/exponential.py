"""ExponentialSurrogate: not yet implemented."""

from typing import ClassVar

from flexcore.config.schema import SurrogateType
from flexops.surrogates.base import Surrogate


class ExponentialSurrogate(Surrogate):
    """An exponential relationship in its inputs.

    Not yet implemented; use
    :class:`~flexops.surrogates.multilinear.MultilinearSurrogate`
    for a relationship linear in each input.
    """

    surrogate_type: ClassVar[SurrogateType] = SurrogateType.EXPONENTIAL

    def _validate(self) -> None:
        """Raise: exponential surrogates are not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "ExponentialSurrogate is not yet implemented. Use "
            "MultilinearSurrogate (surrogate_type='multilinear') for a "
            "relationship linear in each input."
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
