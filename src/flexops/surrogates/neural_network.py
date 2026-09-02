"""NeuralNetworkSurrogate: not yet implemented."""

from typing import ClassVar

from flexcore.config.schema import SurrogateType
from flexops.surrogates.base import Surrogate


class NeuralNetworkSurrogate(Surrogate):
    """A neural-network (e.g. ReLU big-M or ICNN) forward-pass relationship.

    Not yet implemented; use
    :class:`~flexops.surrogates.multilinear.MultilinearSurrogate`
    for a relationship expressible without a network.
    """

    surrogate_type: ClassVar[SurrogateType] = SurrogateType.NEURAL_NETWORK

    def _validate(self) -> None:
        """Raise: neural-network surrogates are not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "NeuralNetworkSurrogate is not yet implemented. Use "
            "MultilinearSurrogate (surrogate_type='multilinear') for a "
            "relationship expressible without a network."
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
