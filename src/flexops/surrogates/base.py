"""Surrogate: the base class every predefined relationship shape subclasses.

A ``Surrogate`` turns a :class:`~flexcore.config.schema.SurrogateSpec`'s
opaque ``data`` mapping into Pyomo objects. It validates that mapping in
``__init__`` (conventions §4: no opaque nested JSON blob passes unvalidated),
so a malformed relationship fails at construction, not mid-solve.
:meth:`~flexops.core.ops_block.OpsBlockData.swap_relation` is the only caller
of :meth:`Surrogate.build`.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from flexcore.config.schema import SurrogateType


class Surrogate(ABC):
    """Base class of every predefined surrogate-structure class.

    A leaf class in ``flexops`` (imports only ``flexcore`` and Pyomo), so
    ``flexops.core.ops_block`` can import the registry that resolves to it
    with no cross-package import.

    Attributes:
        surrogate_type: The :class:`~flexcore.config.schema.SurrogateType`
            this class implements.
        data: The validated data mapping, as given to ``__init__``.
    """

    surrogate_type: ClassVar[SurrogateType]

    def __init__(self, data: dict) -> None:
        """Store and validate ``data``.

        Args:
            data: The relationship's data, in the shape this class defines.

        Raises:
            FlexConfigError: If ``data`` does not match this class's contract.
            NotImplementedError: If this class is not yet implemented.
        """
        self.data = dict(data)
        self._validate()

    @abstractmethod
    def _validate(self) -> None:
        """Validate ``self.data`` against this class's contract.

        Raises:
            FlexConfigError: If ``data`` does not match this class's contract.
            NotImplementedError: If this class is not yet implemented.
        """

    @property
    @abstractmethod
    def input_variables(self) -> dict[str, str]:
        """Return this relationship's input variable names and declared units."""

    @property
    @abstractmethod
    def output_variables(self) -> dict[str, str]:
        """Return this relationship's output variable names and declared units."""

    @abstractmethod
    def build(self, unit, target):
        """Return ``body(t)`` evaluating this relationship at time ``t``.

        Args:
            unit: The :class:`~flexops.core.ops_block.OpsBlockData` the
                relationship is built on; resolve input variables through
                ``unit.resolve_variable``.
            target: The Var/Reference the relationship determines.

        Returns:
            A callable taking a time index and returning a units-carrying
            Pyomo expression in this relationship's declared output units, or
            ``pyomo.environ.Constraint.Skip`` to omit that index. May attach
            auxiliary Vars/Constraints to ``unit``; ``swap_relation`` finds
            and tracks them itself.
        """
