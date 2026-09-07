"""PlantBlock: a collection of unit blocks — one facility.

A thin ``FlowsheetBlockData`` subclass, always ``dynamic=False``, whose time
domain **is** the ``TimeBlock``'s ordered integer set (never Pyomo.DAE): all
dynamics are hand-written difference equations. It holds the arcs between its
units and aggregates their registered power draws, fuel usage, and boundary
(feed/product) flows.

``PlantBlock`` composes **units** only. A plant containing plants is a
:class:`~flexops.core.network_block.NetworkBlock` — this block is deliberately
not overloaded to nest into itself.

**Aggregation is deferred and re-entrant.** Units are added after the plant
exists (the frozen api-freeze script even builds costing before the plant), so
:meth:`PlantBlockData._build_aggregates` may be called at any point and as often
as needed: it creates the aggregation ``Expression`` once and refreshes its body
from the units present at each call. ``FlexCosting.cost_process()`` calls it for
every plant on the model, so the common path needs no user call at all. Nothing
is ever deleted — Pyomo components are always updated in place, never removed
from a built model.

There is no ``replace_unit``: FlexParameterize mutates units in place —
``update_parameters`` for regressed values and the energy-relationship
constraint swap for a richer fit — so a unit's ports and arcs are never
disturbed.
"""

import pyomo.environ as pyo
from idaes.core import FlowsheetBlockData, declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexcore.exceptions import FlexConfigError
from flexops.core.registration import BoundaryKind, IORegistry
from flexops.core.time_block import TimeBlockData, find_time_block


def _refresh_expression(block, name: str, index, doc: str, rule) -> None:
    """Create ``block.name`` indexed over ``index`` if absent, then set its body.

    The re-entrant half of deferred aggregation: an ``Expression`` built before
    a plant's units existed would sum nothing forever, and flex-pse never
    deletes a component to rebuild it, so the body is refreshed in place
    instead.

    Args:
        block: The block to build the Expression on.
        name: Local name of the Expression.
        index: The Pyomo index set(s) to build it over.
        doc: The Expression's ``doc=``.
        rule: ``rule(*idx) -> expression`` giving each entry's current body.
    """
    if block.component(name) is None:
        block.add_component(
            name, pyo.Expression(*index, rule=lambda b, *i: 0.0, doc=doc)
        )
    expression = block.component(name)
    for idx in expression:
        expression[idx].set_value(rule(*idx) if isinstance(idx, tuple) else rule(idx))


class _AggregatingFlowsheet(FlowsheetBlockData):
    """Shared plant/network construction and power aggregation.

    Both composition levels are the same thin steady-state flowsheet over the
    TimeBlock's set; they differ only in *what* they aggregate over, which
    :meth:`_power_terms` supplies.
    """

    CONFIG = FlowsheetBlockData.CONFIG()
    CONFIG.declare(
        "time_block",
        ConfigValue(
            default=None,
            description="The fo.TimeBlock instance whose time_index this block "
            "and its children are built over. Omit it only when the model "
            "carries exactly one TimeBlock, which is then discovered.",
        ),
    )

    def build(self) -> None:
        """Build the steady-state flowsheet and the empty product registry."""
        super().build()
        self._products: dict[str, tuple] = {}

    def _setup_dynamics(self) -> None:
        """Force ``dynamic=False`` and adopt the TimeBlock's set as time domain.

        This is IDAES' own hook for settling a flowsheet's time domain, and it
        runs after the ConfigBlock is populated — the one place where the
        steady-state time convention can be enforced (never ``dynamic=True``,
        never Pyomo.DAE) and the TimeBlock's
        ordered integer Set installed *by reference*, so ``plant.time`` and
        ``time_block.time_index`` are the same object rather than two sets that
        could drift.

        Raises:
            FlexConfigError: If ``time_block=`` was omitted and the model does
                not carry exactly one TimeBlock.
        """
        self.config.dynamic = False
        self.config.time = self.time_block.time_index
        super()._setup_dynamics()

    @property
    def time_block(self) -> TimeBlockData:
        """The configured TimeBlock, or the model's only one (auto-discovery)."""
        if self.config.time_block is not None:
            return self.config.time_block
        return find_time_block(self.model())

    @property
    def products(self) -> dict[str, tuple]:
        """dict: registered products, ``{name: (flow, quality or None)}``."""
        return self._products

    def register_product(self, var, *, name: str, quality=None) -> None:
        """Register a product flow (and optionally its quality) for aggregation.

        A registered product is summed across the enclosing
        :class:`~flexops.core.network_block.NetworkBlock`; a registered quality
        is what lets the network decide whether the streams may be mixed.

        Args:
            var: The time-indexed product-flow Var/Reference/Expression.
            name: The product's name, the key it aggregates under across the
                network (e.g. ``"permeate"``).
            quality: Optional time-indexed quality indicator for the product
                (e.g. total dissolved solids).

        Raises:
            FlexConfigError: If ``name`` is already registered on this block.
        """
        if name in self._products:
            raise FlexConfigError(
                f"Product {name!r} is already registered on {self.name!r}; "
                "aggregate the streams into one Expression and register that.",
                field="name",
                value=name,
            )
        self._products[name] = (var, quality)

    def _power_terms(self, kind: nm.PowerKind) -> list:
        """Return the terms this block's power total sums. Override this.

        Raises:
            NotImplementedError: Always, on this shared base.
        """
        raise NotImplementedError

    def _fuel_terms(self) -> dict[str, list]:
        """Return the terms this block's fuel total sums, by fuel name. Override this.

        Raises:
            NotImplementedError: Always, on this shared base.
        """
        raise NotImplementedError

    def _feed_terms(self) -> dict[str, list]:
        """Return the terms this block's feed total sums, by resource. Override this.

        Raises:
            NotImplementedError: Always, on this shared base.
        """
        raise NotImplementedError

    def _product_terms(self) -> dict[str, list]:
        """Return ``{product name: [getters]}``, each ``getter(t) -> flow``.

        Getters rather than components, because a network's contributions come
        from a two-dimensional ``total_product[product, t]``, not from a
        time-indexed component of its own.

        Returns:
            The registered products of this block, one getter each.
        """
        return {
            name: [lambda t, _var=var: _var[t]]
            for name, (var, _quality) in self._products.items()
        }

    def _build_aggregates(self) -> None:
        """Build (or refresh) this block's aggregation Expressions.

        Idempotent and re-entrant: safe to call before any child exists, again
        after they are added, and as often as a caller likes.
        ``FlexCosting.cost_process()`` calls it on every plant and network.
        """
        time_index = self.time_block.time_index
        for kind, (name, doc) in nm.TOTAL_POWER_VARS.items():
            terms = self._power_terms(kind)
            _refresh_expression(
                self,
                name,
                (time_index,),
                doc,
                lambda t, _terms=terms: sum(
                    pyunits.convert(term[t], pyunits.kW) for term in _terms
                )
                + 0 * pyunits.kW,
            )

        fuel_terms = self._fuel_terms()
        if fuel_terms:
            _refresh_expression(
                self,
                nm.TOTAL_FUEL_USAGE,
                (sorted(fuel_terms), time_index),
                "Sum of the registered fuel-usage flows, by fuel name (m^3/hr).",
                lambda fuel, t, _f=fuel_terms: sum(
                    pyunits.convert(get(t), pyunits.m**3 / pyunits.hr)
                    for get in _f[fuel]
                )
                + 0 * pyunits.m**3 / pyunits.hr,
            )

        feed_terms = self._feed_terms()
        if feed_terms:
            _refresh_expression(
                self,
                nm.TOTAL_FEED,
                (sorted(feed_terms), time_index),
                "Sum of the registered feed flows, by resource name.",
                lambda resource, t, _f=feed_terms: sum(get(t) for get in _f[resource]),
            )

        products = self._product_terms()
        if products:
            _refresh_expression(
                self,
                nm.TOTAL_PRODUCT,
                (sorted(products), time_index),
                "Sum of the registered product flows, by product name.",
                lambda product, t, _p=products: sum(get(t) for get in _p[product]),
            )


@declare_process_block_class("PlantBlock")
class PlantBlockData(_AggregatingFlowsheet):
    """A facility: a collection of unit blocks (module docstring).

    Example:
        >>> import pyomo.environ as pyo
        >>> import flexops as fo
        >>> from pyomo.environ import units as pyunits
        >>> m = pyo.ConcreteModel()
        >>> m.time_block = fo.TimeBlock(
        ...     start_date="2025-01-01", end_date="2025-01-02",
        ...     time_step=1 * pyunits.hr,
        ... )
        >>> m.properties = fo.SimpleAqueousFlow()
        >>> m.waterfacility = fo.PlantBlock(time_block=m.time_block)
        >>> m.waterfacility.plant = fo.ConstantEnergyIntensityModel(
        ...     property_package=m.properties
        ... )
    """

    CONFIG = _AggregatingFlowsheet.CONFIG()

    def _power_terms(self, kind: nm.PowerKind) -> list:
        """Return every child unit's registered power draw of ``kind``.

        Args:
            kind: The :class:`~flexcore.nomenclature.PowerKind` to collect.

        Returns:
            The registered power Vars of this plant's units.
        """
        terms = []
        for block in self.component_data_objects(pyo.Block, descend_into=True):
            registry = getattr(block, "_io_registry", None)
            if isinstance(registry, IORegistry):
                terms.extend(rec.var for rec in registry.power if rec.kind is kind)
        return terms

    def _fuel_terms(self) -> dict[str, list]:
        """Return every child unit's registered fuel-usage flows, by fuel name.

        Getters rather than Vars, for the same reason as
        :meth:`_product_terms`: a network's contributions come from a
        two-dimensional ``total_fuel_usage[fuel, t]``, not from a
        time-indexed component of its own.

        Returns:
            ``{fuel_name: [getters]}``, each ``getter(t) -> flow``, for this
            plant's units.
        """
        terms: dict[str, list] = {}
        for block in self.component_data_objects(pyo.Block, descend_into=True):
            registry = getattr(block, "_io_registry", None)
            if isinstance(registry, IORegistry):
                for rec in registry.fuel:
                    terms.setdefault(rec.fuel_name, []).append(
                        lambda t, _var=rec.var: _var[t]
                    )
        return terms

    def _boundary_terms(self, kind: BoundaryKind) -> dict[str, list]:
        """Return every child unit's registered boundary flows of ``kind``.

        Discovery goes through the ``_io_registry``, never an ``isinstance``
        check against ``Feed``/``Product``: those live in
        ``flexops.unit_models``, which imports this module, so importing them
        back here would be circular.

        Args:
            kind: The :class:`~flexops.core.registration.BoundaryKind` to
                collect.

        Returns:
            ``{resource: [getters]}``, each ``getter(t) -> flow``, for this
            plant's units.
        """
        terms: dict[str, list] = {}
        for block in self.component_data_objects(pyo.Block, descend_into=True):
            registry = getattr(block, "_io_registry", None)
            if isinstance(registry, IORegistry):
                for rec in registry.boundary:
                    if rec.kind is kind:
                        terms.setdefault(rec.resource, []).append(
                            lambda t, _var=rec.var: _var[t]
                        )
        return terms

    def _feed_terms(self) -> dict[str, list]:
        """Return every child unit's registered feed flows, by resource name.

        Returns:
            ``{resource: [getters]}``, each ``getter(t) -> flow``.
        """
        return self._boundary_terms(BoundaryKind.FEED)

    def _product_terms(self) -> dict[str, list]:
        """Return the explicitly registered products merged with the discovered ones.

        Both routes land in ``total_product``: the explicit
        :meth:`~_AggregatingFlowsheet.register_product` registry, and every
        child unit that registered a
        :attr:`~flexops.core.registration.BoundaryKind.PRODUCT` boundary flow.

        Returns:
            ``{product name: [getters]}``, each ``getter(t) -> flow``.
        """
        terms = super()._product_terms()
        for resource, getters in self._boundary_terms(BoundaryKind.PRODUCT).items():
            terms.setdefault(resource, []).extend(getters)
        return terms
