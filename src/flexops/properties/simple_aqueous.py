"""SimpleAqueousFlow: the minimal flow-carrying property package (§3.7).

A minimal IDAES ``PhysicalParameterBlock``/``StateBlock`` pair carrying a
volumetric flow, structurally modeled on WaterTAP's zero-order package
(``prop_ZO``). Ports built from these state blocks carry flow between flex-pse
units via standard IDAES/Pyomo ``Arc``s.

Volumetric flow is *extensive* (conserved across an arc); pressure and
temperature, when enabled, are *intensive* (equal across an arc / at a node).
The topology base classes build ports honoring that distinction
(``Port.Extensive`` for flow, ``Port.Equality`` for the intensive states).
Pressure and temperature are **opt-in** (default off), so the v0 default stays
flow-only in its degrees of freedom.
"""

from idaes.core import (
    Component,
    LiquidPhase,
    PhysicalParameterBlock,
    StateBlock,
    StateBlockData,
    declare_process_block_class,
)
from idaes.core.util.initialization import fix_state_vars, revert_state_vars
from pyomo.common.config import ConfigValue
from pyomo.environ import Expression, NonNegativeReals, PositiveReals, Var
from pyomo.environ import units as pyunits
from pyomo.util.check_units import check_units_equivalent

from flexcore.exceptions import FlexConfigError


@declare_process_block_class("SimpleAqueousFlow")
class SimpleAqueousFlowData(PhysicalParameterBlock):
    """Parameter block for a simple aqueous stream.

    Config options (see the CONFIG entries below):

    * ``has_pressure`` / ``has_temperature`` (default False) add the intensive
      ``pressure`` / ``temperature`` state variables.

    Example:
        >>> import pyomo.environ as pyo
        >>> import flexops as fo
        >>> m = pyo.ConcreteModel()
        >>> m.props = fo.SimpleAqueousFlow()
    """

    CONFIG = PhysicalParameterBlock.CONFIG()
    CONFIG.declare(
        "has_pressure",
        ConfigValue(
            default=False,
            domain=bool,
            description="Whether state blocks carry an intensive pressure state "
            "variable (equal across arcs).",
        ),
    )
    CONFIG.declare(
        "has_temperature",
        ConfigValue(
            default=False,
            domain=bool,
            description="Whether state blocks carry an intensive temperature "
            "state variable (equal across arcs).",
        ),
    )
    CONFIG.declare(
        "density",
        ConfigValue(
            default=1000 * pyunits.kg / pyunits.m**3,
            description="Default density of aqueous phase.",
        ),
    )

    def build(self) -> None:
        """Set the state-block class and the single liquid phase/component."""
        super().build()
        self._state_block_class = SimpleAqueousStateBlock

        self.Liq = LiquidPhase()
        self.H2O = Component()

        self.density = self.config.density
        if not check_units_equivalent(self.density, pyunits.kg / pyunits.m**3):
            raise TypeError(
                f"""Density must have units of mass/volume, specified using pyunits,
                got {self.density}"""
            )

    def get_flow_basis_var_name(self) -> str:
        """Return the name of this package's extensive flow state variable.

        Lets callers (e.g. ``Tank``'s pass-through wiring) exclude "the
        flow" from a generic pass-through without hardcoding a variable name
        that varies by property package (a future mass/TDS package would
        return ``"flow_mass_phase_comp"`` instead).

        Returns:
            The state-variable name carrying extensive flow, ``"flow_vol_phase"``.
        """
        return "flow_vol_phase"

    @classmethod
    def define_metadata(cls, obj) -> None:
        """Declare supported properties and the five required default units."""
        obj.add_properties(
            {
                "flow_vol_phase": {"method": None, "units": "m^3/hr"},
                "pressure": {"method": "_pressure", "units": "Pa"},
                "temperature": {"method": "_temperature", "units": "K"},
                "flow_mass_phase": {"method": "_flow_mass_phase"},
                "dens_mass_phase": {"method": "_dens_mass_phase"},
            }
        )
        obj.add_default_units(
            {
                "time": pyunits.hr,
                "length": pyunits.m,
                "mass": pyunits.kg,
                "amount": pyunits.mol,
                "temperature": pyunits.K,
            }
        )


class _SimpleAqueousStateBlock(StateBlock):
    """Whole-set methods for SimpleAqueous state blocks (fix/unfix hooks)."""

    def fix_initialization_states(self) -> None:
        """Fix all state variables for initialization."""
        fix_state_vars(self)

    def initialize(self, *args, hold_state: bool = False, **kwargs):
        """Fix state vars for initialization; optionally hold them fixed.

        Args:
            hold_state: If True, leave the state vars fixed and return the flags
                needed to release them later.

        Returns:
            The fix flags if ``hold_state`` is True, else None.
        """
        flags = fix_state_vars(self)
        if hold_state:
            return flags
        self.release_state(flags)
        return None

    def release_state(self, flags, **kwargs) -> None:
        """Unfix state vars fixed during initialization.

        Args:
            flags: The fix flags returned by :meth:`initialize`.
        """
        if flags is not None:
            revert_state_vars(self, flags)


@declare_process_block_class(
    "SimpleAqueousStateBlock", block_class=_SimpleAqueousStateBlock
)
class SimpleAqueousStateBlockData(StateBlockData):
    """State block carrying volumetric flow and optional extras.

    State variables are indexed over time directly: the owning unit passes the
    ``time_index`` Set via ``build_state_block(time_index=...)`` and gets a
    single scalar state block whose variables span the horizon. Extensive,
    per-phase quantities lead with time then phase (``flow_vol_phase[t, phase]``);
    intensive stream properties drop the phase index (the opt-in
    ``pressure[t]``/``temperature[t]``, assumed equal across phases).
    """

    CONFIG = StateBlockData.CONFIG()
    CONFIG.declare(
        "time_index",
        ConfigValue(
            default=None,
            description="Ordered Pyomo time Set the state variables are indexed "
            "over (the owning unit passes TimeBlock.time_index).",
        ),
    )
    CONFIG.declare(
        "density",
        ConfigValue(
            default=1000 * pyunits.kg / pyunits.m**3,
            description="Default density of aqueous phase.",
        ),
    )

    def build(self) -> None:
        """Create the time-indexed ``flow_vol_phase`` and any enabled extras."""
        super().build()
        time = self.config.time_index
        if time is None:
            raise FlexConfigError(
                "SimpleAqueousStateBlock requires a time_index; build it with "
                "build_state_block(time_index=tb.time_index).",
                field="time_index",
                value=None,
            )
        self.flow_vol_phase = Var(
            time,
            self.params.phase_list,
            initialize=1.0,
            domain=NonNegativeReals,
            units=pyunits.m**3 / pyunits.hr,
            doc="Volumetric flowrate by time and phase",
        )
        if self.params.config.has_pressure:
            self._pressure()
        if self.params.config.has_temperature:
            self._temperature()

    def _flow_mass_phase(self):
        self.dens_mass_phase[...]

        def rule_cmc(self, t, j):
            return pyunits.convert(
                self.flow_vol_phase[t, j] * self.dens_mass_phase[t, j],
                pyunits.kg / pyunits.hr,
            )

        self.flow_mass_phase = Expression(
            self.config.time_index, self.params.phase_list, rule=rule_cmc
        )

    def _pressure(self):
        self.pressure = Var(
            self.config.time_index,
            initialize=101325.0,
            domain=PositiveReals,
            units=pyunits.Pa,
            doc="Pressure",
        )

    def _temperature(self):
        self.temperature = Var(
            self.config.time_index,
            initialize=293.15,
            domain=PositiveReals,
            units=pyunits.K,
            doc="Pressure",
        )

    def _dens_mass_phase(self):
        self.dens_mass_phase = Var(
            self.config.time_index,
            self.params.phase_list,
            initialize=self.params.density,
            units=pyunits.kg / pyunits.m**3,
            doc="Mass density of flow",
        )
        self.dens_mass_phase.fix()

    def define_state_vars(self) -> dict:
        """Return the state-variable dict (flow plus any enabled extras)."""
        state_vars = {"flow_vol_phase": self.flow_vol_phase}
        if self.params.config.has_pressure:
            state_vars["pressure"] = self.pressure
        if self.params.config.has_temperature:
            state_vars["temperature"] = self.temperature
        return state_vars

    def define_display_vars(self) -> dict:
        """Return the display-variable dict for reporting."""
        return {name: var for name, var in self.define_state_vars().items()}


# ``declare_process_block_class`` injects the constructible ``SimpleAqueousStateBlock``
# wrapper into this module's namespace at runtime; bind the name explicitly so
# static tools resolve the forward reference in ``SimpleAqueousFlowData.build``.
SimpleAqueousStateBlock = globals()["SimpleAqueousStateBlock"]
