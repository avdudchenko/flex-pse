"""Parse the compact units strings a persisted config or surrogate data uses.

A leaf module (imports only ``flexcore`` and Pyomo) so both ``build.py`` and
``flexops.surrogates`` can parse a units string without a cross-package
import cycle.
"""

import re

from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError
from flexops.costing import currency_units

_UNIT_TOKEN = re.compile(r"^([A-Za-z$]+)(?:\*\*|\^)?(-?\d+)?$")


def parse_units(text: str):
    """Parse a units string into a Pyomo units expression.

    Handles the compact forms persisted configs use: ``"min"``, ``"m^3/hr"``,
    ``"kWh/m^3"``, ``"USD/kWh"`` — ``*``-separated factors, at most one ``/``,
    and ``^``/``**`` exponents. A token Pyomo does not know is registered as a
    currency (so ``"USD"`` works without the costing block existing yet).

    Args:
        text: The units string.

    Returns:
        The corresponding Pyomo units expression.

    Raises:
        FlexConfigError: If a token is not a parsable unit name and exponent.
    """
    numerator, _, denominator = text.strip().partition("/")
    result = 1
    for side, factors in ((1, numerator), (-1, denominator)):
        for token in factors.split("*") if factors.strip() else []:
            token = token.strip()
            if not token:
                continue
            match = _UNIT_TOKEN.match(token)
            if match is None:
                raise FlexConfigError(
                    f"Could not parse {token!r} in units string {text!r}. Write "
                    "units as '*'-separated factors with at most one '/', e.g. "
                    "'kWh/m^3'.",
                    field="units",
                    value=text,
                )
            name, exponent = match.group(1), int(match.group(2) or 1)
            unit = getattr(pyunits, name, None)
            if unit is None:
                unit = currency_units(name)
            result = result * unit ** (side * exponent)
    return result
