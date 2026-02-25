"""
Docstring for quanta.constants
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict

class Dimension(Enum):
    """Fundamental physical dimensions"""
    TIME = "T"
    LENGTH = "L"
    MASS = "M"
    CHARGE = "Q"
    TEMPERATURE = "Θ"
    AMOUNT = "N"
    LUMINOUS_INTENSITY = "J"
    ANGLE = "A"

    def __mul__(self, other):
        if isinstance(other, Dimension):
            return CompositeDimension({self: 1, other: 1})
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Dimension):
            return CompositeDimension({self: 1, other: -1})
        return NotImplemented

    def __pow__(self, power):
        return CompositeDimension({self: power})

@dataclass(frozen=True)
class CompositeDimension:
    """
    Composite of fundamental dimensions with exponents

    Attributes:
        dimensions (Dict[Dimension, float]): The composed dimensions with respective powers
    """
    dimensions: Dict[Dimension, float]

    def __mul__(self, other):
        if isinstance(other, CompositeDimension):
            new_dims = dict(self.dimensions)
            for dim, exp in other.dimensions.items():
                new_dims[dim] = new_dims.get(dim, 0) + exp
            return CompositeDimension({k: v for k, v in new_dims.items() if v != 0})
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, CompositeDimension):
            new_dims = dict(self.dimensions)
            for dim, exp in other.dimensions.items():
                new_dims[dim] = new_dims.get(dim, 0) - exp
            return CompositeDimension({k: v for k, v in new_dims.items() if v != 0})
        return NotImplemented

    def __pow__(self, power):
        return CompositeDimension({dim: exp * power for dim, exp in self.dimensions.items()})

    def __str__(self):
        parts = []
        for dim, exp in sorted(self.dimensions.items()):
            if exp == 1:
                parts.append(dim.value)
            else:
                parts.append(f"{dim.value}^{exp}")
        return "·".join(parts) if parts else "1"

@dataclass
class Unit:
    """
    A physical unit with scaling and dimensions
    
    Attributes:
        name (str): Name of the unit (e.g., "meter")
        symbol (str): Symbol (e.g., "m")
        scale (float): Multiplication factor to convert to base units. Default 1.0
        dimensions (CompositeDimension): CompositeDimension object
    """
    name: str
    symbol: str
    scale: float = 1.0
    dimensions: CompositeDimension = CompositeDimension({})

    def __mul__(self, other):
        if isinstance(other, Unit):
            return Unit(
                name=f"{self.name}·{other.name}",
                symbol=f"{self.symbol}·{other.symbol}",
                scale=self.scale * other.scale,
                dimensions=self.dimensions * other.dimensions
            )
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Unit):
            return Unit(
                name=f"{self.name}/{other.name}",
                symbol=f"{self.symbol}/{other.symbol}",
                scale=self.scale / other.scale,
                dimensions=self.dimensions / other.dimensions
            )
        return NotImplemented

    def __pow__(self, power):
        return Unit(
            name=f"{self.name}^{power}",
            symbol=f"{self.symbol}^{power}",
            scale=self.scale ** power,
            dimensions=self.dimensions ** power
        )

    def __rmul__(self, value):
        """Enable 5 * meter syntax"""
        return Quantity(value, self)

    def __rtruediv__(self, value):
        """Enable 1 / meter syntax"""
        return Quantity(value, Unit("", "", 1.0, CompositeDimension({}))) / self

@dataclass
class Quantity:
    """
    A physical quantity with value and unit
    
    Attributes:
        value (float): Numerical value
        unit (Unit): Unit object
    """
    value: float
    unit: Unit

    def to(self, target_unit: Unit) -> 'Quantity':
        """Convert to target unit"""
        if self.unit.dimensions != target_unit.dimensions:
            raise ValueError(
                f"Incompatible dimensions: {self.unit.dimensions} vs {target_unit.dimensions}"
            )
        scale_factor = self.unit.scale / target_unit.scale
        return Quantity(self.value * scale_factor, target_unit)

    def __mul__(self, other):
        if isinstance(other, Quantity):
            return Quantity(
                self.value * other.value,
                self.unit * other.unit
            )
        elif isinstance(other, (int, float)):
            return Quantity(self.value * other, self.unit)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            return Quantity(
                self.value / other.value,
                self.unit / other.unit
            )
        if isinstance(other, (int, float)):
            return Quantity(self.value / other, self.unit)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Quantity):
            if self.unit.dimensions != other.unit.dimensions:
                raise ValueError("Cannot add quantities with different dimensions")
            # Convert other to self's unit
            other_in_self = other.to(self.unit)
            return Quantity(self.value + other_in_self.value, self.unit)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Quantity):
            if self.unit.dimensions != other.unit.dimensions:
                raise ValueError("Cannot subtract quantities with different dimensions")
            other_in_self = other.to(self.unit)
            return Quantity(self.value - other_in_self.value, self.unit)
        return NotImplemented

    def __str__(self):
        return f"{self.value:.6g} {self.unit.symbol}"

    def __repr__(self):
        return f"Quantity({self.value}, {self.unit.symbol})"

# ============================================================================
# BASE UNITS
# ============================================================================

second = Unit("second", "s", dimensions=CompositeDimension({Dimension.TIME: 1}))
meter = Unit("meter", "m", dimensions=CompositeDimension({Dimension.LENGTH: 1}))
kilogram = Unit("kilogram", "kg", dimensions=CompositeDimension({Dimension.MASS: 1}))
coulomb = Unit("coulomb", "C", dimensions=CompositeDimension({Dimension.CHARGE: 1}))
kelvin = Unit("kelvin", "K", dimensions=CompositeDimension({Dimension.TEMPERATURE: 1}))
mole = Unit("mole", "mol", dimensions=CompositeDimension({Dimension.AMOUNT: 1}))
candela = Unit("candela", "cd", dimensions=CompositeDimension({Dimension.LUMINOUS_INTENSITY: 1}))
radian = Unit("radian", "rad", dimensions=CompositeDimension({}))

# ============================================================================
# DERIVED UNITS
# ============================================================================

hertz = Unit(
    "hertz",
    "Hz", 
    dimensions=CompositeDimension({Dimension.TIME: -1})
)

newton = Unit(
    "newton",
    "N",
    dimensions=(kilogram * meter**2 / second**2).dimensions
)

joule = Unit(
    "joule",
    "J",
    dimensions=(newton * meter).dimensions
)

volt = Unit(
    "volt",
    "V",
    dimensions=(joule / coulomb).dimensions
)

watt = Unit(
    "watt",
    "W",
    dimensions=(joule / second).dimensions
)

bohr = Unit(
    "bohr radius",
    "a_o",
    5.29177210903e-11,
    dimensions=meter.dimensions
)

hartree = Unit(
    "hartree",
    "E_h",
    4.3597447222071e-18,
    dimensions=joule.dimensions
)

electron_volt = Unit(
    "electron volt",
    "eV",
    1.602176634e-19,
    dimensions=joule.dimensions
)

h_bar = Unit(
    "planck's constant",
    "h_bar",
    6.62607015e-34,
    dimensions=(joule*second).dimensions
)
