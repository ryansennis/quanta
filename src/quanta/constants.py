"""
Docstring for quanta.constants
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
from functools import lru_cache
import numpy as np

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
            return DimensionComposite({self: 1, other: 1})
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Dimension):
            return DimensionComposite({self: 1, other: -1})
        return NotImplemented

    def __pow__(self, power):
        return DimensionComposite({self: power})

@dataclass(frozen=True)
class DimensionComposite:
    """Composite of fundamental dimensions with exponents"""
    dimensions: Dict[Dimension, float]

    def __mul__(self, other):
        if isinstance(other, DimensionComposite):
            new_dims = dict(self.dimensions)
            for dim, exp in other.dimensions.items():
                new_dims[dim] = new_dims.get(dim, 0) + exp
            return DimensionComposite({k: v for k, v in new_dims.items() if v != 0})
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, DimensionComposite):
            new_dims = dict(self.dimensions)
            for dim, exp in other.dimensions.items():
                new_dims[dim] = new_dims.get(dim, 0) - exp
            return DimensionComposite({k: v for k, v in new_dims.items() if v != 0})
        return NotImplemented

    def __pow__(self, power):
        return DimensionComposite({dim: exp * power for dim, exp in self.dimensions.items()})

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
        name: Name of the unit (e.g., "meter")
        symbol: Symbol (e.g., "m")
        scale: Multiplication factor to convert to base units
        dimensions: DimensionComposite object
    """
    name: str
    symbol: str
    scale: float = 1.0
    dimensions: DimensionComposite = DimensionComposite({})

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
        return Quantity(value, Unit("", "", 1.0, DimensionComposite({}))) / self

@dataclass
class Quantity:
    """
    A physical quantity with value and unit
    
    Attributes:
        value: Numerical value
        unit: Unit object
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
# BASE UNITS (SI)
# ============================================================================

# Fundamental units
second = Unit("second", "s", dimensions=DimensionComposite({Dimension.TIME: 1}))
meter = Unit("meter", "m", dimensions=DimensionComposite({Dimension.LENGTH: 1}))
kilogram = Unit("kilogram", "kg", dimensions=DimensionComposite({Dimension.MASS: 1}))
coulomb = Unit("coulomb", "C", dimensions=DimensionComposite({Dimension.CHARGE: 1}))
kelvin = Unit("kelvin", "K", dimensions=DimensionComposite({Dimension.TEMPERATURE: 1}))
mole = Unit("mole", "mol", dimensions=DimensionComposite({Dimension.AMOUNT: 1}))
candela = Unit("candela", "cd", dimensions=DimensionComposite({Dimension.LUMINOUS_INTENSITY: 1}))
radian = Unit("radian", "rad", dimensions=DimensionComposite({}))
