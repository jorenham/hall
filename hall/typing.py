from __future__ import annotations


__all__ = [
    "Integral",
    "Real",
    "Complex",
    "Float",
    "Probability",
    "Z",
    "R",
    "C",
    "is_integral",
    "is_real",
    "is_complex",
    "is_probability",
]

import numbers
import sys
from decimal import Decimal
from fractions import Fraction
from typing import Any, SupportsFloat, SupportsInt, TypeVar, Union


if sys.version_info >= (3, 10):
    from types import EllipsisType  # noqa
else:
    EllipsisType = None

import mpmath
from mpmath import libmp


# numbers.Complex, numbers.Real, etc. as typevar bound are bugged in mypy :(
# https://github.com/python/mypy/issues/3186
Integral = Union[bool, int, libmp.MPZ_TYPE]
Float = Union[float, mpmath.mpf]
Real = Union[Integral, Float, Decimal, Fraction]
Complex = Union[Real, complex, mpmath.mpc]
Probability = Float


# natural numbers
Z = TypeVar(
    "Z",
    mpmath.libmp.MPZ_TYPE,
    int,
)
# real numbers
R = TypeVar(
    "R",
    mpmath.mpf,
    float,
    Decimal,
    Fraction,
    mpmath.libmp.MPZ_TYPE,
    int,
)
# complex numbers
C = TypeVar(
    "C",
    mpmath.mpc,
    complex,
    mpmath.mpf,
    float,
    Decimal,
    Fraction,
    mpmath.libmp.MPZ_TYPE,
    int,
)


def is_integral(x: Any) -> bool:
    return (
        isinstance(x, SupportsInt)
        and isinstance(x, (int, mpmath.libmp.MPZ_TYPE))
        or not getattr(x, "imag", 0)
        and int(x) == x
    )


def is_real(x: Any, /, strict: bool = False) -> bool:
    return (
        isinstance(x, SupportsFloat)
        and isinstance(x, (float, Decimal, Fraction, mpmath.mpf))
        and not getattr(x, "imag", 0)
        or not strict
        and is_integral(x)
    )


def is_complex(x: Any, /, strict: bool = False) -> bool:
    return isinstance(x, (complex, mpmath.mpc)) or not strict and is_real(x)


def is_probability(y: Any) -> bool:
    if isinstance(y, numbers.Real):
        return bool(0.0 <= float(y) <= 1.0)

    return False
