from __future__ import annotations


__all__ = [
    "Integral",
    "Real",
    "Complex",
    "Float",
    "Z",
    "R",
    "C",
    "is_integral",
    "is_real",
    "is_complex",
    "is_probability",
    "Interval",
]

import decimal
import fractions
import numbers
import sys
from typing import Any, Generic, Optional, TypeVar, Union, cast


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
Real = Union[Integral, Float, decimal.Decimal, fractions.Fraction]
Complex = Union[Real, complex, mpmath.mpc]


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
    decimal.Decimal,
    fractions.Fraction,
    mpmath.libmp.MPZ_TYPE,
    int,
    bool,
)
# complex numbers
C = TypeVar(
    "C",
    mpmath.mpc,
    complex,
    mpmath.mpf,
    float,
    decimal.Decimal,
    fractions.Fraction,
    mpmath.libmp.MPZ_TYPE,
    int,
    bool,
)


def is_integral(x: Any) -> bool:
    return isinstance(x, (int, mpmath.libmp.MPZ_TYPE))


def is_real(x: Any, /, strict: bool = False) -> bool:
    return (
        isinstance(x, (float, decimal.Decimal, fractions.Fraction, mpmath.mpf))
        or not strict
        and is_integral(x)
    )


def is_complex(x: Any, /, strict: bool = False) -> bool:
    return isinstance(x, (complex, mpmath.mpc)) or not strict and is_real(x)


def _cmax(*args: C) -> C:
    res = None
    for arg in args:
        if res is None:
            res = arg
        elif isinstance(res, numbers.Real):  # not complex
            if arg > res:
                res = arg
        else:
            if arg.real > res.real and arg.imag > res.imag:
                res = arg
            elif arg.real > res.real:
                res = type(res)(arg.real, res.imag)
            elif arg.imag > res.imag:
                res = type(res)(res.real, arg.imag)

    if res is None:
        raise ValueError("cmax expected 1 argument, got 0")

    return res


def _cmin(*args: C) -> C:
    return cast(C, -_cmax(*(-arg for arg in args)))


class Interval(Generic[C]):
    """Closed interval (endpoints a and b are included)"""

    a: C
    b: C

    def __init__(self, a: C, b: C, /):
        if a.real > b.real:
            raise ValueError("a cannot be larger than b")
        if a.imag > b.imag:
            raise ValueError("Im(a) cannot be larger than Im(b)")

        self.a = a
        self.b = b

    def __repr__(self):
        return f"{type(self).__name__}({self.a!r}, {self.b!r})"

    def __str__(self):
        return f"[{self.a}, {self.b}]"

    def __contains__(self, x: Union[C, Interval[C], object]):
        if isinstance(x, Interval):
            # subinterval check
            return x.a in self and x.b in self
        if isinstance(x, numbers.Complex):
            return (
                self.a.real <= x.real <= self.b.real
                and self.a.imag <= x.imag <= self.b.imag
            )

        return False

    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.a == other.a and self.b == other.b
        elif self.is_degenerate and isinstance(other, numbers.Complex):
            return other == self.a

        return False

    def __lt__(self, other) -> bool:
        if isinstance(other, Interval):
            # subinterval check
            other = other.a
        if isinstance(other, numbers.Complex):
            return bool(self.b.real < other.real or self.b.imag < other.imag)

        return NotImplemented

    def __le__(self, other) -> bool:
        if isinstance(other, (Interval, numbers.Complex)):
            return self == other or self < other

        return NotImplemented

    def __gt__(self, other) -> bool:
        if isinstance(other, Interval):
            # subinterval check
            other = other.b
        if isinstance(other, numbers.Complex):
            return bool(self.a.real > other.real or self.a.real > other.imag)

        return NotImplemented

    def __ge__(self, other) -> bool:
        if isinstance(other, (Interval, numbers.Complex)):
            return self == other or self > other

        return NotImplemented

    def __and__(self, other: Interval[Any]) -> Optional[Interval[C]]:
        if isinstance(other, Interval):
            if self.isdisjoint(other):
                return None

            return type(self)(_cmax(self.a, other.a), _cmin(self.b, other.b))
        return NotImplemented

    def __or__(self, other: Interval[C]) -> Interval[C]:
        if isinstance(other, Interval):
            if self.isdisjoint(other):
                # TODO mask the gap, or create a disjoint interval class
                raise NotImplementedError

            return type(self)(_cmin(self.a, other.a), _cmax(self.b, other.b))
        return NotImplemented

    def __bool__(self) -> bool:
        return True

    def __hash__(self):
        return hash((self.a, self.b))

    @property
    def is_bounded(self) -> bool:
        return bool(
            mpmath.isfinite(self.a.real)
            and mpmath.isfinite(self.a.imag)
            and mpmath.isfinite(self.b.real)
            and mpmath.isfinite(self.b.imag)
        )

    @property
    def is_degenerate(self) -> bool:
        return bool(self.a == self.b)

    @property
    def size(self) -> Union[C, int]:  # lower type bound for bool
        return self.b - self.a

    @property
    def radius(self) -> Union[C, float]:  # lower type bound for int
        return self.size / 2

    @property
    def mid(self) -> Union[C, float]:  # lower type bound for bool
        return (self.a + self.b) / 2

    def isdisjoint(self, other: Interval[Any]) -> bool:
        return bool(
            self.a.real > other.b.real
            or self.a.imag > other.b.imag
            or self.b.real < other.a.real
            or self.a.imag < other.a.imag
        )


def is_probability(y: Any) -> bool:
    if isinstance(y, numbers.Real):
        return bool(0.0 <= float(y) <= 1.0)

    return False
