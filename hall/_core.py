from __future__ import annotations


__all__ = ["Stochast", "Distribution", "Interval"]

import abc
import numbers
from typing import (
    Any,
    ClassVar,
    Generic,
    Optional,
    Protocol,
    Union,
    cast,
    runtime_checkable,
)

import mpmath

from hall._types import C, Float, Probability
from hall.event import Event, EventEq, EventInterval


class Stochast(Generic[C]):
    # TODO (linear) arithmatic
    __slots__ = ("distribution",)

    distribution: Distribution[C]

    def __init__(self, distribution: Distribution[C]):
        self.distribution = distribution

    def __eq__(self, other: C) -> EventEq[C]:  # type: ignore
        return EventEq(self, other)

    def __ne__(self, other: C) -> EventEq[C]:  # type: ignore
        return EventEq(self, other, _inv=True)

    def __lt__(self, other: C) -> EventInterval[C]:
        if self.distribution.discrete:
            return EventInterval(self, b=other - 1)
        else:
            return EventInterval(self, b=other)  # TODO some tiny amount

    def __le__(self, other: C) -> EventInterval[C]:
        return EventInterval(self, b=other)

    def __gt__(self, other: C) -> EventInterval[C]:
        return EventInterval(self, a=other)

    def __ge__(self, other: C) -> EventInterval[C]:
        if self.distribution.discrete:
            return EventInterval(self, a=other + 1)
        else:
            return EventInterval(self, a=other)  # TODO some tiny amount

    def __invert__(self) -> EventEq[C]:
        return EventEq(self, 0)

    def __or__(self, other: Union[Event[C], Stochast[C]]) -> EventEq[C]:
        # TODO precedence hacking (don't collapse X | Y to X when uncorr!)
        #  P(X == x | Y == y) => P(X == (X | Y) == y)
        #  P(X | Y == y) => P((X | Y) == y)
        #  P(X == x | Y) => P(X == (x | Y))
        #  P(X, Y | Z) => P(X, (Y | Z))
        return NotImplemented

    def __contains__(self, item) -> bool:
        return item in self.distribution.support

    @property
    def outcomes(self) -> Interval[C]:
        return self.distribution.support


@runtime_checkable
class Distribution(Protocol[C]):
    __slots__ = ()

    discrete: ClassVar[bool]

    def __invert__(self) -> Stochast[C]:
        return Stochast(self)

    @property
    @abc.abstractmethod
    def support(self) -> Interval[C]:
        ...

    @property
    @abc.abstractmethod
    def mean(self) -> Union[C, Float]:
        """
        The mean/expected value of the distribution
        """
        ...

    @property
    @abc.abstractmethod
    def variance(self) -> Union[C, Float]:
        """
        The variance of the distribution
        """
        ...

    @abc.abstractmethod
    def f(self, x: C) -> Probability:
        """
        Probability Mass/Density Function (PMF/PDF) that must integrate to 1.
        """
        ...

    @abc.abstractmethod
    def F(self, x: C) -> Probability:
        """
        Cumulative Distribution Function (CDF); the integral of `f`.
        """
        ...

    @abc.abstractmethod
    def G(self, y: Probability) -> C:
        """
        Percent Point Function (PPF); the inverse of the CDF `F`
        """
        ...


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
