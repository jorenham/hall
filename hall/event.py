from __future__ import annotations


__all__ = [
    "Event",
    "EventEq",
    "EventInterval",
]

import abc
from typing import TYPE_CHECKING, Final, Optional, Protocol

import mpmath

from hall._types import C
from hall._types import Float as Probability
from hall._types import is_probability


if TYPE_CHECKING:
    from hall._core import Stochast


class Event(Protocol[C]):
    __slots__ = ()

    X: Stochast[C]

    def __float__(self) -> float:
        """return the probability of the event occurring"""
        return float(self.p)

    def __bool__(self) -> bool:
        """Returns `True` with probability `self.p`, otherwise `False`"""
        return bool(mpmath.rand() < self.p)

    def __int__(self) -> int:
        return int(bool(self))

    @property
    @abc.abstractmethod
    def p(self) -> Probability:
        """return the probability of the event occurring"""
        ...


class EventEq(Event[C]):
    """
    Symbolic (in)equality expression of a stochast and a constant.
    """

    __slots__ = ("X", "x", "_inv")  # noqa

    X: Stochast[C]
    x: C
    _inv: Final[bool]

    def __init__(self, X: Stochast[C], x: C, _inv: bool = False):
        self.X = X
        self.x = x
        self._inv = _inv

    def __repr__(self):
        return " ".join((str(self.X), self._op_symbol, str(self.x)))

    __str__ = __repr__

    def __invert__(self) -> EventEq[C]:
        return type(self)(self.X, self.x, not self._inv)

    def __hash__(self):
        return hash((self.X, self.x, self._inv))

    @property
    def p(self) -> Probability:
        p = self.X.distribution.f(self.x)
        if self._inv:
            return 1.0 - p
        else:
            return p

    @property
    def _op_symbol(self) -> str:
        return "≠" if self._inv else "="


class EventInterval(Event[C]):
    """
    Symbolic "greater/less than" expression of a stochast and a constant.
    """

    __slots__ = ("X", "a", "b", "_inv")  # noqa

    X: Stochast[C]
    a: Optional[C]
    b: Optional[C]
    _inv: Final[bool]

    def __init__(
        self,
        X: Stochast[C],
        a: Optional[C] = None,
        b: Optional[C] = None,
        _inv: bool = False,
    ):
        if a is None and b is None:
            raise ValueError("a or b required")

        self.X = X
        self.a = a
        self.b = b
        self._inv = _inv

    def __repr__(self):
        if self.a is None:
            res = f"{self.X} <= {self.b:.4f}"
        elif self.b is None:
            res = f"{self.X} > {self.a:.4f}"
        else:
            res = f"{self.a:.4f} < {self.X} <= {self.b:.4f}"

        if self._inv:
            return f"~({res})"

        return res

    __str__ = __repr__

    def __invert__(self) -> EventInterval:
        if self.a is None:
            return type(self)(self.X, a=self.b, _inv=self._inv)
        if self.b is None:
            return type(self)(self.X, b=self.a, _inv=self._inv)

        return type(self)(self.X, a=self.a, b=self.b, _inv=not self._inv)

    def __hash__(self):
        return hash((self.X, self.a, self.b, self._inv))

    @property
    def p(self) -> Probability:
        p: Probability
        if self.b is None:
            p = mpmath.mpf(1)
        else:
            p = self.X.distribution.F(self.b)

        if self.a is not None:
            p -= self.X.distribution.F(self.a)

        if self._inv:
            p = mpmath.mpf(1) - p

        assert is_probability(p), f"{p!r} is not a probability"

        return p
