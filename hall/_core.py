from __future__ import annotations


__all__ = ["RandomVar", "Distribution"]

import abc
import numbers
from typing import Final, Generic, Protocol, Union, runtime_checkable

from hall._event import Event, EventEq, EventInterval
from hall.analysis import Function, Interval
from hall.numbers import (
    CleanNumber,
    FloatType,
    IntType,
    Number,
    Probability,
    clean_number,
    is_number,
)


class RandomVar(Generic[Number]):
    __slots__ = (
        "distribution",
        "_addend",
        "_multiplier",
        "_exponent",
    )

    distribution: Distribution[Number]

    # constant arithmatic
    _addend: Final[CleanNumber]
    _multiplier: Final[CleanNumber]
    _exponent: Final[IntType]

    def __init__(
        self,
        distribution: Distribution[Number],
        /,
        _addend: CleanNumber = 0,
        _multiplier: CleanNumber = 1,
        _exponent: IntType = 1,
    ):
        self.distribution = distribution

        if not isinstance(_addend, numbers.Number):
            raise TypeError("addend must be a constant number")
        if not isinstance(_multiplier, numbers.Number):
            raise TypeError("multiplier must be a constant number")
        if not isinstance(_exponent, numbers.Integral):
            raise TypeError("exponent must be a constant integer")
        if _exponent < 1:
            raise ValueError("exponent must be a non-zero positive integer")

        self._addend = clean_number(_addend)
        self._multiplier = clean_number(_multiplier)
        self._exponent = clean_number(_exponent)

    # discrete events

    def __eq__(self, other) -> EventEq[Number]:  # type: ignore
        if isinstance(other, RandomVar):
            return self - other == 0

        return EventEq(self, other)

    def __ne__(self, other) -> EventEq[Number]:  # type: ignore
        if isinstance(other, RandomVar):
            return self - other != 0

        return EventEq(self, other, _inv=True)

    def __invert__(self: RandomVar[Number]) -> EventEq[Number]:
        return EventEq(self, 0)

    # continuous events

    def __lt__(self: RandomVar[Number], other: Number) -> EventInterval[Number]:
        if self.__discrete__:
            return EventInterval(self, b=other - 1)

        return EventInterval(self, b=other)  # TODO some tiny amount

    def __le__(self: RandomVar[Number], other: Number) -> EventInterval[Number]:
        return EventInterval(self, b=other)

    def __gt__(self: RandomVar[Number], other: Number) -> EventInterval[Number]:
        return EventInterval(self, a=other)

    def __ge__(self: RandomVar[Number], other: Number) -> EventInterval[Number]:
        if self.__discrete__:
            return EventInterval(self, a=other + 1)

        return EventInterval(self, a=other)  # TODO some tiny amount

    # conditional probability

    def __or__(
        self: RandomVar[Number], other: Union[Event[Number], RandomVar[Number]]
    ) -> EventEq[Number]:
        # TODO precedence hacking (don't collapse X | Y to X when uncorr!)
        #  P(X == x | Y == y) => P(X == (X | Y) == y)
        #  P(X | Y == y) => P((X | Y) == y)
        #  P(X == x | Y) => P(X == (x | Y))
        #  P(X, Y | Z) => P(X, (Y | Z))
        return NotImplemented

    # algebra

    def __add__(self, other) -> RandomVar[Number]:
        if is_number(other):
            return type(self)(
                self.distribution,
                _addend=self._addend + clean_number(other),
                _multiplier=self._multiplier,
                _exponent=self._exponent,
            )

        return NotImplemented

    def __sub__(self, other) -> RandomVar[Number]:
        if is_number(other):
            return self.__add__(-other)

        return NotImplemented

    def __mul__(self, other) -> RandomVar[Number]:
        if is_number(other):
            x = clean_number(other)
            return type(self)(
                self.distribution,
                _addend=self._addend * x,
                _multiplier=self._multiplier * x,
                _exponent=self._exponent,
            )

        return NotImplemented

    def __truediv__(self, other) -> RandomVar[Number]:
        if is_number(other):
            x = clean_number(other)
            return type(self)(
                self.distribution,
                _addend=self._addend / x,
                _multiplier=self._multiplier / x,
                _exponent=self._exponent,
            )

        return NotImplemented

    def __radd__(self, other) -> RandomVar[Number]:
        return self.__add__(other)

    def __rsub__(self, other) -> RandomVar[Number]:
        return self.__neg__().__add__(other)

    def __rmul__(self, other) -> RandomVar[Number]:
        return self.__mul__(other)

    def __neg__(self) -> RandomVar[Number]:
        return type(self)(
            self.distribution,
            _addend=-self._addend,
            _multiplier=-self._multiplier,
            _exponent=self._exponent,
        )

    # utility

    def __contains__(self, x) -> bool:
        """returns True iff x is a valid outcome"""
        return x in self.outcomes

    @property
    def __discrete__(self) -> bool:
        return self.distribution.__discrete__

    @property
    def outcomes(self) -> Interval:
        a = self.distribution.__support__.a
        b = self.distribution.__support__.b
        return Interval(self._transform(a), self._transform(b))

    def f(self, x: CleanNumber) -> Probability:
        return self.distribution.f(self._transform_inv(x))

    def F(self, x: CleanNumber) -> Probability:
        return self.distribution.F(self._transform_inv(x))

    def G(self, y: Probability) -> CleanNumber:
        return self._transform(self.distribution.G(y))

    @property
    def mean(self) -> Union[Number, FloatType]:
        # TODO exponent
        return self.distribution.mean + self._addend

    @property
    def variance(self) -> Union[Number, FloatType]:
        # TODO exponent
        return self.distribution.variance * self._multiplier * self._multiplier

    def _transform(self, x: CleanNumber) -> CleanNumber:
        return x ** self._exponent * self._multiplier + self._addend

    def _transform_inv(self, y: CleanNumber) -> CleanNumber:
        x = ((y - self._addend) / self._multiplier) ** 1 / self._exponent
        if self.__discrete__:
            return IntType(round(x))
        return x


@runtime_checkable
class Distribution(Function[Number, Probability], Protocol[Number]):
    __slots__ = ()

    def __invert__(self) -> RandomVar[Number]:
        return RandomVar(self)

    @property
    @abc.abstractmethod
    def mean(self) -> Union[Number, FloatType]:
        """
        The mean/expected value of the distribution
        """
        ...

    @property
    @abc.abstractmethod
    def variance(self) -> Union[Number, FloatType]:
        """
        The variance of the distribution
        """
        ...

    @abc.abstractmethod
    def f(self, x: Number) -> Probability:
        """
        Probability Mass/Density Function (PMF/PDF) that must integrate to 1.
        """
        ...

    @abc.abstractmethod
    def F(self, x: Number) -> Probability:
        """
        Cumulative Distribution Function (CDF); the integral of `f`.
        """
        ...

    @abc.abstractmethod
    def G(self, y: Probability) -> Number:
        """
        Percent Point Function (PPF); the inverse of the CDF `F`
        """
        ...
