from __future__ import annotations


__all__ = ["Distribution", "RandomVar"]

import abc
from typing import (
    Generic,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

import mpmath

from hall._event import EventEq, EventInterval
from hall.analysis import Function, Interval
from hall.numbers import (
    CleanNumber,
    ComplexType,
    FloatType,
    IntType,
    Number,
    Probability,
    clean_number,
    is_number,
)


@runtime_checkable
class BaseDistribution(Function[Number, Probability], Protocol[Number]):
    __slots__ = ()

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


class Distribution(BaseDistribution[Number], Protocol[Number]):
    __slots__ = ()

    def __invert__(self) -> RandomVar[Number]:
        return RandomVar(self)


_MapValue = TypeVar("_MapValue", IntType, FloatType, ComplexType)


@runtime_checkable
class BaseRandomVar(BaseDistribution[Number], Protocol[Number]):
    __slots__ = ()

    # discrete events

    def __eq__(self, other) -> EventEq:  # type: ignore
        if isinstance(other, BaseRandomVar):
            return self - other == 0

        return EventEq(self, other)

    def __ne__(self, other) -> EventEq[Number]:  # type: ignore
        if isinstance(other, BaseRandomVar):
            return self - other != 0

        return EventEq(self, other, _inv=True)

    def __invert__(self) -> EventEq[Number]:
        return EventEq(self, 0)

    # continuous events

    def __lt__(self, other) -> EventInterval[Number]:
        if not is_number(other):
            raise TypeError(type(other).__name__)

        return EventInterval(self, b=clean_number(other - self.__support__.d))

    def __le__(self, other) -> EventInterval[Number]:
        if not is_number(other):
            raise TypeError(type(other).__name__)

        return EventInterval(self, b=other)

    def __gt__(self, other) -> EventInterval[Number]:
        if not is_number(other):
            raise TypeError(type(other).__name__)

        return EventInterval(self, a=other)

    def __ge__(self, other) -> EventInterval[Number]:
        if not is_number(other):
            raise TypeError(type(other).__name__)

        if self.__discrete__:
            return EventInterval(self, a=other + 1)

        return EventInterval(self, a=other)  # TODO some tiny amount

    # algebra

    def __add__(self, other) -> MappedRandomVar[Number, _MapValue]:
        if is_number(other):
            return self._as_mapped(add=clean_number(other))

        return NotImplemented

    def __sub__(self, other) -> MappedRandomVar[Number, _MapValue]:
        if is_number(other):
            return self._as_mapped(add=clean_number(-other))

        return NotImplemented

    def __mul__(self, other) -> MappedRandomVar[Number, _MapValue]:
        if is_number(other):
            return self._as_mapped(mul=clean_number(other))

        return NotImplemented

    def __truediv__(self, other) -> MappedRandomVar[Number, _MapValue]:
        if is_number(other):
            return self._as_mapped(div=clean_number(other))

        return NotImplemented

    def __radd__(self, other) -> MappedRandomVar[Number, _MapValue]:
        return self.__add__(other)

    def __rsub__(self, other) -> MappedRandomVar[Number, _MapValue]:
        return self.__neg__().__add__(other)

    def __rmul__(self, other) -> MappedRandomVar[Number, _MapValue]:
        return self.__mul__(other)

    def __neg__(self) -> MappedRandomVar[Number, _MapValue]:
        return self._as_mapped(mul=-1)

    # utility

    def __contains__(self, x) -> bool:
        """returns True iff x is a valid outcome"""
        return x in self.__support__

    @abc.abstractmethod
    def _as_mapped(
        self,
        *,
        add: Optional[_MapValue] = None,
        mul: Optional[_MapValue] = None,
        div: Optional[_MapValue] = None,
    ) -> MappedRandomVar[Number, _MapValue]:
        ...


class RandomVar(BaseRandomVar[Number], Generic[Number]):
    __slots__ = ("distribution",)  # noqa

    distribution: Distribution[Number]

    def __init__(self, distribution: Distribution[Number], /):
        self.distribution = distribution

    @property
    def __discrete__(self) -> bool:
        return self.distribution.__discrete__

    @property
    def __support__(self) -> Interval[Number]:
        return self.distribution.__support__

    @property
    def mean(self) -> Number:
        return self.distribution.mean

    @property
    def variance(self) -> Number:
        return self.distribution.variance

    def f(self, x: CleanNumber) -> Probability:
        return self.distribution.f(x)

    def F(self, x: CleanNumber) -> Probability:
        return self.distribution.F(x)

    def G(self, y: Probability) -> Number:
        return self.distribution.G(y)

    def _as_mapped(
        self,
        *,
        add: Optional[_MapValue] = None,
        mul: Optional[_MapValue] = None,
        div: Optional[_MapValue] = None,
    ) -> MappedRandomVar[Number, _MapValue]:
        kwargs = {}
        if add is not None:
            kwargs["add"] = add
        if mul is not None:
            kwargs["mul"] = mul
        if div is not None:
            kwargs["div"] = div

        return MappedRandomVar(self, **kwargs)


_NumberV = TypeVar(
    "_NumberV", IntType, FloatType, ComplexType, contravariant=True
)
_NumberR = TypeVar(
    "_NumberR", IntType, FloatType, ComplexType, contravariant=True
)


class MappedRandomVar(BaseRandomVar[_NumberR], Generic[_NumberV, _NumberR]):
    """Linearly mapped random var, i.e. Y -> aX + b"""

    __arg__: RandomVar[_NumberV]

    __add: _NumberR
    __mul: _NumberR
    __div: _NumberR  # kept separately to avoid rounding errors

    def __init__(
        self,
        X: RandomVar[_NumberV],
        /,
        *,
        add: _NumberR = 0,
        mul: _NumberR = 1,
        div: _NumberR = 1,
    ):
        self.__arg__ = X

        if not is_number(add):
            raise TypeError("addend must be a number")
        if not is_number(mul):
            raise TypeError("multiplier must be a number")
        if not is_number(div):
            raise TypeError("divisor must be a number")

        self.__add = add
        self.__mul = mul
        self.__div = div

    @property
    def __discrete__(self) -> bool:
        return self.__arg__.__discrete__

    @property
    def __support__(self) -> Interval[_NumberR]:
        a = self.__arg__.__support__.a
        b = self.__arg__.__support__.b
        return Interval(self._apply(a), self._apply(b))

    @property
    def mean(self) -> _NumberR:
        return self._apply(self.__arg__.mean)

    @property
    def variance(self) -> _NumberR:
        a = mpmath.fraction(self.__add, self.__div)
        return cast(_NumberR, self.__arg__.variance * a * a)

    def f(self, x: _NumberR) -> Probability:
        return self.__arg__.f(self._unapply(x))

    def F(self, x: _NumberR) -> Probability:
        return self.__arg__.F(self._unapply(x))

    def G(self, y: Probability) -> _NumberR:
        return self._apply(self.__arg__.G(y))

    def _apply(self, x: _NumberV) -> _NumberR:
        return cast(
            _NumberR, x * mpmath.fraction(self.__mul, self.__div) + self.__add
        )

    def _unapply(self, y: _NumberR) -> _NumberV:
        x = mpmath.fraction((y - self.__add) * self.__div, self.__mul)
        if self.__discrete__:
            return IntType(round(x))
        return cast(_NumberV, clean_number(x))

    def _as_mapped(
        self,
        *,
        add: Optional[_MapValue] = None,
        mul: Optional[_MapValue] = None,
        div: Optional[_MapValue] = None,
    ) -> MappedRandomVar[Number, _MapValue]:
        kwargs = dict(add=self.__add, mul=self.__mul, div=self.__div)
        if add is not None:
            kwargs["add"] += add
        if mul is not None:
            kwargs["add"] *= mul
            kwargs["mul"] *= mul
        if div is not None:
            kwargs["add"] /= div  # type: ignore
            kwargs["div"] = div

        return MappedRandomVar(self.__arg__, **kwargs)
