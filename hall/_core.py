from __future__ import annotations


__all__ = [
    "Stochast",
    "Distribution",
]

import abc
from typing import ClassVar, Generic, Protocol, Union, runtime_checkable

from hall._types import C
from hall._types import Float
from hall._types import Float as Probability
from hall._types import Interval
from hall.event import EventEq, EventInterval


class Stochast(Generic[C]):
    # TODO (linear) arithmatic
    __slots__ = ("distribution",)

    distribution: Distribution[C]

    def __init__(self, distribution: Distribution[C]):
        self.distribution = distribution

    def __eq__(self, other) -> EventEq:  # type: ignore
        return EventEq(self, other)

    def __ne__(self, other) -> EventEq:  # type: ignore
        return EventEq(self, other, _inv=True)

    def __lt__(self, other) -> EventInterval:
        if self.distribution.discrete:
            return EventInterval(self, b=other - 1)
        else:
            return EventInterval(self, b=other)  # TODO some tiny amount

    def __le__(self, other) -> EventInterval:
        return EventInterval(self, b=other)

    def __gt__(self, other) -> EventInterval:
        return EventInterval(self, a=other)

    def __ge__(self, other) -> EventInterval:
        if self.distribution.discrete:
            return EventInterval(self, a=other + 1)
        else:
            return EventInterval(self, a=other)  # TODO some tiny amount

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
