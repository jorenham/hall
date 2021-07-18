from __future__ import annotations


__all__ = ["Stochast", "Distribution"]

import abc
from typing import Generic, Protocol, Union, runtime_checkable

from hall.analysis import Function, Interval
from hall.event import Event, EventEq, EventInterval
from hall.typing import C, Float, Probability


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

    def __lt__(self: Stochast[C], other: C) -> EventInterval[C]:
        if self.distribution.__discrete__:
            return EventInterval(self, b=other - 1)
        else:
            return EventInterval(self, b=other)  # TODO some tiny amount

    def __le__(self: Stochast[C], other: C) -> EventInterval[C]:
        return EventInterval(self, b=other)

    def __gt__(self: Stochast[C], other: C) -> EventInterval[C]:
        return EventInterval(self, a=other)

    def __ge__(self: Stochast[C], other: C) -> EventInterval[C]:
        if self.distribution.__discrete__:
            return EventInterval(self, a=other + 1)
        else:
            return EventInterval(self, a=other)  # TODO some tiny amount

    def __invert__(self: Stochast[C]) -> EventEq[C]:
        return EventEq(self, 0)

    def __or__(
        self: Stochast[C], other: Union[Event[C], Stochast[C]]
    ) -> EventEq[C]:
        # TODO precedence hacking (don't collapse X | Y to X when uncorr!)
        #  P(X == x | Y == y) => P(X == (X | Y) == y)
        #  P(X | Y == y) => P((X | Y) == y)
        #  P(X == x | Y) => P(X == (x | Y))
        #  P(X, Y | Z) => P(X, (Y | Z))
        return NotImplemented

    def __contains__(self, item) -> bool:
        return item in self.distribution.__support__

    @property
    def outcomes(self) -> Interval[C]:
        return self.distribution.__support__


@runtime_checkable
class Distribution(Function[C, Probability], Protocol[C]):
    __slots__ = ()

    def __invert__(self) -> Stochast[C]:
        return Stochast(self)

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
