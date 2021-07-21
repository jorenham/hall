__all__ = [
    "Bernoulli",
    "Binomial",
    "Uniform",
]

import abc
from typing import ClassVar, Final, Protocol, Union

import mpmath

from hall import DiscreteInterval as Interval
from hall import Distribution
from hall.numbers import (
    AnyInt,
    FloatType,
    IntType,
    Probability,
    clean_number,
    is_int,
    is_probability,
)


class DistributionD(Distribution[IntType], Protocol):
    __discrete__: ClassVar[bool] = True  # noqa

    @abc.abstractmethod
    def pmf(self, x: IntType) -> Probability:
        """Probability Mass Function"""
        ...

    @abc.abstractmethod
    def cmf(self, x: IntType) -> Probability:
        """Cumulative Mass Function"""
        ...

    def cmf_inv(self, y: Probability) -> IntType:
        """Inverse of the cmf"""
        support = self.__support__
        if not support.is_bounded:
            raise NotImplementedError

        x_prev = 0
        for x in range(support.a, support.b + 1):
            # noinspection PyTypeChecker
            yi = self.cmf(x)
            assert mpmath.mp.zero <= yi <= mpmath.mp.one

            if yi > y:
                break

            x_prev = x

        # noinspection PyTypeChecker
        return x_prev

    def f(self, x: IntType) -> Probability:
        if not is_int(x):
            raise TypeError("value must be an integral number")

        if x not in self.__support__:
            return mpmath.mpf(0)

        return self.pmf(x)

    def F(self, x: IntType) -> Probability:
        if not is_int(x):
            raise TypeError("value must be an integral number")

        support = self.__support__
        if x < support.a:
            return mpmath.mpf(0)
        if x > support.b:
            return mpmath.mpf(1)

        return self.cmf(x)

    def G(self, y: Probability) -> IntType:
        if not (0 <= y <= 1):
            raise TypeError("probability must be between 0 and 1 inclusively")

        return self.cmf_inv(y)


class Binomial(DistributionD):
    __slots__ = ("n", "p")

    n: IntType
    p: Final[Probability]

    def __init__(self, n: IntType, p: Probability):
        if not is_int(n):
            raise TypeError("n must be an integral number")
        if n < 0:
            raise ValueError("n must be postive")
        if not is_probability(p):
            raise ValueError("p must lie between 0 and 1 inclusively")

        self.n = clean_number(n)
        self.p = clean_number(p)

        super().__init__()

    @property
    def __support__(self) -> Interval:
        return Interval(0, self.n)

    @property
    def q(self) -> Probability:
        return FloatType(1) - self.p

    @property
    def mean(self) -> FloatType:
        return self.n * self.p

    @property
    def variance(self) -> FloatType:
        return self.n * self.p * self.q

    def pmf(self, x: IntType) -> Probability:
        if x < self.__support__.a:
            raise TypeError(f"x={x} must be larger than {self.__support__.a}")

        return mpmath.binomial(self.n, x) * self.p ** x * self.q ** (self.n - x)

    def cmf(self, x: IntType) -> Probability:
        if x < self.__support__.a:
            raise TypeError(f"x must be larger than {self.__support__.a}")

        return mpmath.betainc(self.n - x, x + 1, 0, self.q, regularized=True)


class Bernoulli(Binomial):
    __slots__ = ()

    def __init__(self, p: Probability):
        # noinspection PyTypeChecker
        super().__init__(1, p)

    def cmf_inv(self, y: Probability) -> int:
        if y <= mpmath.mp.one - self.p:
            return 0

        return 1


class Uniform(DistributionD):
    __slots__ = ("a", "b")

    a: Final[IntType]
    b: Final[IntType]

    def __init__(self, a: AnyInt, b: AnyInt):
        if not is_int(a):
            raise TypeError("a must be an integral number")
        if not is_int(b):
            raise TypeError("b must be an integral number")
        if b <= a:
            raise ValueError("a must be strictly less then b")

        self.a = IntType(a)
        self.b = IntType(b)

        super().__init__()

    @property
    def __support__(self) -> Interval:
        return Interval(self.a, self.b)

    @property
    def n(self) -> int:
        return self.b - self.a + 1

    @property
    def mean(self) -> Union[IntType, FloatType]:
        if self.n % 2 == 1:
            return (self.a + self.b) // 2
        else:
            return (self.a + self.b) / 2

    @property
    def variance(self) -> FloatType:
        return (self.n ** 2 - 1) / 12

    def pmf(self, x: IntType) -> Probability:
        # if x in self.x:
        if x in self.__support__:
            return mpmath.fraction(1, self.n)

        return FloatType(0)

    def cmf(self, x: IntType) -> Probability:
        # if x in self.x:
        if self.a <= x <= self.b:
            return (x - self.a + 1) / mpmath.mpf(self.n)
        return FloatType(0)

    def cmf_inf(self, y: Probability) -> IntType:
        return int(self.n * y + self.a - 1)
