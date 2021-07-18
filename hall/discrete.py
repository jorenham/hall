__all__ = [
    "Bernoulli",
    "Binomial",
    "Uniform",
]

import abc
import numbers
from typing import ClassVar, Final, Protocol, Union

import mpmath

from hall import DiscreteInterval as Interval
from hall import Distribution
from hall.typing import Float, Integral, Probability, Z, is_probability


class DistributionD(Distribution[Z], Protocol[Z]):
    __discrete__: ClassVar[bool] = True

    @abc.abstractmethod
    def pmf(self, x: Z) -> Probability:
        """Probability Mass Function"""
        ...

    @abc.abstractmethod
    def cmf(self, x: Z) -> Probability:
        """Cumulative Mass Function"""
        ...

    def cmf_inv(self, y: Probability) -> Z:
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

    def f(self, x: Z) -> Probability:
        if not isinstance(x, numbers.Integral):
            raise TypeError("value must be an integral number")

        if x not in self.__support__:
            return mpmath.mpf(0)

        return self.pmf(x)

    def F(self, x: Z) -> Probability:
        if not isinstance(x, numbers.Integral):
            raise TypeError("value must be an integral number")

        support = self.__support__
        if x < support:
            return mpmath.mpf(0)
        if x > support:
            return mpmath.mpf(1)

        return self.cmf(x)

    def G(self, y: Probability) -> Z:
        if not (0 <= y <= 1):
            raise TypeError("probability must be between 0 and 1 inclusively")

        return self.cmf_inv(y)


class Binomial(DistributionD[Z]):
    __slots__ = ("n", "p")

    n: Z
    p: Final[Probability]

    def __init__(self, n: Z, p: Probability):
        if not isinstance(n, numbers.Integral):
            raise TypeError("n must be an integral number")
        if n < 0:
            raise ValueError("n must be postive")
        if not is_probability(p):
            raise ValueError("p must lie between 0 and 1 inclusively")

        self.n = n
        self.p = mpmath.mpf(p)

        super().__init__()

    @property
    def __support__(self) -> Interval[Z]:
        return Interval(type(self.n)(0), self.n)

    @property
    def q(self) -> Probability:
        return mpmath.mpf(1) - self.p

    @property
    def mean(self) -> Float:
        return self.n * self.p

    @property
    def variance(self) -> Float:
        return self.n * self.p * self.q

    def pmf(self, x: Integral) -> Probability:
        if x < self.__support__:
            raise TypeError(f"x must be larger than {self.__support__.a}")
        if self.__support__ < x:
            return mpmath.mpf(0)

        return mpmath.binomial(self.n, x) * self.p ** x * self.q ** (self.n - x)

    def cmf(self, x: Integral) -> Probability:
        if x < self.__support__:
            raise TypeError(f"x must be larger than {self.__support__.a}")
        if self.__support__ < x:
            return mpmath.mpf(1)

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


class Uniform(DistributionD[Z]):
    __slots__ = ("a", "b")

    a: Final[int]
    b: Final[int]

    def __init__(self, a: Z, b: Z):
        if not isinstance(a, numbers.Integral):
            raise TypeError("a must be an integral number")
        if not isinstance(b, numbers.Integral):
            raise TypeError("b must be an integral number")
        if b <= a:
            raise ValueError("a must be strictly less then b")

        self.a = int(a)
        self.b = int(b)

        super().__init__()

    @property
    def __support__(self) -> Interval[Z]:
        return Interval(self.a, self.b)

    @property
    def n(self) -> int:
        return self.b - self.a + 1

    @property
    def mean(self) -> Union[int, Float]:
        if self.n % 2 == 0:
            return (self.a + self.b) // 2
        else:
            return mpmath.fraction(self.a + self.b, 2)

    @property
    def variance(self) -> Float:
        return mpmath.fraction(self.n ** 2 - 1, 12)

    def pmf(self, x: Integral) -> Probability:
        # if x in self.x:
        if x in self.__support__:
            return mpmath.fraction(1, self.n)

        return mpmath.mpf(0)

    def cmf(self, x: Integral) -> Probability:
        # if x in self.x:
        if self.a <= x <= self.b:
            return (x - self.a + 1) / mpmath.mpf(self.n)
        return mpmath.mp.zero

    def cmf_inf(self, y: Probability) -> int:
        return int(self.n * y + self.a - 1)
