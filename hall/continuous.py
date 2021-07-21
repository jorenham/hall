__all__ = [
    "Normal",
]

import abc
from typing import ClassVar, Protocol

import mpmath

from hall import Interval
from hall._core import Distribution
from hall.numbers import (
    AnyFloat,
    FloatType,
    Probability,
    clean_number,
    is_float,
    is_int,
)


class DistributionC(Distribution[FloatType], Protocol):
    __discrete__: ClassVar[bool] = False  # noqa

    @abc.abstractmethod
    def pdf(self, x: FloatType) -> Probability:
        """Probability Density Function"""
        raise NotImplementedError

    @abc.abstractmethod
    def cdf(self, x: FloatType) -> Probability:
        """Cumulative Density/Distribution Function"""
        raise NotImplementedError

    def cdf_inv(self, y: Probability) -> FloatType:
        """Inverse of the cdf"""
        # TODO
        #  https://mpmath.org/doc/current/calculus/optimization.html#mpmath.calculus.optimization.Ridder
        raise NotImplementedError

    def f(self, x: FloatType) -> Probability:
        if not is_float(x) and not is_int(x):
            raise TypeError("value must be an real number")

        if x not in self.__support__:
            return mpmath.mpf(0)

        return self.pdf(x)

    def F(self, x: FloatType) -> Probability:
        if not is_float(x) and not is_int(x):
            raise TypeError("value must be an real number")

        support = self.__support__
        if x < support:
            return mpmath.mpf(0)
        if x > support:
            return mpmath.mpf(1)

        return self.cdf(x)

    def G(self, y: Probability) -> FloatType:
        if not (0 <= y <= 1):
            raise TypeError("probability must be between 0 and 1 inclusively")

        return self.cdf_inv(y)


class Normal(DistributionC):
    __slots__ = ("mu", "sigma")

    mu: mpmath.mpf
    sigma: mpmath.mpf

    def __init__(
        self,
        mu: AnyFloat = mpmath.mpf(0),
        sigma: AnyFloat = mpmath.mpf(1),
    ):
        if sigma < 0:
            raise ValueError("sigma must be positive")

        self.mu = clean_number(mu)
        self.sigma = clean_number(sigma)

        super().__init__()

    @property
    def __support__(self):
        return Interval(mpmath.mpf("-inf"), mpmath.mpf("+inf"))

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.sigma * self.sigma

    def pdf(self, x: AnyFloat) -> Probability:
        return mpmath.npdf(x, self.mu, self.sigma)

    def cdf(self, x: AnyFloat) -> Probability:
        return mpmath.mp.ncdf(x, self.mu, self.sigma)

    def cdf_inv(self, y: Probability) -> FloatType:
        return self.mu + self.sigma * mpmath.sqrt(2) * mpmath.erfinv(2 * y - 1)
