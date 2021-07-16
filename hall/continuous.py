__all__ = [
    "Normal",
    "N",
]

import abc
import numbers
import statistics
from typing import ClassVar, Protocol, cast

import mpmath

from hall._core import Distribution as _Distribution
from hall._types import Float as Probability
from hall._types import Interval, R, Real


class Distribution(_Distribution[R], Protocol[R]):
    discrete: ClassVar[bool] = False

    @abc.abstractmethod
    def pdf(self, x: R) -> Probability:
        """Probability Density Function"""
        raise NotImplementedError

    @abc.abstractmethod
    def cdf(self, x: R) -> Probability:
        """Cumulative Density/Distribution Function"""
        raise NotImplementedError

    def cdf_inv(self, y: Probability) -> R:
        """Inverse of the cdf"""
        # TODO
        #  https://mpmath.org/doc/current/calculus/optimization.html#mpmath.calculus.optimization.Ridder
        raise NotImplementedError

    def f(self, x: R) -> Probability:
        if not isinstance(x, numbers.Real):
            raise TypeError("value must be an real number")

        if x not in self.support:
            return mpmath.mpf(0)

        return self.pdf(x)

    def F(self, x: R) -> Probability:
        if not isinstance(x, numbers.Real):
            raise TypeError("value must be an real number")

        support = self.support
        if x < support:
            return mpmath.mpf(0)
        if x > support:
            return mpmath.mpf(1)

        return self.cdf(x)

    def G(self, y: Probability) -> R:
        if not (0 <= y <= 1):
            raise TypeError("probability must be between 0 and 1 inclusively")

        return self.cdf_inv(y)


class Normal(Distribution[mpmath.mpf]):
    __slots__ = ("mu", "sigma")

    mu: mpmath.mpf
    sigma: mpmath.mpf

    def __init__(
        self,
        mu: Real = mpmath.mpf(0),
        sigma: Real = mpmath.mpf(1),
    ):
        if sigma < 0:
            raise ValueError("sigma must be positive")

        self.mu = mpmath.mpf(mu)
        self.sigma = mpmath.mpf(sigma)

    @property
    def support(self):
        return Interval(mpmath.mpf("-inf"), mpmath.mpf("+inf"))

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.sigma * self.sigma

    def pdf(self, x: Real) -> Probability:
        return mpmath.npdf(x, self.mu, self.sigma)

    def cdf(self, x) -> Probability:
        return mpmath.mp.ncdf(x, self.mu, self.sigma)

    def cdf_inv(self, y: Probability) -> R:
        x = statistics._normal_dist_inv_cdf(y, self.mu, self.sigma)  # type: ignore  # noqa
        return cast(R, mpmath.mpf(x))


# shortcuts

N = Normal
