__all__ = ["P", "E", "Var", "Std", "Cov", "Corr"]

import functools
from typing import Any, Callable, Generic, Tuple, TypeVar, Union

import mpmath

from hall._core import Stochast
from hall._types import C, Complex, Float, Probability
from hall.event import Event, EventEq


_F = TypeVar("_F", bound=Callable[..., Complex])


class Operator(Generic[_F]):
    __wrapped__: _F

    def __init__(self, __func: _F, /):
        self.__wrapped__ = __func
        functools.update_wrapper(self, self.__wrapped__)

    def __call__(self, *__vals) -> Complex:
        return self.__wrapped__(*__vals)

    def __getitem__(self, __vals: Union[Tuple[Any, ...], Any]) -> Complex:
        if isinstance(__vals, tuple):
            return self.__wrapped__(*__vals)
        return self.__wrapped__(__vals)


@Operator
def P(event: Union[Event[C], Stochast[C]]) -> Probability:
    """
    Probability measure for the given event, i.e. the probability that the
    event occurs:
    https://en.wikipedia.org/wiki/Probability
    """
    if isinstance(event, Stochast):
        # P[X] -> P[X != 0]
        event = EventEq(event, 0, _inv=True)

    return event.p


@Operator
def E(X: Stochast[C]) -> Union[C, Float]:
    """
    Expected value: https://en.wikipedia.org/wiki/Expected_value
    """
    return X.distribution.mean


@Operator
def Var(X: Stochast[C]) -> Union[C, Float]:
    """
    Variance:
    https://en.wikipedia.org/wiki/Variance
    """
    return X.distribution.variance


@Operator
def Std(X: Stochast[C]) -> Float:
    """
    Standard deviation:
    https://en.wikipedia.org/wiki/Standard_deviation
    """
    return mpmath.mp.sqrt(X.distribution.variance)


@Operator
def Cov(X: Stochast[C], Y: Stochast[C]) -> Float:
    """
    Covariance:
    https://en.wikipedia.org/wiki/Covariance
    """
    if X is Y:
        return Var(X)

    # TODO actually calculate one joint distributions have been implemented:
    #  return E[X * Y] - E[X] * E[Y]

    return mpmath.mpf(0)


@Operator
def Corr(X: Stochast[C], Y: Stochast[C]) -> Float:
    """
    Pearson correlation coefficient:
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    """
    if X is Y:
        return mpmath.mpf(1)

    return Cov(X, Y) / mpmath.fmul(Std(X), Std(Y))
