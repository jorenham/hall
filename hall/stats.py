__all__ = ["P", "E", "Var", "Std", "Cov", "Corr", "sample"]

import functools
from typing import Any, Callable, Generic, Tuple, TypeVar, Union

import mpmath

from hall._core import BaseRandomVar  # noqa
from hall._event import Event, EventEq
from hall.numbers import CleanNumber, FloatType, Number, Probability


_F = TypeVar("_F", bound=Callable[..., CleanNumber])


class Operator(Generic[_F]):
    __wrapped__: _F

    def __init__(self, __func: _F, /):
        self.__wrapped__ = __func
        functools.update_wrapper(self, self.__wrapped__)

    def __call__(self, *__vals) -> CleanNumber:
        return self.__wrapped__(*__vals)

    def __getitem__(self, __vals: Union[Tuple[Any, ...], Any]) -> CleanNumber:
        if isinstance(__vals, tuple):
            return self.__wrapped__(*__vals)
        return self.__wrapped__(__vals)


@Operator
def P(event: Union[Event[Number], BaseRandomVar[Number]]) -> Probability:
    """
    Probability measure for the given event, i.e. the probability that the
    event occurs:
    https://en.wikipedia.org/wiki/Probability
    """
    if isinstance(event, BaseRandomVar):
        # P[X] -> P[X != 0]
        event = EventEq(event, 0, _inv=True)

    return event.p


@Operator
def E(X: BaseRandomVar[Number]) -> Union[Number, FloatType]:
    """
    Expected value: https://en.wikipedia.org/wiki/Expected_value
    """
    return X.mean


@Operator
def Var(X: BaseRandomVar[Number]) -> Union[Number, FloatType]:
    """
    Variance:
    https://en.wikipedia.org/wiki/Variance
    """
    return X.variance


@Operator
def Std(X: BaseRandomVar[Number]) -> Union[Number, FloatType]:
    """
    Standard deviation:
    https://en.wikipedia.org/wiki/Standard_deviation
    """
    return mpmath.mp.sqrt(X.variance)


@Operator
def Cov(
    X: BaseRandomVar[Number],
    Y: BaseRandomVar[Number],
) -> Union[Number, FloatType]:
    """
    Covariance:
    https://en.wikipedia.org/wiki/Covariance
    """
    if X is Y:
        return Var(X)

    # TODO actually calculate one joint distributions have been implemented:
    #  return E[X * Y] - E[X] * E[Y]

    return FloatType(0)


@Operator
def Corr(
    X: BaseRandomVar[Number], Y: BaseRandomVar[Number]
) -> Union[Number, FloatType]:
    """
    Pearson correlation coefficient:
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    """
    if X is Y:
        return FloatType(1)

    return Cov(X, Y) / mpmath.fmul(Std(X), Std(Y))


def sample(X: BaseRandomVar[Number]) -> Number:
    return X.G(mpmath.rand())
