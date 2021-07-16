__all__ = ["P", "E", "Var", "Std"]

import functools
from typing import Any, Callable, Generic, Tuple, TypeVar, Union

import mpmath

from hall._core import Stochast
from hall._types import C, Complex
from hall._types import Float
from hall._types import Float as Probability
from hall.event import Event


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
def P(event: Event[C]) -> Probability:
    """
    Probability measure for the given event, i.e. the probability that the
    event occurs
    """
    return event.p


@Operator
def E(X: Stochast[C]) -> Union[C, Float]:
    """Expected value"""
    return X.distribution.mean


@Operator
def Var(X: Stochast[C]) -> Union[C, Float]:
    """Variance"""
    return X.distribution.variance


@Operator
def Std(X: Stochast[C]) -> Union[C, Float]:
    """Standard deviation"""
    return mpmath.mp.sqrt(X.distribution.variance)


# TODO Covariance (Cov)
#  https://en.wikipedia.org/wiki/Covariance

# TODO Correlation
#  https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
