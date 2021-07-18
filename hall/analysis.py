from __future__ import annotations


__all__ = ["Function", "Interval", "DiscreteInterval", "supp", "convolve"]

import abc
import numbers
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

import mpmath

from hall.typing import C, Complex, Z, is_integral


def _cmax(*args: C) -> C:
    res = None
    for arg in args:
        if res is None:
            res = arg
        elif isinstance(res, numbers.Real):  # not complex
            if arg > res:
                res = arg
        else:
            if arg.real > res.real and arg.imag > res.imag:
                res = arg
            elif arg.real > res.real:
                res = type(res)(arg.real, res.imag)
            elif arg.imag > res.imag:
                res = type(res)(res.real, arg.imag)

    if res is None:
        raise ValueError("cmax expected 1 argument, got 0")

    return res


def _cmin(*args: C) -> C:
    return cast(C, -_cmax(*(-arg for arg in args)))


class Interval(Generic[C]):
    """Closed interval (endpoints a and b are included)"""

    a: C
    b: C

    def __init__(self, a: C, b: C, /):
        if a.real > b.real:
            raise ValueError("a cannot be larger than b")
        if a.imag > b.imag:
            raise ValueError("Im(a) cannot be larger than Im(b)")

        self.a = a
        self.b = b

    def __repr__(self):
        return f"{type(self).__name__}({self.a!r}, {self.b!r})"

    def __str__(self):
        return f"[{self.a}, {self.b}]"

    def __contains__(self, x: Union[C, Interval[C], object]):
        if isinstance(x, Interval):
            # subinterval check
            return x.a in self and x.b in self
        if isinstance(x, numbers.Complex):
            return (
                self.a.real <= x.real <= self.b.real
                and self.a.imag <= x.imag <= self.b.imag
            )

        return False

    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.a == other.a and self.b == other.b
        elif self.is_degenerate and isinstance(other, numbers.Complex):
            return other == self.a

        return False

    def __lt__(self, other) -> bool:
        if isinstance(other, Interval):
            # subinterval check
            other = other.a
        if isinstance(other, numbers.Complex):
            return bool(self.b.real < other.real or self.b.imag < other.imag)

        return NotImplemented

    def __le__(self, other) -> bool:
        if isinstance(other, (Interval, numbers.Complex)):
            return self == other or self < other

        return NotImplemented

    def __gt__(self, other) -> bool:
        if isinstance(other, Interval):
            # subinterval check
            other = other.b
        if isinstance(other, numbers.Complex):
            return bool(self.a.real > other.real or self.a.real > other.imag)

        return NotImplemented

    def __ge__(self, other) -> bool:
        if isinstance(other, (Interval, numbers.Complex)):
            return self == other or self > other

        return NotImplemented

    def __and__(self, other: Interval[Any]) -> Optional[Interval[C]]:
        if isinstance(other, Interval):
            if self.isdisjoint(other):
                return None

            return type(self)(_cmax(self.a, other.a), _cmin(self.b, other.b))
        return NotImplemented

    def __or__(self, other: Interval[C]) -> Interval[C]:
        if isinstance(other, Interval):
            if self.isdisjoint(other):
                # TODO mask the gap, or create a disjoint interval class
                raise NotImplementedError

            return type(self)(_cmin(self.a, other.a), _cmax(self.b, other.b))
        return NotImplemented

    def __add__(self, other: Interval[C]) -> Interval[C]:
        return type(self)(self.a + other.a, self.b + other.b)

    def __bool__(self) -> bool:
        return True

    def __hash__(self):
        return hash((self.a, self.b))

    @property
    def is_bounded(self) -> bool:
        return bool(
            mpmath.isfinite(self.a.real)
            and mpmath.isfinite(self.a.imag)
            and mpmath.isfinite(self.b.real)
            and mpmath.isfinite(self.b.imag)
        )

    @property
    def is_degenerate(self) -> bool:
        return bool(self.a == self.b)

    @property
    def size(self) -> Union[C, int]:  # lower type bound for bool
        return self.b - self.a

    @property
    def radius(self) -> Union[C, float]:  # lower type bound for int
        return self.size / 2

    @property
    def mid(self) -> Union[C, float]:  # lower type bound for bool
        return (self.a + self.b) / 2

    def isdisjoint(self, other: Interval[Any]) -> bool:
        return bool(
            self.a.real > other.b.real
            or self.a.imag > other.b.imag
            or self.b.real < other.a.real
            or self.a.imag < other.a.imag
        )


class DiscreteInterval(Interval[Z], Generic[Z]):
    def __contains__(self, x: Union[Z, Interval[Z], object]):
        if not isinstance(x, DiscreteInterval) and not is_integral(x):
            return False

        return super().__contains__(x)


_C_co = TypeVar("_C_co", bound=Complex, covariant=True)


@runtime_checkable
class Function(Protocol[C, _C_co]):
    __slots__ = ()

    __discrete__: ClassVar[bool]

    @property
    @abc.abstractmethod
    def __support__(self) -> Interval[C]:
        ...

    @abc.abstractmethod
    def f(self, x: C) -> _C_co:
        ...


class _FunctionAlias(Function[C, _C_co], Generic[C, _C_co]):
    __slots__ = ("__wrapped__", "__support")  # noqa

    __discrete__: ClassVar[bool] = False
    __support: Interval[C]

    def __init__(self, fn: Callable[[C], _C_co], support: Interval[C]):
        self.__wrapped__: Callable[[C], _C_co] = fn
        self.__support = support

    @property
    def __support__(self) -> Interval[C]:
        return self.__support

    def f(self, x: C) -> _C_co:
        return self.__wrapped__(x)


class _DiscreteFunctionAlias(_FunctionAlias[Z, _C_co], Generic[Z, _C_co]):
    __discrete__ = True


def supp(f: Function[C, Any]) -> Interval[C]:
    return f.__support__


def convolve(
    f: Function[C, _C_co], g: Function[C, _C_co]
) -> Function[C, _C_co]:
    """
    Convolution (lazy):
    https://en.wikipedia.org/wiki/Convolution
    """
    res_support = f.__support__ + g.__support__

    fa, fb = f.__support__.a, f.__support__.b
    ga, gb = g.__support__.a, g.__support__.b

    integrator: Callable[[Callable[[C], _C_co], List[C]], _C_co]
    if f.__discrete__ and g.__discrete__:
        integrator = mpmath.nsum
    else:
        integrator = mpmath.quad

    def res(n: C) -> _C_co:
        if isinstance(res, numbers.Real):
            # clip the support bounds
            a = fb if n - ga > fb else ga
            b = fa if n - gb < fa else gb
        else:
            a, b = ga, gb

        def dres(m: C):
            return mpmath.fmul(f.f(n - m), g.f(m))

        return integrator(dres, [a, b])

    if f.__discrete__ and g.__discrete__:
        return _DiscreteFunctionAlias(res, res_support)
    else:
        return _FunctionAlias(res, res_support)
