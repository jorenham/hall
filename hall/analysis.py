from __future__ import annotations


__all__ = ["Function", "Interval", "DiscreteInterval", "supp", "convolve"]

import abc
from typing import (
    Callable,
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

from hall.numbers import (
    AnyNumber,
    CleanNumber,
    ComplexType,
    FloatType,
    IntType,
    Number,
    clean_number,
    is_complex,
    is_float,
    is_int,
    is_number,
)


def _cmax(*args: Number) -> Number:
    res = None
    for arg in args:
        if res is None:
            res = arg
        elif is_int(res) or is_float(res):  # not complex
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


def _cmin(*args: Number) -> Number:
    return cast(Number, -_cmax(*(-arg for arg in args)))


class Interval(Generic[Number]):
    """Closed interval (endpoints a and b are included)"""

    a: Number
    b: Number

    def __init__(self, a: AnyNumber, b: AnyNumber, /):
        if a.real > b.real:
            a, b = b, a
        if a.imag > b.imag:
            a, b = ComplexType(a.real, b.imag), ComplexType(b.real, a.imag)

        self.a, self.b = clean_number(a), clean_number(b)

    def __repr__(self):
        return f"{type(self).__name__}({self.a!r}, {self.b!r})"

    def __str__(self):
        return f"[{self.a}, {self.b}]"

    def __contains__(self, x: Union[Number, Interval[Number], object]):
        if isinstance(x, Interval):
            # subinterval check
            return x.a in self and x.b in self
        if is_number(x):
            y = clean_number(x)
            return (
                self.a.real <= y.real <= self.b.real
                and self.a.imag <= y.imag <= self.b.imag
            )

        return False

    def __eq__(self, other) -> bool:
        if isinstance(other, Interval):
            return bool(self.a == other.a and self.b == other.b)
        elif self.is_degenerate and is_number(other):
            return bool(self.a == clean_number(other))

        return False

    def __lt__(self, other) -> bool:
        if isinstance(other, Interval):
            # subinterval check
            other = other.a
        if is_number(other):
            x = clean_number(other)
            return bool(self.b.real < x.real or self.b.imag < x.imag)

        return NotImplemented

    def __le__(self, other) -> bool:
        if isinstance(other, Interval) or is_number(other):
            return bool(self == other or self < other)

        return NotImplemented

    def __gt__(self, other) -> bool:
        if isinstance(other, Interval):
            # subinterval check
            other = other.b
        if is_number(other):
            x = clean_number(other)
            return bool(self.a.real > x.real or self.a.real > x.imag)

        return NotImplemented

    def __ge__(self, other) -> bool:
        if isinstance(other, Interval) or is_number(other):
            return bool(self == other or self > other)

        return NotImplemented

    def __and__(self, other: Interval) -> Optional[Interval]:
        if isinstance(other, Interval):
            if self.isdisjoint(other):
                return None

            return type(self)(_cmax(self.a, other.a), _cmin(self.b, other.b))
        return NotImplemented

    def __or__(self, other: Interval) -> Interval:
        if isinstance(other, Interval):
            if self.isdisjoint(other):
                # TODO mask the gap, or create a disjoint interval class
                raise NotImplementedError

            return type(self)(_cmin(self.a, other.a), _cmax(self.b, other.b))
        return NotImplemented

    def __add__(self, other) -> Interval:
        if isinstance(other, Interval):
            return type(self)(self.a + other.a, self.b + other.b)
        if is_number(other):
            x = clean_number(other)
            return type(self)(self.a + x, self.b + x)
        return NotImplemented

    def __sub__(self, other) -> Interval:
        if isinstance(other, Interval):
            return type(self)(self.a - other.a, self.b - other.b)
        if is_number(other):
            x = clean_number(other)
            return type(self)(self.a - x, self.b - x)
        return NotImplemented

    def __mul__(self, other) -> Interval:
        if isinstance(other, Interval):
            return type(self)(self.a * other.a, self.b * other.b)
        if is_number(other):
            x = clean_number(other)
            return type(self)(self.a * x, self.b * x)
        return NotImplemented

    def __truediv__(self, other) -> Interval:
        if isinstance(other, Interval):
            return type(self)(self.a / other.a, self.b / other.b)
        if is_number(other):
            x = clean_number(other)
            return type(self)(self.a / x, self.b / x)
        return NotImplemented

    def __pow__(self, other) -> Interval:
        if isinstance(other, Interval):
            return type(self)(self.a ** other.a, self.b ** other.b)
        if is_number(other):
            x = clean_number(other)
            return type(self)(self.a ** x, self.b ** x)
        return NotImplemented

    def __radd__(self, other) -> Interval:
        return self.__add__(other)

    def __rsub__(self, other) -> Interval:
        return self.__sub__(other)

    def __rmul__(self, other) -> Interval:
        return self.__mul__(other)

    def __neg__(self) -> Interval:
        return type(self)(-self.a, -self.b)

    def __bool__(self) -> bool:
        return True

    def __hash__(self):
        return hash((self.a, self.b))

    @property
    def is_complex(self) -> bool:
        if self.a.imag or self.b.imag:
            return True

        return is_complex(self.a) or is_complex(self.b)

    @property
    def is_discrete(self) -> bool:
        if self.is_complex:
            return False

        return is_int(self.a) and is_int(self.b)

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
    def size(self) -> Number:
        return self.b - self.a

    @property
    def radius(self) -> Union[Number, FloatType]:  # lower type bound for int
        return self.size / 2

    @property
    def mid(self) -> Union[Number, FloatType]:  # lower type bound for bool
        return (self.a + self.b) / 2

    @property
    def d(self) -> Number:
        """
        the difference between two consecutive values for the given precision
        of the endpoint types
        """
        if self.is_complex:
            d = ComplexType(mpmath.mp.eps, mpmath.mp.eps)
        elif self.is_discrete:
            d = IntType(1)
        else:
            d = FloatType(mpmath.mp.eps)

        return cast(Number, d)

    def isdisjoint(self, other: Interval) -> bool:
        # for collections.AbstractSet compatiblity
        return bool(
            self.a.real > other.b.real
            or self.a.imag > other.b.imag
            or self.b.real < other.a.real
            or self.a.imag < other.a.imag
        )


class DiscreteInterval(Interval[IntType]):
    def __contains__(self, x: Union[IntType, Interval[IntType], object]):
        if not isinstance(x, DiscreteInterval) and not is_int(x):
            return False

        return super().__contains__(x)


_N_co = TypeVar("_N_co", bound=CleanNumber, covariant=True)


@runtime_checkable
class Function(Protocol[Number, _N_co]):
    __slots__ = ()

    @property
    @abc.abstractmethod
    def __discrete__(self) -> bool:
        ...

    @property
    @abc.abstractmethod
    def __support__(self) -> Interval[Number]:
        ...

    @abc.abstractmethod
    def f(self, x: Number) -> _N_co:
        ...


class _FunctionAlias(Function[Number, _N_co], Generic[Number, _N_co]):
    __slots__ = ("__wrapped__", "__support")  # noqa

    __support: Interval[Number]

    def __init__(
        self, fn: Callable[[Number], _N_co], support: Interval[Number]
    ):
        self.__wrapped__: Callable[[Number], _N_co] = fn
        self.__support = support

    @property
    def __discrete__(self):
        return False

    @property
    def __support__(self) -> Interval[Number]:
        return self.__support

    def f(self, x: Number) -> _N_co:
        return self.__wrapped__(x)


class _DiscreteFunctionAlias(_FunctionAlias[IntType, _N_co], Generic[_N_co]):
    __discrete__ = True


def supp(f: Function[Number, AnyNumber]) -> Interval[Number]:
    return f.__support__


def convolve(
    f: Function[Number, _N_co], g: Function[Number, _N_co]
) -> Function[Number, _N_co]:
    """
    Convolution (lazy):
    https://en.wikipedia.org/wiki/Convolution
    """
    res_support = f.__support__ + g.__support__

    fa, fb = f.__support__.a, f.__support__.b
    ga, gb = g.__support__.a, g.__support__.b

    integrator: Callable[[Callable[[Number], _N_co], List[Number]], _N_co]
    if f.__discrete__ and g.__discrete__:
        integrator = mpmath.nsum
    else:
        integrator = mpmath.quad

    def res(n: Number) -> _N_co:
        if not is_complex(n):
            # clip the support bounds
            a = fb if n - ga > fb else ga
            b = fa if n - gb < fa else gb
        else:
            a, b = ga, gb

        def dres(m: Number):
            return mpmath.fmul(f.f(n - m), g.f(m))

        return integrator(dres, [a, b])

    if f.__discrete__ and g.__discrete__:
        return _DiscreteFunctionAlias(res, res_support)
    else:
        return _FunctionAlias(res, res_support)
