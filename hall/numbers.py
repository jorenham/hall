from __future__ import annotations


__all__ = [
    "INT_TYPES",
    "FLOAT_TYPES",
    "COMPLEX_TYPES",
    "NUMBER_TYPES",
    "NUMBER_TYPES_RAW",
    "IntType",
    "FloatType",
    "Probability",
    "ComplexType",
    "AnyInt",
    "AnyFloat",
    "AnyComplex",
    "AnyNumber",
    "CleanNumber",
    "Float",
    "Complex",
    "Number",
    "is_int",
    "is_float",
    "is_probability",
    "is_complex",
    "is_number",
    "is_mp_number",
    "clean_number",
]

import sys
from decimal import Decimal
from fractions import Fraction
from typing import Any, TypeVar, Union, overload

import mpmath
from typing_extensions import TypeGuard


if sys.version_info >= (3, 10):
    from types import EllipsisType  # noqa
else:
    EllipsisType = None

from mpmath import ctx_mp_python, libmp


"""
for all `numbers.*`, `T = TypeVar("T", bound=numbers.Real)` is bugged in mypy:
https://github.com/python/mypy/issues/3186
"""

# integers
# IntType: Any = libmp.MPZ_TYPE  # either `gmpy.mpz`, `sage.Integer` or `int`
IntType = int
INT_TYPES = (int, IntType)
AnyInt = Union[int, IntType]
Int = TypeVar("Int", bound=AnyInt)


def is_int(x: object) -> TypeGuard[AnyInt]:
    return isinstance(x, INT_TYPES)


# floats
FloatType: Any = mpmath.mpf
ConstantType: Any = mpmath.mp.constant
# float alias that implies the value to be within [0, 1]
Probability = FloatType
FLOAT_TYPES = (float, FloatType, ConstantType)
AnyFloat = Union[float, FloatType, ConstantType]
Float = TypeVar("Float", bound=AnyFloat)


def is_float(x: object) -> TypeGuard[AnyFloat]:
    return isinstance(x, FLOAT_TYPES)


def is_probability(x: object) -> TypeGuard[AnyFloat]:
    return is_float(x) and bool(0.0 <= FloatType(x) <= 1.0)


# complex
ComplexType: Any = ctx_mp_python._mpc  # noqa
COMPLEX_TYPES = (complex, ComplexType)
AnyComplex = Union[complex, ComplexType]
Complex = TypeVar("Complex", bound=AnyComplex)


def is_complex(x: object) -> TypeGuard[AnyComplex]:
    return isinstance(x, COMPLEX_TYPES)


# all supported numeric types
NUMBER_TYPES_RAW = INT_TYPES + FLOAT_TYPES + COMPLEX_TYPES + (Decimal, Fraction)
AnyNumber = Union[AnyInt, AnyFloat, AnyComplex, Decimal, Fraction]
RawNumber = TypeVar("RawNumber", bound=AnyNumber)

NUMBER_TYPES = IntType, FloatType, ComplexType
CleanNumber = Union[IntType, FloatType, ComplexType]
Number = TypeVar("Number", IntType, FloatType, ComplexType)

NUMBER_TYPES_PY = (int, float, complex, Decimal, Fraction)


def is_number(x: object):
    return isinstance(x, NUMBER_TYPES_RAW)


def is_mp_number(x: object) -> bool:
    return is_number(x) and not isinstance(x, NUMBER_TYPES_PY)


@overload
def clean_number(x: AnyInt) -> IntType:
    ...


@overload
def clean_number(x: Union[AnyFloat, Decimal, Fraction]) -> FloatType:
    ...


@overload
def clean_number(x: AnyComplex) -> ComplexType:
    ...


def clean_number(x: AnyNumber) -> CleanNumber:
    if not is_number(x):
        raise TypeError("not a number")

    if is_int(x):
        return IntType(x)
    if is_float(x):
        return FloatType(x)
    if isinstance(x, (Decimal, Fraction)):
        return FloatType(x)
    if is_complex(x):
        return ComplexType(x)

    raise TypeError(f"unknown number type {type(x).__name__!r}")
