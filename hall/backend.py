__all__ = ["MPBackend", "get_backend", "mp_configure"]

from typing import Final, Tuple

import mpmath
from mpmath import ctx_base, libmp
from typing_extensions import Literal


MPBackend = Literal["python", "gmpy", "sage"]

MP_CONTEXTS: Final[Tuple[mpmath.ctx_base.StandardBaseContext, ...]] = (
    mpmath.fp,
    mpmath.mp,
    mpmath.iv,
)


def get_backend() -> MPBackend:
    backend: MPBackend = libmp.BACKEND
    return backend


def mp_configure(
    *,
    prec: int = 53,
    dps: int = 15,
    pretty: bool = True,
) -> None:

    """Configure all mpmath contexts.

    :param prec: binary precision in bits
    :param dps: decimal precision
    :param pretty: pretty formatting for `repr()`
    """
    for context in MP_CONTEXTS:
        context.prec = prec
        context.dps = dps
        context.pretty = pretty
