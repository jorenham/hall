__all__ = ["MPBackend", "get_backend"]

from mpmath import libmp
from typing_extensions import Literal


MPBackend = Literal["python", "gmpy", "sage"]


def get_backend() -> MPBackend:
    backend: MPBackend = libmp.BACKEND
    return backend
