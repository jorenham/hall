import hypothesis.strategies as st
import mpmath
from hypothesis import assume, given
from pytest import approx

from hall import E, P, Std, Var
from hall.discrete import Bernoulli, Binomial, Uniform


mpmath.mp.dps = 24


@given(st.floats(0, 1))
def test_bernoulli(p):
    assume(0 < p < 1)

    X = ~Bernoulli(p)

    assert p == approx(E[X])
    assert p >= Var[X]
    assert p == approx(P[X == 1])
    assert p == approx(1 - P[X == 0])
    assert p == approx(P[X != 0])


@given(st.integers(4, 1000), st.floats(0, 1))
def test_binomial(n, p):
    assume(0 < p < 1)

    X = ~Binomial(n, p)

    assert n * p == approx(E[X])
    assert n * p >= Var[X]

    assert 0 in X
    assert n in X
    assert n + 1 not in X

    assert P[X == 2] <= P[X <= 2]
    assert P[X < 1] <= P[X <= 2]

    assert 0 == P[X < 0]
    assert 1 == P[X <= n + 100000]


@given(st.integers(), st.integers())
def test_uniform(a, b):
    assume(a < b)

    X = ~Uniform(a, b)

    assert (a + b) / 2 == approx(E[X])
    assert b - a >= Std[X]
