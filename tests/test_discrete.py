import mpmath

from hall import E, P, Var
from hall.discrete import Bernoulli, Binomial, Uniform


def test_bernoulli():
    X = ~Bernoulli(0.4)
    assert E[X] == 0.4
    assert Var[X] == 0.24
    assert P[X == 1] == 1 - P[X == 0] == P[X != 0] == 0.4


def test_binomial():
    X = ~Binomial(4, mpmath.fraction(1, 2))

    assert E[X] == 2
    assert Var[X] == 1

    assert 0 in X.outcomes, str(X.distribution.__support__)
    assert 4 in X

    assert P[X == 0] == P[X == 4] == 1 / 16
    assert P[X == 1] == P[X == 3] == 4 / 16
    assert P[X == 2] == 6 / 16
    assert P[X <= 2] == 1.0 - P[X > 2] < 1.0
    assert P[X <= 2] == P[X < 3] < 1.0
    assert sum(P[X == x] for x in range(5)) == 1
    assert P[X < 1000] == 1


def test_uniform():
    X = ~Uniform(-5, 5)
    assert E[X] == 0
    assert Var[X] == 10
    assert mpmath.almosteq(sum(P[X == x] for x in range(-6, 7)), 1)
