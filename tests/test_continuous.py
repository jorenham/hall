from hall import E, P, Var
from hall.continuous import N


def test_normal():
    X = ~N()
    assert E[X] == 0
    assert Var[X] == 1
    assert P[X <= 0.0] == 1 / 2
