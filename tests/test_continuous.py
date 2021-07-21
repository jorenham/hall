import hypothesis.strategies as st
import mpmath
from hypothesis import assume, given
from pytest import approx

from hall import E, P, Std, Var
from hall.continuous import Normal


mpmath.mp.dps = 24


@given(
    st.floats(-1e10, 1e10, allow_infinity=False, allow_nan=False),
    st.floats(0.1, allow_infinity=False, allow_nan=False),
)
def test_normal(mu, sigma):
    X = ~Normal(mu, sigma)

    assert mu == E[X]
    assert sigma == Std[X]

    assert 1 / 2 == P[X <= mu]

    assert P[X <= mu - sigma] < P[X <= mu]
    assert P[X <= mu - sigma] == approx(P[X >= mu + sigma])
