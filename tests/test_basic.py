import numpy as np
import pytest

from framework_test import forward_euler_solver


@pytest.fixture
def simpleODE():
    def ode(t, y):
        return -y

    return ode


@pytest.fixture
def otherODE():
    def ode(t, y):
        return -y

    return ode


@pytest.fixture(params=["simpleODE", "otherODE"])
def ODE(request):
    return request.getfixturevalue(request.param)


def test_solve(ODE):
    _sol = forward_euler_solver(ODE, np.array([0.0, 1.0]), np.array([1.0]))
