import numpy as np
import pytest

from modules import Control, Model

class test_env:
    def __init__(self) -> None:
        self.T = 1
        self.steps = 1
        self.d = 3
        self.n = 1
        self.max_value_of_trade = 10
        self.number_of_points_in_sampling = 10000
        self.kappa = np.array([[1, 2, 3], [-4, 2, -3], [1, 3, 4]], dtype=float)
        self.sigma = np.array([[4, 6, 6], [-8, -4, -2], [2, 4, 4]], dtype=float)
        self.psi = np.array([-2, 0.05, 3])
        self.eta = np.array([0.003, 1, 3])
        self.gamma = 1.
        self.market_impact = np.array([0.1, 0.5, 0.03])

    def lambdas(self, x):
        return 1 / (1+x)**3
    
    def compute_logistic_function(self, p):
        return 1 / (10+np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]) + p**2)
    
    def get_hamiltonians_alphas(self, x):
        return x, 2 * x
    
    def mu(self, x):
        return np.array([1, 3, 6])
    
    def transaction_loss(self, x):
        return self.psi * np.abs(x) + self.eta * x**2

@pytest.fixture(scope='function')
def control_env():
    return test_env()

@pytest.fixture(scope='function')
def test_control(control_env):
    return Control(control_env, debug=True)

@pytest.fixture(scope='function')
def model(control_env, test_control):
    return Model(
        q_0=np.ones((3)) * 100,
        X_0=100,
        S_0=np.array([100, 20, 350]),
        environment=control_env,
        optimal_control=test_control
    )

@pytest.fixture(scope='function')
def client_trades():
    return np.array([[[0, 2, 6], [1, 0, 3], [7, 4, 0]]])
    
@pytest.fixture(scope='function')
def diller_trades():
    return np.array([[0, 4, -2], [0, 0, 3], [0, 0, 0]])
