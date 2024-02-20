from dataclasses import dataclass
import typing as tp

import numpy as np

@dataclass
class Environment:
    # Technical params
    steps: int # How many steps we will modulate, delta_T = T / steps
    max_value_of_trade: float # maximum stocks can be trade in the moment
    number_of_points_in_sampling: int

    # Market params
    n: int # Number of clusters
    d: int # Number of stocks
    kappa: np.ndarray # kappa from final loss
    alpha: tp.Callable[[float], np.ndarray] # a^{n, i, j} from density function
    beta: tp.Callable[[float], np.ndarray] # b^{n, i, j} from density function
    psi: np.ndarray # L = \psi^{i, j} * |\xi| + \eta^{i, j} * |\xi|^2
    eta: np.ndarray # L = \psi^{i, j} * |\xi| + \eta^{i, j} * |\xi|^2
    lambdas: tp.Callable[[float], np.ndarray] # lambda^{n, i, j}
    mu: tp.Callable[[float], np.ndarray] # mu^i(t), drift of S_t^i
    sigma: np.ndarray # sigma^i noise of S_t^i, covariations of brownian motions
    market_impact: np.ndarray # impact of our trading operations

    # Model params
    T: float # Time of our interval
    gamma: float # risk-penalty coefficient

    def compute_logistic_function(self, delta: np.ndarray) -> np.ndarray:
        """
        f(z, delta^{n, i, j}) = 1 / (1+exp(a^{n, i, j} + b^{n, i, j}*delta^{n, i, j}))
        """
        return 1 / (1 + np.exp(self.alpha + self.beta * delta)) * (np.ones(delta.shape) - np.eye(delta.shape[-1]))

    def  _intensity_function(self, z: float, delta: np.ndarray) -> np.ndarray:
        """
        mu_t^{n, i, j}(dz, delta) = lambda^{n, i, j}(z) * f^{n, i, j}(z, delta)
        """
        return self.lambdas(z) * self.compute_logistic_function(delta)

    def sampling(self, t: float, delta: tp.Callable[[float, float], np.array]) -> tuple[np.ndarray, np.ndarray]:
        """
        \int_0^{+inf} z * mu_t^{n, i, j}(dz)
        """
        trades = np.random.poisson(self._intensity_function(0, delta(t, 0)) * self.T / self.steps).astype(float)
        comissions = np.random.poisson(self._intensity_function(0, delta(t, 0)) * self.T / self.steps).astype(float)
        pred_z = 0
        for z in np.logspace(0, np.log10(self.max_value_of_trade), self.number_of_points_in_sampling):
            curr_delta = delta(z, t)
            result = np.random.poisson(self._intensity_function((z+pred_z)/2, curr_delta) * self.T / self.steps * (z-pred_z))
            trades += z * result
            comissions += curr_delta * z * result
            z = pred_z
        return trades, comissions

    def transaction_loss(self, xi: np.ndarray) -> np.ndarray:
        """
        L(xi) = psi  * |xi| + eta * |xi|^2
        """
        return self.psi * np.abs(xi) + self.eta * xi ** 2

    def risk_loss(self, Y_t: np.ndarray) -> float:
        """
        r(Y_t) = gamma / 2 * Y_t^T @ kappa @ Y_t
        """
        return self.gamma / 2 * np.dot(np.dot(Y_t, self.kappa), Y_t)

    def final_loss(self, Y_t: np.ndarray) -> float:
        """
        l(Y_t) = Y_t^T @ Sigma @ Y_t
        """
        return np.dot(np.dot(Y_t, self.sigma), Y_t)