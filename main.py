import warnings

import numpy as np

from modules import Control
from modules import Environment
from modules import Model

def get_env():
    environment_input = Environment(
        steps=100,
        max_value_of_trade=2000.0,
        number_of_points_in_sampling=25,
        n=1,
        d=2,
        kappa=np.array([[0, 0], [0, 1]]) * 0,  # Имитация уровня риска и взаимосвязи между валютами (+)
        alpha=np.array([[[0, 1], [1, 0]]]),  # Установка коэффициентов для логистической функции
        beta=np.array([[[0, 1], [1, 0]]]),  # Установка коэффициентов для логистической функции
        psi=np.array([[0, 1.], [1., 0]]) * 0.1,  # Имитация уровня риска для транзакций (+)
        eta=np.array([[0, 1.], [1., 0]]) * 1e-5,  # Имитация уровня риска для транзакций (+)
        lambdas=lambda z: np.array([[[0, 1000 / np.exp(z/250)], [1000 / np.exp(z/250), 0]]]),  # Установка интенсивности для каждой валюты
        mu=lambda t: np.zeros((2)),  # Установка дрейфа для каждой валюты
        sigma=np.array([[0, 0], [0, 1]]),  # Установка шума для каждой валюты
        market_impact=np.ones((2)) * 5 * 10**-3,  # Влияние на рынок для каждой валюты (+)
        T=1.0,  # Определение временного интервала
        gamma=0.1  # Определение коэффициента риска
    )
    return environment_input

def main():
    env = get_env()
    ctrl = Control(env)
    model = Model(
        q_0=np.ones((2)) * 100,
        X_0=100,
        S_0=np.ones((2)) * 100,
        environment=env,
        optimal_control=ctrl
    )
    model.modulation()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()