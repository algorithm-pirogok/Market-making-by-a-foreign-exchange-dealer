import numpy as np
from scipy.optimize import minimize

from modules import Environment

class Control:
    def __init__(self, environment: Environment, epsilon: float = 1e-4, debug: bool = False):
        self.env = environment
        self.eps = epsilon
        self.delta_t = self.env.T / self.env.steps
        
        if not debug:
            self._M_overline, self._M_underline, self._P = self._compute_base_params()
            self._M, self._V = self._compute_main_params()
            self._A_memory, self._B_memory = self._compute_time_params()

    def H_beautiful(self, p: np.ndarray):
        """
        Compute
        H(p) = sup_xi p^{i, j}*xi - L^{i, j}(xi)
        """
        def functional(xi):
            return p * xi.reshape(p.shape) - self.env.transaction_loss(xi.reshape(p.shape))
        
        def f(xi):
            return -np.sum(functional(xi))

        initial_guess = np.zeros(p.shape).reshape(-1)
        result = minimize(f, initial_guess, options={'disp': False}, tol=1e-6)
        return functional(result['x'])

    def Hamiltonian(self, p: np.ndarray):
        """
        Compute
        H(p) = sup_delta f^{n, i, j}(delta) * (delta - p)
        """
        def functional(delta):
            return self.env.compute_logistic_function(delta.reshape(p.shape)) * (delta.reshape(p.shape) - p)

        def f(delta):
            return -np.sum(functional(delta))

        initial_guess = np.zeros(p.shape).reshape(-1, )
        result = minimize(f, initial_guess, options={'disp': False})
        return functional(result['x'])

    def _derivative_of_hamiltonian(self, p: np.ndarray):
        """
        Compute
        dH^{n, i, j}/dp(z, p)
        """
        return (self.Hamiltonian(p+self.eps) - self.Hamiltonian(p-self.eps)) / (2*self.eps)

    def _derivative_of_h_beautiful(self, p: np.ndarray):
        """
        Compute
        dH^{i, j}(p)/dp
        """
        ans = 1 / (2 * self.env.eta) * np.sign(p) * np.maximum(0, np.abs(p) - self.env.psi)
        ans[np.isinf(ans)] = 0
        return np.nan_to_num(ans)

    def _optimal_delta(self, p: np.ndarray):
        """
        Compute
        delta_optimal(p) = (f^{n, i, j})^{-1}(-\partial_p H^{n, i, j}(p))
        """
        der = -self._derivative_of_hamiltonian(p)
        #print("DER", der)
        ans = (np.log(1 / der - 1) - self.env.alpha) / (self.env.beta)
        ans[np.isinf(ans)] = 0
        return np.nan_to_num(ans)

    def _get_hamiltonians_alphas(self):
        """
        Compute
        a_1(z) = dH/dp(z, 0)
        a_2(z) = d^2H/d^2p(z, 0)
        """
        zeros = np.zeros((self.env.n, self.env.d, self.env.d))
        a_1 = self._derivative_of_hamiltonian(zeros)
        a_2 = (self._derivative_of_hamiltonian(zeros+self.eps) - self._derivative_of_hamiltonian(zeros-self.eps)) / (2*self.eps)
        return a_1, a_2

    def _compute_base_params(self, debug: bool =False):
        """
        Compute
        M_over  = \sum_{n=1}^N \int_{R_+} a_2^{n, i, j}(z)*z*lambda^{n, i, j}(z)dz
        M_under = \sum_{n=1}^N \int_{R_+} a_1^{n, i, j}(z)*z*lambda^{n, i, j}(z)dz
        P       = \sum_{n=1}^N \int_{R_+} a_2^{n, i, j}(z)*z^2*lambda^{n, i, j}(z)dz
        """
        M_over, M_under, P = np.zeros((self.env.d, self.env.d)).astype(float), np.zeros((self.env.d, self.env.d)).astype(float), np.zeros((self.env.d, self.env.d)).astype(float)  
        z_coeff = self.env.max_value_of_trade / self.env.number_of_points_in_sampling
        if not debug:
            a_1, a_2 = self._get_hamiltonians_alphas()
        for z in np.linspace(0, self.env.max_value_of_trade, self.env.number_of_points_in_sampling):
            if debug:
                a_1, a_2 = self.env.get_hamiltonians_alphas(z)
            coeff_1, coeff_2 = np.sum(a_1*self.env.lambdas(z), axis=0), np.sum(a_2*self.env.lambdas(z), axis=0)
            M_over += coeff_2 * z * z_coeff
            M_under += coeff_1 * z * z_coeff
            P += coeff_2 * z**2 * z_coeff
        return M_over, M_under, P

    def _compute_overline_v(self, A: np.ndarray):
        """
        Compute
        V_over(A) = D(A)P+PD(A)-2P*A
        """
        A_diag = np.diag(np.diag(A))
        V_overline = A_diag @ self._P + self._P @ A_diag - 2 * self._P * A
        return V_overline

    def _compute_hat_v(self, A: np.ndarray):
        """
        Compute
        V_hat(A) = (V_over(A) - V_over(A)^T) @ U
        """
        V_over = self._compute_overline_v(A)
        V_hat = np.dot(V_over - V_over.transpose(1, 0), np.ones(V_over.shape[0]))
        return V_hat

    def _compute_main_params(self):
        """
        Compute
        M = D((M_over + M_over^T) @ U) - (M_over + M_over^T)
        V = (M_under - M_under^T) @ U
        """
        M_plus = self._M_overline + self._M_overline.transpose(1, 0)
        M_minus = self._M_underline - self._M_underline.transpose(1, 0)
        
        M = np.diag(np.dot(M_plus, np.ones(M_plus.shape[0]))) - M_plus
        V = np.dot(M_minus, np.ones(M.shape[0]))
        return M, V

    def _compute_time_params(self, debug: bool = False):
        """
        Solve the system
        A'(t) = 2A(t)MA(t) - Sigma * A(t) - 2D(mu(t))A(t) - gamma/2Sigma
        B'(t) = mu(t) - D(mu(t))B(t) + 2A(t)V + 2A(t)V_hat(A(t)) + 2A(t)MB(t)
        A(T) = kappa, B(T) = 0
        """
        A = np.zeros((self.env.steps+1, self.env.d, self.env.d))
        B = np.zeros((self.env.steps+1, self.env.d))
        A[-1] = self.env.kappa
        if debug:
            B[-1] = np.array([2, 4, 3])
        env = self.env
        for t in range(self.env.steps, 0, -1):
            dA = 2 * A[t] @ self._M @ A[t] - env.sigma * A[t] - 2*np.diag(env.mu(t).squeeze()) @ A[t] - env.gamma / 2 * env.sigma
            A[t-1] = A[t] - dA * self.delta_t
            dB = env.mu(t) - env.mu(t) * B[t] + 2 * np.dot(A[t], self._V) + 2 * np.dot(A[t], self._compute_hat_v(A[t])) + 2 * np.dot(np.dot(A[t], self._M), B[t])
            B[t-1] = B[t] - dB * self.delta_t
        return A, B

    def _A(self, t: float):
        """
        Get
        A(t)
        """
        index = int(t * self.env.T)
        return self._A_memory[index]

    def _B(self, t: float):
        """
        Get
        B(t)
        """
        index = int(t * self.env.T)
        return self._B_memory[index]

    def clients_policy(self, Y_t, z, t):
        """
        delta_opt^{n, i, j}(t, z) = delta_hat^{n, i, j}((2Y_t+z(e^i-e^j))^TA(t)+B(t)^T)(e^i-e^j))
        """
        exp_delta = np.fromfunction(lambda i, j, k: np.float64(k == i) - np.float64(k == j), (self.env.d, self.env.d, self.env.d))
        param = np.sum((np.tensordot(2 * Y_t + z * exp_delta, self._A(t), (2, 0)) + self._B(t)) * exp_delta, axis=2)
        return self._optimal_delta(param)

    def dillers_policy(self, t, Y_t):
        """
        xi_opt^{i, j}(t) = H_beaut^{i, j,'}(-(A(t)Y_t+B(t))^T(e^i-e^j) + k^iY_t^i(1-(A(t)Y_t+B(t))^Te^i)
                                                                       - k^jY_t^j(1-(A(t)Y_t+B(t))^Te^j)
        """
        A = self._A(t)
        B = self._B(t)

        kernel = np.dot(A, Y_t)+ B
        
        first_part = np.tile(kernel, (len(kernel), 1)).T - np.tile(kernel, (len(kernel), 1))
                         
        second_part = self.env.market_impact * Y_t * (1-kernel)

        param = -first_part + np.tile(second_part, (len(second_part), 1)).T - np.tile(second_part, (len(second_part), 1))        
        diller_policy = self._derivative_of_h_beautiful(param)

        sz = np.arange(diller_policy.shape[0])
        sz = (sz[:, np.newaxis] < sz).astype(int)

        return diller_policy * sz