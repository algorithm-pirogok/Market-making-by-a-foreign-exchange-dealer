import numpy as np



def test_h_beautiful(test_control):
    """
    H(p) = sup_xi p^{i, j}*xi - L^{i, j}(xi)
    """
    p = np.ones((1, 3))
    res = test_control.H_beautiful(p)
    
    assert res.shape == p.shape, "H_beautiful has a wrong shape" # (?)
    assert np.allclose(res, np.array([[750, 361 / 1600, 0]]), atol=0.002), f"H_beatiful is incorrect {res}"


def test_hamiltonian(test_control):
        """
        Compute
        H(p) = sup_delta f^{n, i, j}(delta) * (delta - p)
        """
        p = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
        res = test_control.Hamiltonian(p)
        
        assert res.shape == p.shape, f"Hamiltonian has a wrong shape: {res.shape}"
        assert np.allclose(res, 1/22 *(2*3**0.5-1)), "Hamiltonian computation has mistakes"


def test_derivative_of_h_beautiful(test_control):
    """
    dH^{i, j}(p)/dp
    """
    eps = 1e-3
    
    p = np.array([[-100, -0.005, 0]])
    res = test_control._derivative_of_h_beautiful(p)
    real_res = (test_control.H_beautiful(p+eps) - test_control.H_beautiful(p-eps)) / (2*eps)
    assert res.shape == p.shape, "Der shape is wrong"
    assert np.allclose(res, real_res, rtol=0.01, atol=0.2), f"Derivative of h_beautiful is incorrect 1 {res}, {real_res}"
    
    p = np.array([[100, 0.005, 0]])
    res = test_control._derivative_of_h_beautiful(p)
    real_res = (test_control.H_beautiful(p+eps) - test_control.H_beautiful(p-eps)) / (2*eps)
    assert np.allclose(res, real_res, rtol=0.01, atol=0.2), f"Derivative of h_beautiful is incorrect 2 {res}, {real_res}"


def test_base_params(test_control):
    """
    M_over  = \sum_{n=1}^N \int_{R_+} a_2^{n, i, j}(z)*z*lambda^{n, i, j}(z)dz
    M_under = \sum_{n=1}^N \int_{R_+} a_1^{n, i, j}(z)*z*lambda^{n, i, j}(z)dz
    P       = \sum_{n=1}^N \int_{R_+} a_2^{n, i, j}(z)*z^2*lambda^{n, i, j}(z)dz
    """
    M_over, M_under, P = test_control._compute_base_params(True)
    
    assert M_over.shape == M_under.shape == P.shape == (test_control.env.d, test_control.env.d)
    
    assert np.allclose(M_over, 2.1512, atol=0.02), "M_over is incorrect"
    assert np.allclose(M_under, 1.0756, atol=0.02), "M_under is incorrect"
    assert np.allclose(P, 10.075, atol=0.02), f"P is incorrect"


def test_V_params(test_control):
    """
    V_over(A) = D(A)P-PD(A)-2P*A
    V_hat(A) = (V_over(A) - V_over(A)^T) @ U
    """
    test_control._P = np.array([[1, 2, 3], [-4, 2, -3], [1, 3, 4]], dtype=float)
    A = np.array([[1, 3, 5], [-7, -3, -2], [2, 3, 3]], dtype=float)
    V_over = test_control._compute_overline_v(A)
    
    test_control._compute_overline_v = lambda x: x
    V_hat = test_control._compute_hat_v(A)
    
    assert V_over.shape == (test_control.env.d, test_control.env.d), "V_over has a bad shape"
    assert V_hat.shape == (test_control.env.d, ), "V_hat has a bad shape"
    
    assert np.allclose(V_over, np.array([[ 0, -16, -18],[-48, 0, -12], [ 0, -18, 0]])), "V_overline is incorrect"
    assert np.allclose(V_hat, np.array([13, -15, 2])), "V_hat os incorrect"


def test_main_params(test_control):
    """
    M = D((M_over + M_over^T) @ U) - (M_over + M_over^T)
    V = (M_under - M_under^T) @ U
    """
    test_control._M_overline = np.array([[1, 2, 3], [-4, 2, -3], [1, 3, 4]], dtype=float)
    test_control._M_underline = np.array([[1, 3, 5], [-7, -3, -2], [2, 3, 3]], dtype=float)
    
    M, V = test_control._compute_main_params()
    
    assert M.shape == (test_control.env.d, test_control.env.d), "M has a bad shape"
    assert V.shape == (test_control.env.d, ), "V has a bad shape"
    
    assert np.allclose(M, np.array([[2, 2, -4], [2, -2, 0], [-4, 0, 4]])), "M is incorrect"
    assert np.allclose(V, np.array([13, -15, 2])), "V is incorrect"
    

def test_time_params(test_control):
    """
    A'(t) = 2A(t)MA(t) - Sigma * A(t) - 2D(mu(t))A(t) - gamma/2Sigma
    B'(t) = mu(t) - D(mu(t))B(t) + 2A(t)V + 2A(t)V_hat(A(t)) + 2A(t)MB(t)
    """
    test_control._M = np.array([[2, 3, -4], [2, -2, -1], [-4, 3, 4]])
    test_control._V = np.array([4, -2, -1])
    test_control._compute_hat_v = lambda x: x[0]
    A, B = test_control._compute_time_params(debug=True)
    
    assert isinstance(A, np.ndarray), "A[t] is not list"
    assert isinstance(B, np.ndarray), "B[t] is not list"
    
    assert A[0].shape == (test_control.env.d, test_control.env.d), "A[i] has wrong dim"
    assert B[0].shape == (test_control.env.d, ), "B[i] has wrong dim"
    
    assert np.allclose(A[0], np.array([[73, -23, 66], [-220, 60, -230], [86, -5, 100]])), "A[t] is incorrect"
    assert np.allclose(B[0], np.array([-95, 221, -105])), "B[t] is incorrect"


def test_clients_policy(test_control):
    """
    delta_opt^{n, i, j}(t, z) = delta_hat^{n, i, j}((2Y_t+z(e^i-e^j))^TA(t)+B(t)^T)(e^i-e^j))
    """
    test_control._A = lambda x: np.array([[6, -2, 1], [2, 4, 9], [-3, -5, 7]])
    test_control._B = lambda x: np.array([4, 3, 6])
    
    A = np.array([[6, -2, 1], [2, 4, 9], [-3, -5, 7]])
    B = np.array([4, 3, 6])
    
    test_control._optimal_delta = lambda x: np.sqrt(np.abs(x))
    
    z = 5
    t = 0
    Y_t = np.array([7, 2, 1])
    
    ans = np.zeros((test_control.env.n, test_control.env.d, test_control.env.d))
    
    for n in range(test_control.env.n):
        for i in range(test_control.env.d):
            for j in range(test_control.env.d):
                e_i, e_j = np.zeros(Y_t.shape), np.zeros(Y_t.shape)
                e_i[i], e_j[j] = 1, 1
                ans[n][i][j] = test_control._optimal_delta(np.dot(np.dot(2*Y_t+z*(e_i-e_j), A) + B, e_i-e_j))
    res = test_control.clients_policy(Y_t, z, t)
    assert res.shape != ans.shape, "Client policy has a bad shape"
    assert np.allclose(res, ans), "Client policy is incorrect"


def test_dillers_policy(test_control):
    """
    xi_opt^{i, j}(t) = H_beaut^{i, j,'}(-(A(t)Y_t+B(t))^T(e^i-e^j) + k^iY_t^i(1-(A(t)Y_t+B(t))^Te^i)
                                                                    - k^jY_t^j(1-(A(t)Y_t+B(t))^Te^j)
    """
    test_control._A = lambda x: np.array([[6, -2, 1], [2, 4, 9], [-3, -5, 7]])
    test_control._B = lambda x: np.array([4, 3, 6])
    
    A = np.array([[6, -2, 1], [2, 4, 9], [-3, -5, 7]])
    B = np.array([4, 3, 6])
    
    test_control._derivative_of_h_beautiful = lambda x: x**3 - 1
    
    Y_t = np.array([7, 2, 1])
    t = 1
    
    ans = np.zeros((test_control.env.d, test_control.env.d))

    kernel = np.dot(A, Y_t) + B
    k = test_control.env.market_impact
    for i in range(test_control.env.d):
        for j in range(test_control.env.d):
            e_i, e_j = np.zeros(Y_t.shape), np.zeros(Y_t.shape)
            e_i[i], e_j[j] = 1, 1
            if i < j:
                ans[i][j] = test_control._derivative_of_h_beautiful(-np.dot(kernel, e_i-e_j) + k[i]*Y_t[i]*np.dot(1-kernel, e_i) - k[j]*Y_t[j]*np.dot(1-kernel, e_j))
    
    res = test_control.dillers_policy(t, Y_t)
    assert res.shape == ans.shape, "Dillers policy has a wrong shape"
    assert np.allclose(res, ans), "Dillers policy is incorrect"
    