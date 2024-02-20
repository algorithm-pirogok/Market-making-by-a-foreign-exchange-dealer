import numpy as np

def test_dq(model, client_trades, diller_trades):
    r"""
    dq_t^i=\sum\limits_{n=1}^N\sum\limits_{j\neq i}\int_{z\in\mathbf{R}_+^*}\frac{z}{S_t^i}\Big(J^{n, i, j}(dt, dz)-J^{n, j, i}(dt, dz)\Big)dz+
    \Big(\sum\limits_{j=i+1}^{d}\frac{\xi_t^{i, j}}{S_t^i}-\sum\limits_{j=1}^{i-1}\frac{\xi_t^{j, i}}{S_t^i}\Big)dt
    """
    dq_t = np.zeros(model.q_t.shape)
    env = model.env
    dt = env.T / env.steps
    
    for i in range(env.d):
        client_part = 0
        for k in range(env.n):
            for j in range(env.d):
                client_part += (client_trades[k][i][j] - client_trades[k][j][i]) / model.S_t[i]
        diller_part = 0
        for j in range(i+1, env.d):
            diller_part += diller_trades[i][j] / model.S_t[i]
        for j in range(0, i):
            diller_part -= diller_trades[j][i] / model.S_t[i]
        dq_t[i] =(client_part + diller_part) * dt
    res = model._dq(client_trades, diller_trades)
    assert np.allclose(res, dq_t), "dq_t is incorrect"
    

def test_dX_t(model, client_trades, diller_trades):
    r"""
    dX_t = ...
    """
    env = model.env
    dt = env.T / env.steps
    loss = env.transaction_loss(diller_trades)
    dX_t = 0
    
    for k in range(env.n):
        for i in range(env.d):
            for j in range(env.d):
                if i != j:
                    dX_t += client_trades[k][i][j]
    for i in range(env.d):
        for j in range(i+1, env.d):
            dX_t -= loss[i][j]
    dX_t *= dt
    
    res = model._dX(client_trades, diller_trades)
    
    assert np.allclose(res, dX_t), "dX_t is incorrect"
    

def test_dS_t(model, diller_trades):
    r"""
        dS_t^i = \mu_t^i*S_t^idt + sigma^iS_t^idW_t^i+k^i(\sum_{j=i+1}^d\xi_t^{i, j} - \sum_{j=1}^{i-1} \xi_t^{j, i})S_t^idt
    """
    t = 0
    dS_t = np.zeros(model.S_t.shape)
    noise = np.array([0.5, -0.12, 2.17])
    dt = model.env.T / model.env.steps
    env = model.env
    
    for i in range(env.d):
        sm = 0
        for j in range(i+1, env.d):
            sm += diller_trades[i][j]
        for j in range(0, i):
            sm -= diller_trades[j][i]
        dS_t[i] = env.mu(t)[i] * model.S_t[i] * dt + noise[i] * model.S_t[i] * dt ** 0.5 + env.market_impact[i] * sm * model.S_t[i] * dt

    res = model._dS(t, noise, diller_trades)
    
    assert np.allclose(dS_t, res), "dS_t is incorrect"
        