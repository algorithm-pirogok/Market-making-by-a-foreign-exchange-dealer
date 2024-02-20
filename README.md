# Dealing with Multi-Currency Inventory Risk in FX Cash Markets

## Market-Making by a Foreign Exchange Dealer

## Theoretical model

### Model

We aim to develop a model for optimal asset management by a dealer. The dealer can adjust commissions for each client group and trade currencies with other dealers. Let's define it formally:

#### Clients

There are n clusters of clients, each with its own exchange rate commissions. Text formatted in bold

Clients in each cluster buy currency j for currency i of size z of the base currency at time t:

$$J^{n, i, j}(dt, dz)\sim\big(\nu_t^{n, i, j}(t, z)\big)dt$$

For each transaction, we receive $z\delta^{n, i, j}(t, z)$ in the base currency.

$\nu_t^{n, i, j}(t, z)= Λ^{n, i, j}\big(z, \delta^{n, i, j}(t, z)\big)dz$ - the process density at time t.

$$Λ^{n, i, j}(z, δ) = λ^{n, i, j}(z)f^{n, i, j}(z, δ),\ f^{n, i, j}(z, δ)=\frac{1}{1+\exp\big(α^{n, i, j}(z)+\beta^{n, i, j}(z)\delta\big)}$$

By managing commissions for each cluster, we can adjust the intensity of client purchases.

#### D2D

The process of buying currency i from other dealers for currency j, calculated in the base currency, with i < j:

$(\xi_t^{i, j})_{t\in[0;T]}$

The transactional costs for currency exchange, calculated in rubles:

$$L^{i, j}(\xi)=ψ^{i, j}|\xi|+\mu^{i, j}|\xi|^{1+ϕ^{i, j}}$$
Here, $\phi^{i, j}=1$

### Dynamics

The amount of currency i we hold in the portfolio:

$$dq_t^i=\sum\limits_{n=1}^N\sum\limits_{j\neq i}\int_{z\in\mathbf{R}_+^*}\frac{z}{S_t^i}\Big(J^{n, i, j}(dt, dz)-J^{n, j, i}(dt, dz)\Big)dz+\Big(\sum\limits_{j=i+1}^{d}\frac{\xi_t^{i, j}}{S_t^i}-\sum\limits_{j=1}^{i-1}\frac{\xi_t^{j, i}}{S_t^i}\Big)dt$$

The amount of money in the bank account:

$$dX_t=\sum\limits_{n=1}^N\sum\limits_{i\neq j}\int_{z\in\mathbf{R}_+^*}zδ^{n, i, j}(t, z)J^{n, i, j}(dt, dz)-\sum\limits_{i<j}L^{i, j}(\xi_t^{i, j})$$

Currency exchange rate movement:

$$dS_t^i = \mu_t^iS_t^idt+\sigma^iS_t^idW_t^i+k^i\Big(\sum\limits_{j=i+1}^{d}\xi_t^{i, j}-\sum\limits_{j=1}^{i-1}\xi_t^{j, i}\Big)S_t^idt$$

Let's define the vectorized form as:

$$S_t=(S_1,\ldots, S_d)^T\in\mathbf{R}^d,\ \mu(t)=\mu_t(\mu_t^1,\ldots, \mu_t^d)^T\in\mathbf{R}^d,\ \Sigma=(\rho^{i, j}\sigma^i\sigma^j)_{1\leq i, j\leq d}\in\mathcal{S}_d^+(\mathbf{R})$$

We also define the influence of the cost of foreign currency as $(Y_t^i)_{t\in[0; T]}=(q_t^iS_t^i)_{t\in[0;T]}$:

$$dY_t^i=\mu_t^iY_{t-}^idt+\sigma^iY_{t-}^idW_t^i+k^i\Big(\sum\limits_{j=i+1}^{d}\xi_t^{i, j}-\sum_{j=1}^{i-1}\xi_t^{j, i}\Big)Y_{t-}^idt+\sum\limits_{n=1}^N\sum\limits_{j\neq i}^d\int_{z\in\mathbf{R}_+^*}z\Big(J^{n, i, j}(dt, dz)-J^{n, j, i}(dt, dz)\Big)dz+\Big(\sum\limits_{j=i+1}^{d}\xi_{t}^{i, j}-\sum\limits_{j=1}^{i-1}\xi_t^{j, i}\Big)dt$$

### Task

We aim to maximize our earnings while minimizing risks, formalized as:

$$EBig[X_T+sumlimits_{i=1}^{d}Y_T^i-frac{gamma}{2}int_0^TY_t^TSigma Y_tdt-mathcal{l}(Y_T)Big]tomaxlimits_{delta, xi}$$

### Solution

Define three functions:

$$begin{cases}H^{n, i, j}: (z, p)inmathbf{R}_+^*timesmathbf{R}tosuplimits_{delta}f^{n, i, j}(z, delta)(delta-p)\
mathcal{H}^{i, j}: pinmathbf{R}tosuplimits_{xi}pxi-L^{i, j}(xi)\
overline{delta}^{n, i, j}(z, p) = (f^{n, i, j})^{-1}big(-partial_p H^{n, i, j}(z, p)big)
end{cases}$$

Then, express the optimal values $delta^*, xi^*$ as:

$$begin{cases}
delta^{n, i, j, *}(t, z) = overline{delta}^{n, i, j}Big(z, frac{θ(t, Y_{t-})-thetabig(t, Y_{t-}+ze^i-ze^jbig)}{z}Big)\
xi^{i, j, *}_t=mathcal{H}^{i, j, '}Big(partial_{y^i}theta(t, Y_{t-})-partial_{y^j}theta(t, Y_{t-})+k^iY^i_{t-}big(1+∂_{y^i}theta(t, Y_{t-})big)-k^jY_{t-}^jbig(1+partial_{y^j}theta(t, Y_{t-})big)Big)
end{cases}$$

Introduce the approximation of the Hessian $hat{H}^{n, i, j}(z, p)$ with coefficients $alpha_0^{n, i, j}(z)$, $alpha_1^{n, i, j}(z)$, $alpha_2^{n, i, j}(z)$.

Next, parameterize with matrices and vectors:

 $$M=mathcal{D}Big(big(overline{M}+overline{M}^Tbig)UBig)-Big(underline{M}-underline{M}^TBig)U, hat{V}(A) = big(overline{V}(A)-overline{V}(A)^Tbig)U$$

Further:

$$begin{cases}
overline{M}_{i, j}=sumlimits_{n=1}^Nint_{zinmathbf{R}_+^*}alpha_2^{n, i, j}(z)zlambda^{n, i, j}(z)dz, underline{M}_{i, j}=sumlimits_{n=1}^Nintlimits_{zinmathbf{R}_+^*}alpha_1^{n, i, j}(z)zlambda^{n, i, j}(z)dz\
overline{V}(A)=overline{D}(A)P+Poverline{D}-2Pcirc A, P_{i,j}=sumlimits_{n=1}^Nint_{mathbf{R}_+^*}alpha_2^{n, i, j}(z)z^2lambda^{n, i, j}(z)dz
end{cases}$$
