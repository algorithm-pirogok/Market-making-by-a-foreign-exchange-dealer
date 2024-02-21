import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from modules import Environment
from modules import Control

class Model:
    def __init__(self, q_0: np.ndarray, X_0: float, S_0: np.ndarray, environment: Environment, optimal_control: Control):
        self.q_t, self.X_t, self.S_t, self.Y_t, self.t = q_0.copy(), X_0, S_0.copy(), q_0.copy() * S_0.copy(), 0
        
        self.env = environment
        self.delta_t = self.env.T / self.env.steps
        self.control = optimal_control
        self.q_list, self.X_list, self.S_list, self.Y_list, self.t_list = [q_0], [X_0], [S_0], [q_0 * S_0], [0]
        self.client_list, self.diller_list = [], []

    def _update_list(self, client, diller):
        self.q_list.append(self.q_t.copy())
        self.X_list.append(self.X_t)
        self.S_list.append(self.S_t.copy())
        self.Y_list.append(self.Y_t.copy())
        self.t_list.append(self.t)
        self.client_list.append(client.copy())
        self.diller_list.append(diller.copy())
    
    def _dq(self, client_trades: np.ndarray, diller_trades: np.ndarray):
        """
        dq_t^i = ...
        """
        dt = self.delta_t
        
        clients = np.sum(np.sum(client_trades-client_trades.transpose(0, 2, 1), axis=2), axis=0)
        dillers = np.sum(diller_trades, axis=1) - np.sum(diller_trades, axis=0)
        dq_t = (clients + dillers) * dt / self.S_t
        return dq_t
    
    def _dX(self, client_comissions: np.ndarray, diller_trades: np.ndarray):
        r"""
        dX_t =  ...
        """
        dX_t = (np.sum(client_comissions) - np.sum(self.env.transaction_loss(diller_trades)) * self.delta_t)
        return dX_t
    
    def _dS(self, t: float, noise: np.ndarray, diller_trades: np.ndarray):
        r"""
        dS_t^i = \mu_t^i*S_t^idt + sigma^iS_t^idW_t^i+k^i(\sum_{j=i+1}^d\xi_t^{i, j} - \sum_{j=1}^{i-1} \xi_t^{j, i})S_t^idt
        """
        dt = self.delta_t
        dS_t = (self.env.mu(t) * dt + noise * dt ** 0.5 + self.env.market_impact * (np.sum(diller_trades, axis=1) - np.sum(diller_trades, axis=0))) * self.S_t
        return dS_t
    
    def _step(self):
        """
        Upadate our processes with dynamic rules, described above
        """

        clients_policy = lambda z, t: self.control.clients_policy(self.Y_t, z, t)
        clients_trades, comissions = self.env.sampling(self.t, clients_policy)
        dillers_trades = self.control.dillers_policy(self.t, self.Y_t)
        
        self.q_t += self._dq(
                            client_trades=clients_trades,
                            diller_trades=dillers_trades)
        
        self.X_t += self._dX(
            client_comissions=comissions,
            diller_trades=dillers_trades
        )

        self.S_t += self._dS(t=self.t, 
                             noise=np.random.multivariate_normal(mean=np.zeros((self.env.d)), cov=self.env.sigma),
                             diller_trades=dillers_trades)

        self.Y_t = self.q_t.copy() * self.S_t.copy()

        self.risk_penalty += self.env.risk_loss(self.Y_t) * self.delta_t
        return self.control.clients_policy(self.Y_t, 0, self.t), dillers_trades

    def _plot_list(self):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs[0, 0].plot(self.X_list)
        axs[0, 0].set_title('X_t')
        
        for num_of_stock in range(self.env.d):
            q_t = [q_stock[num_of_stock] for q_stock in self.q_list]
            axs[0, 1].plot(q_t, label=f'{num_of_stock}')
        axs[0, 1].legend()
        axs[0, 1].set_title('q_t')
        
        for num_of_stock in range(self.env.d):
            S_t = [S_stock[num_of_stock] for S_stock in self.S_list]
            axs[0, 2].plot(S_t, label=f'{num_of_stock}')
        axs[0, 2].legend()
        axs[0, 2].set_title('S_t')
        
        for num_of_stock in range(self.env.d):
            Y_t = [Y_stock[num_of_stock] for Y_stock in self.Y_list]
            axs[1, 0].plot(Y_t, label=f'{num_of_stock}')
        axs[1, 0].legend()
        axs[1, 0].set_title('Y_t')
        
        for cluster in range(self.env.n):
            for num_of_input_stock in range(self.env.d):
                for num_of_output_stock in range(self.env.d):
                    if num_of_input_stock == num_of_output_stock:
                        continue
                    delta_t = [client[cluster][num_of_input_stock][num_of_output_stock] for client in self.client_list]
                    axs[1, 1].plot(delta_t, label=f'{cluster}: {num_of_input_stock}->{num_of_output_stock}')
        axs[1, 1].legend()
        axs[1, 1].set_title('Clients comissions')
        
        for num_of_input_stock in range(self.env.d):
                for num_of_output_stock in range(num_of_input_stock+1, self.env.d):
                    xi_t = [diller[num_of_input_stock][num_of_output_stock] for diller in self.diller_list]
                    axs[1, 2].plot(xi_t, label=f'{num_of_input_stock}->{num_of_output_stock}')
        axs[1, 2].legend()
        axs[1, 2].set_title('Dillers trades')
        
        
        plt.show()
    
    def modulation(self):
        """
        Run one attempt to check results
        """
        q_0, X_0, S_0, Y_0, t = self.q_t.copy(), self.X_t, self.S_t.copy(), self.Y_t.copy(), self.t
        self.q_list, self.X_list, self.S_list, self.Y_list, self.t_list = [q_0], [X_0], [S_0], [q_0 * S_0], [0]
        self.risk_penalty = 0
        with tqdm(total=self.env.steps) as pbar:
            while self.t < self.env.T:
                clients, dillers = self._step()
                self.t += self.delta_t
                self._update_list(clients, dillers)
                pbar.update(1)
        ans = self.X_t + np.sum(self.Y_t) - self.risk_penalty - self.env.final_loss(self.Y_t)
        print("Final Metric:", self.X_t + np.sum(self.Y_t) - self.env.final_loss(self.Y_t))
        print(f"Stocks: {q_0}---->{self.q_t}")
        print(f"Prices: {S_0}---->{self.S_t}")
        print(f"Foreign capital: {Y_0}---->{self.Y_t}")
        self._plot_list()
        # self.q_t, self.X_t, self.S_t, self.Y_t, self.t = q_0, X_0, S_0, q_0 * S_0, 0
        return ans