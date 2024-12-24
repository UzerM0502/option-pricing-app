import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

class Simulation:
    # Refactored function to fetch data once and compute required metrics
    def __init__(self, stock_list=['CBA']):
        self.stocks = [stock + '.AX' for stock in stock_list]


    def get_stock_data(self, stocks, start, end):
        data = yf.download(stocks, start=start, end=end)
        mean_returns = data['Close'].pct_change().mean()
        cov_matrix = data['Close'].pct_change().cov()
        return mean_returns, cov_matrix

    def mc_sim_stock_port(self):
        # Input params
        # stock_list = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
        stock_list = ['CBA']
        stocks = [stock + '.AX' for stock in stock_list]
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=300)

        # Fetch mean returns and covariance matrix
        mean_returns, cov_matrix = self.get_stock_data(stocks, start_date, end_date)

        # Generate random weights and normalize
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)

        # Monte Carlo Method implementation
        mc_sims = 100  # number of simulations
        T = 100  # number of days
        initial_port_value = 10000

        # Create a mean matrix and portfolio simulations matrix
        mean_matrix = np.tile(mean_returns.values, (T, 1)).T  # Shape: (6, T)
        portfolio_sims = np.zeros((T, mc_sims))

        # Main loop
        for m in range(mc_sims):
            Z = np.random.normal(size=(T, len(weights))).T  # Shape: (6, T)
            L = np.linalg.cholesky(cov_matrix)
            daily_returns = mean_matrix + np.dot(L, Z)  # Shape: (6, T)
            portfolio_sims[:, m] = np.cumprod(np.dot(weights, daily_returns) + 1) * initial_port_value

        return portfolio_sims


sim = Simulation()
sim = sim.mc_sim_stock_port()

# Plot the results
plt.plot(sim)
plt.ylabel("Portfolio returns in $")
plt.xlabel("Days")
plt.title("Monte Carlo Simulation of Stock Portfolio")
plt.show()