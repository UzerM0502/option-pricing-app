import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import math
import scipy.stats as stats
import yfinance as yf

# TODO: refactor these two functions so that we get data once only and passes down to these params

# functions to get data and mean and covariance
def get_mean_returns(stocks, start, end):
    data = yf.download(stocks, start=start, end=end)
    mean_returns = data['Close'].pct_change().mean()
    return mean_returns


def get_cov_matrix(stocks, start, end):
    data = yf.download(stocks, start=start, end=end)
    cov_matrix = data['Close'].pct_change().cov()
    return cov_matrix


# Input params
stock_list = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stock_list]
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=300)

mean_returns = get_mean_returns(stocks, start_date, end_date)
cov_matrix = get_cov_matrix(stocks, start_date, end_date)

# Generate random weights then normalize

weights = np.random.random(len(mean_returns))
weights = weights / np.sum(weights)

# Monte Carlo Method implementation

mc_sims = 100  # number of simulations
T = 100  # number of days
initial_port_value = 10000
mean_matrix = np.full(shape=(T, len(weights)), fill_value=mean_returns)
mean_matrix = mean_matrix.T
portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

# Main loop
# Assumes mean returns are distributed by multivariate distribution

print(mean_returns.shape)
print(mean_returns)

for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(cov_matrix)
    daily_returns = mean_returns + np.inner(L, Z)
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, daily_returns.T) + 1) * initial_port_value

plt.plot(portfolio_sims)
plt.ylabel("Portfolio returns in $")
plt.xlabel("Days")
plt.title("Monte Carlo Simulation of stock portfolio")
plt.show()
