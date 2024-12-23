import datetime as dt
import numpy as np
from mc_derivative_pricer import call_calculator
import seaborn as sns
import matplotlib.pyplot as plt

# Initial Derivative parameters
K = 98.01
# TODO: input the stock price and vol from keyboard and go +/- 15% with 20 steps in total
vol = 0.0991
S = 101.15

S = np.round(S, 2)
vol = np.round(vol, 2)
r = 0.01
N = 10  # No. of time steps
M = 1000
market_value = 3.86

start_date = dt.datetime(2022, 1, 17)
expiry_date = dt.datetime(2022, 3, 17)
T = expiry_date - start_date
T = (T.days + 1) / 365


# Precompute Constants

dt = T / N
nudt = np.array([])
volsdt = np.array([])
lnS = np.array([])
S = np.linspace(0.85 * S, 1.15 * S, 20)
vol = np.linspace(0.85 * vol, 1.15 * vol, 20)
for i in range(len(vol)):
    nudt = np.append(nudt, (r - 0.5 * vol[i] ** 2) * dt)
    volsdt = np.append(volsdt, vol[i] * np.sqrt(dt))
for i in range(len(S)):
    lnS = np.append(lnS, np.log(S[i]))
call_value_matrix = np.zeros((len(volsdt), len(lnS)))
standard_error_matrix = np.zeros((len(volsdt), len(lnS)))

for v in range(len(volsdt)):
    for l in range(len(lnS)):
        call_value, standard_error = call_calculator(lnS[l], nudt[v], volsdt[v], M, N, K, T, r)
        call_value_matrix[v, l] = call_value
        standard_error_matrix[v, l] = standard_error

plt.figure(figsize=(16, 8))
sns.heatmap(call_value_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True,
            xticklabels=S, yticklabels=vol)

plt.title('Heatmap of Call Values')
plt.xlabel('Stock Price')
plt.ylabel('Volatility')
plt.show()
