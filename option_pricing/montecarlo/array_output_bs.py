import datetime as dt
import numpy as np
from mc_derivative_pricer import call_calculator, put_calculator
import seaborn as sns
import matplotlib.pyplot as plt

# Initial Derivative parameters
K = 98.01
vol = 0.0991
S = 101.15
r = 0.01
N = 10  # No. of time steps
M = 1000
market_value = 3.86

start_date = dt.datetime(2022, 1, 17)
expiry_date = dt.datetime(2022, 3, 17)
T = expiry_date - start_date
T = (T.days + 1) / 365

# Precompute Constants

'''dt = T / N
nudt = np.array([])
volsdt = np.array([])
lnS = np.array([])
S = np.linspace(0.85 * S, 1.15 * S, 20)
vol = np.linspace(0.85 * vol, 1.15 * vol, 20)'''


def to_array(S):
    S = np.linspace(0.9 * S, 1.1 * S, 10)
    S = np.round(S,2)
    return S


def call_matrix(vol, S, T, K, r):
    dt = T / 10
    nudt = np.array([])
    volsdt = np.array([])
    lnS = np.array([])
    S = to_array(S)
    vol = to_array(vol)
    for i in range(len(vol)):
        nudt = np.append(nudt, (r - 0.5 * vol[i] ** 2) * dt)
        volsdt = np.append(volsdt, vol[i] * np.sqrt(dt))
    for i in range(len(S)):
        lnS = np.append(lnS, np.log(S[i]))
    call_value_matrix = np.zeros((len(volsdt), len(lnS)))
    standard_error_matrix = np.zeros((len(volsdt), len(lnS)))

    for v in range(len(volsdt)):
        for l in range(len(lnS)):
            call_value, standard_error = call_calculator(lnS[l], nudt[v], volsdt[v], K, T, r)
            call_value_matrix[v, l] = call_value
            standard_error_matrix[v, l] = standard_error

    return call_value_matrix, standard_error_matrix

def put_matrix(vol, S, T, K, r):
    dt = T / 10
    nudt = np.array([])
    volsdt = np.array([])
    lnS = np.array([])
    S = to_array(S)
    vol = to_array(vol)
    for i in range(len(vol)):
        nudt = np.append(nudt, (r - 0.5 * vol[i] ** 2) * dt)
        volsdt = np.append(volsdt, vol[i] * np.sqrt(dt))
    for i in range(len(S)):
        lnS = np.append(lnS, np.log(S[i]))
    put_value_matrix = np.zeros((len(volsdt), len(lnS)))
    standard_error_matrix = np.zeros((len(volsdt), len(lnS)))

    for v in range(len(volsdt)):
        for l in range(len(lnS)):
            put_value, standard_error = put_calculator(lnS[l], nudt[v], volsdt[v], K, T, r)
            call_value_matrix[v, l] = put_value
            standard_error_matrix[v, l] = standard_error

    return call_value_matrix, standard_error_matrix

call_value_matrix, standard_error_matrix = call_matrix(vol, S, T, K, r)
plt.figure(figsize=(16, 8))
S_Xlabel = to_array(S)
vol_Ylabel = to_array(vol)
sns.heatmap(call_value_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True,
            xticklabels=S_Xlabel, yticklabels=vol_Ylabel)

plt.title('Heatmap of Call Values')
plt.xlabel('Stock Price')
plt.ylabel('Volatility')
plt.show()
