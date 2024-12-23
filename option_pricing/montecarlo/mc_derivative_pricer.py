import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Initial Derivative parameters
# TODO: takes in an array of 5 vols and stocks and calculates the call price for each combo therefore outputs 5x5 matrix
S = 101.15
K = 98.01
vol = 0.0991
r = 0.01
N = 10  # No. of time steps
M = 1000  # No. of sims
market_value = 3.86

# start_date = dt.datetime.now()
# expiry_date = dt.datetime(2025, 2, 20)
start_date = dt.datetime(2022, 1, 17)
expiry_date = dt.datetime(2022, 3, 17)
T = expiry_date - start_date
T = (T.days + 1) / 365

# Precompute Constants
dt = T / N

nudt = (r - 0.5 * vol ** 2) * dt
volsdt = vol * np.sqrt(dt)
lnS = np.log(S)


# standard error place holders


def call_calculator(lnS, nudt, volsdt, M, N, K, T, r):
    sum_CT = 0
    sum_CT2 = 0

    # Monte Carlo Method
    for i in range(M):
        lnSt = lnS
        for j in range(N):
            lnSt = lnSt + nudt + volsdt * np.random.normal()

        ST = np.exp(lnSt)
        CT = max(0, ST - K)
        sum_CT = sum_CT + CT
        sum_CT2 = sum_CT2 + CT ** 2

    # Calculate Call Value
    C0 = np.exp(-r * T) * sum_CT / M
    call_value = np.round(C0, 2)

    # Calculate standard error
    std = np.sqrt((sum_CT2 - sum_CT * sum_CT / M) * np.exp(-2 * r * T) / (M - 1))
    standard_error = std / np.sqrt(M)
    standard_error = np.round(standard_error, 2)

    return call_value, standard_error


C, SE = call_calculator(lnS, nudt, volsdt, M, N, K, T, r)
