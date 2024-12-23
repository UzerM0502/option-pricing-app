import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Black-Scholes Option Pricing Formula (for demonstration)
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes Call Option Price.
    S: Stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate
    sigma: Volatility (standard deviation of stock's returns)
    """
    from scipy.stats import norm

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes Put Option Price.
    """
    from scipy.stats import norm

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Generate Call and Put Value Matrices
def generate_option_matrices(S_range, K_range, T, r, sigma):
    call_matrix = np.zeros((len(S_range), len(K_range)))
    put_matrix = np.zeros((len(S_range), len(K_range)))

    for i, S in enumerate(S_range):
        for j, K in enumerate(K_range):
            call_matrix[i, j] = black_scholes_call(S, K, T, r, sigma)
            put_matrix[i, j] = black_scholes_put(S, K, T, r, sigma)

    return call_matrix, put_matrix

# Streamlit App
st.title("Option Pricing Heatmap App")

# User Inputs
st.sidebar.header("Input Parameters")
stock_prices = st.sidebar.slider("Stock Prices Range (Min, Max)", 50, 200, (80, 120), step=5)
strike_prices = st.sidebar.slider("Strike Prices Range (Min, Max)", 50, 200, (80, 120), step=5)
volatility = st.sidebar.slider("Volatility (0.1 - 1.0)", 0.1, 1.0, 0.2)
time_to_maturity = st.sidebar.slider("Time to Maturity (Years)", 0.1, 5.0, 1.0)
risk_free_rate = st.sidebar.slider("Risk-Free Rate (0.01 - 0.1)", 0.01, 0.1, 0.03)

# Generate Price Ranges
S_range = np.arange(stock_prices[0], stock_prices[1] + 1, 5)
K_range = np.arange(strike_prices[0], strike_prices[1] + 1, 5)

# Generate Matrices
call_matrix, put_matrix = generate_option_matrices(S_range, K_range, time_to_maturity, risk_free_rate, volatility)

# Display Results
st.subheader("Call Option Price Matrix")
st.write(pd.DataFrame(call_matrix, index=S_range, columns=K_range))

st.subheader("Put Option Price Matrix")
st.write(pd.DataFrame(put_matrix, index=S_range, columns=K_range))

# Heatmap
st.subheader("Heatmap of Call Option Prices")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(call_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=K_range, yticklabels=S_range, ax=ax)
ax.set_xlabel("Strike Prices")
ax.set_ylabel("Stock Prices")
st.pyplot(fig)

st.subheader("Heatmap of Put Option Prices")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(put_matrix, annot=True, fmt=".2f", cmap="viridis", xticklabels=K_range, yticklabels=S_range, ax=ax)
ax.set_xlabel("Strike Prices")
ax.set_ylabel("Stock Prices")
st.pyplot(fig)

st.sidebar.info("Adjust parameters on the left to see updated heatmaps and matrices.")
