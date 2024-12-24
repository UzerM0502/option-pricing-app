import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from array_output_bs import call_matrix, to_array, put_matrix
from simulation import Simulation
import datetime as dt

# Streamlit App
st.title("Option Pricing Heatmap App")

# User Inputs
st.sidebar.header("Input Parameters")

S = 101.15
vol = 0.09
K = 98.01
r = 0.01
T = 60
N = 10  # No. of time steps
M = 1000
market_value = 3.86

S = st.sidebar.slider("Stock Prices ", 50, 200, 101, step=5)
vol = st.sidebar.slider("Volatility ", 0.01, 0.4, vol)
K = st.sidebar.slider("Strike Price ", 50.0, 150.0, K)
r = st.sidebar.slider("risk free rate ", 0.005, 0.1, r)
T = st.sidebar.slider("Time in days to maturity", 30, 360, T)
stocks = ('CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO')
stock_list = st.selectbox("Select Stocks",
                          stocks, )
T = (T + 1) / 365

# Generate Matrices
call_value_matrix, standard_error_matrix_call = call_matrix(vol, S, T, K, r)
put_value_matrix, standard_error_matrix_put = put_matrix(vol, S, T, K, r)
S_Xlabel = to_array(S)
vol_Ylabel = to_array(vol)
# Display Results
# st.subheader("Call Option Price Matrix")
# st.write(pd.DataFrame(call_value_matrix, index=to_array(vol), columns=to_array(S)))


col1, col2 = st.columns(2)

with col1:
    st.subheader("Call Option Prices")
    S_Xlabel = to_array(S)
    vol_Ylabel = to_array(vol)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(call_value_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=S_Xlabel, yticklabels=vol_Ylabel,
                ax=ax)
    ax.set_xlabel("Stock Prices")
    ax.set_ylabel("Volatility")
    st.pyplot(fig)

    # Standard Error heatmap
    st.subheader("Heatmap of Standard error for call options")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(standard_error_matrix_call, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=S_Xlabel,
                yticklabels=vol_Ylabel, ax=ax)
    ax.set_xlabel("Stock Prices")
    ax.set_ylabel("Volatility")
    st.pyplot(fig)

with col2:
    st.subheader("Put Option Prices")
    S_Xlabel = to_array(S)
    vol_Ylabel = to_array(vol)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(put_value_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=S_Xlabel, yticklabels=vol_Ylabel,
                ax=ax)
    ax.set_xlabel("Stock Prices")
    ax.set_ylabel("Volatility")
    st.pyplot(fig)

    st.subheader("Heatmap of Standard error for put options")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(standard_error_matrix_put, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=S_Xlabel,
                yticklabels=vol_Ylabel, ax=ax)
    ax.set_xlabel("Stock Prices")
    ax.set_ylabel("Volatility")
    st.pyplot(fig)
    st.sidebar.info("Adjust stock price and volatility using sliders to see updated heatmaps and matrices.")

st.subheader("Monte Carlo Simulation of Stock Portfolio")
sim = Simulation(stock_list)
sim = sim.mc_sim_stock_port()

# Plot the results
fig, ax = plt.subplots(figsize=(10, 8))
plt.plot(sim)
ax.set_xlabel("Days")
ax.set_ylabel("Portfolio returns in $")
st.pyplot(fig)
