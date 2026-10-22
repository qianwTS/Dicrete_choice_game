import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Lunch Choice Estimator", layout="centered")

st.title("🍔 Interactive Lunch Estimator & Forecaster")
st.markdown("Enter the survey results from the class. The app will estimate the cost parameter (γ) that best explains your choices, and then use it to forecast what happens if prices change!")

# Default base prices
base_costs = np.array([109, 79, 129])
labels = ['Meatballs', 'Salad', 'Pho']

# --- STEP 1: SURVEY RESULTS ---
st.header("1. Class Survey Results")
st.markdown("How many students chose each option at the current prices?")

col1, col2, col3 = st.columns(3)
with col1:
    n_meatballs = st.number_input("Meatballs (109 SEK)", min_value=0, value=30, step=1)
with col2:
    n_salad = st.number_input("Salad (79 SEK)", min_value=0, value=50, step=1)
with col3:
    n_pho = st.number_input("Pho (129 SEK)", min_value=0, value=20, step=1)

counts = np.array([n_meatballs, n_salad, n_pho])
total_students = np.sum(counts)

# --- STEP 2: ESTIMATION (MLE) ---
st.header("2. MNL Model Estimation")

if total_students == 0:
    st.warning("Please enter at least one vote to estimate the model.")
else:
    # Maximum Likelihood Estimation function
    def neg_log_likelihood(gamma):
        V = gamma[0] * base_costs
        # Subtract max V for numerical stability in exponentiation
        exp_V = np.exp(V - np.max(V))
        P = exp_V / np.sum(exp_V)
        # LL = Sum(N_i * ln(P_i))
        # Add tiny epsilon to avoid log(0)
        LL = np.sum(counts * np.log(P + 1e-10))
        return -LL

    # Run optimizer (looking for a negative cost parameter)
    res = minimize(neg_log_likelihood, x0=[-0.01], bounds=[(-1.0, 0.1)])
    estimated_gamma = res.x[0]
    
    st.success(f"**Estimated Cost Parameter (γ): {estimated_gamma:.4f}**")
    st.markdown("*The model used Maximum Likelihood Estimation to find the parameter that best fits your class's survey data.*")

    # --- STEP 3: FORECASTING ---
    st.header("3. Policy Forecast")
    st.markdown("Now that we have our model, what happens if the restaurant increases the price of Meatballs?")
    
    new_meatball_price = st.slider("New Price for Meatballs (SEK)", min_value=50, max_value=200, value=109, step=1)
    new_costs = np.array([new_meatball_price, 79, 129])
    
    # Calculate New Probabilities using the ESTIMATED gamma
    V_new = estimated_gamma * new_costs
    exp_V_new = np.exp(V_new - np.max(V_new))
    P_new = exp_V_new / np.sum(exp_V_new)
    
    # Random Utility Simulation based on the new prices
    num_draws = 5000
    np.random.seed(42)
    eps = -np.log(-np.log(np.random.rand(num_draws, 3)))
    U = V_new + eps
    choices = np.argmax(U, axis=1)
    P_simulated = np.array([np.mean(choices == 0), np.mean(choices == 1), np.mean(choices == 2)])
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, P_new * 100, width, label='Analytical MNL Forecast (%)', color='#2ca02c')
    ax.bar(x + width/2, P_simulated * 100, width, label='Simulated Forecast (%)', color='#ff7f0e')

    ax.set_ylabel('Forecasted Market Share (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n({c} SEK)" for l, c in zip(labels, new_costs)])
    ax.set_ylim(0, 100)
    ax.legend()
    st.pyplot(fig)
