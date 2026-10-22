import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Live Lunch Estimator", layout="centered")

st.title("🍔 Live Lunch Choice Experiment")
st.markdown("Cast your vote! We are collecting real data from the class to estimate our mathematical cost parameter, and then using it to forecast behavior.")

# --- INITIALIZE MEMORY ---
# This allows Streamlit to remember the votes as different students click the buttons
if 'votes' not in st.session_state:
    st.session_state.votes = np.array([0, 0, 0])

base_costs = np.array([109, 79, 129])
labels = ['Meatballs', 'Salad', 'Pho']

# --- STEP 1: LIVE VOTING ---
st.header("1. Live Class Survey")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🧆 Choose Meatballs (109 SEK)", use_container_width=True):
        st.session_state.votes[0] += 1
        st.rerun()
with col2:
    if st.button("🥗 Choose Salad (79 SEK)", use_container_width=True):
        st.session_state.votes[1] += 1
        st.rerun()
with col3:
    if st.button("🍜 Choose Pho (129 SEK)", use_container_width=True):
        st.session_state.votes[2] += 1
        st.rerun()

counts = st.session_state.votes
total_students = np.sum(counts)

# Live Scoreboard
st.info(f"**Total Votes Cast:** {total_students}  ➔  (Meatballs: {counts[0]} | Salad: {counts[1]} | Pho: {counts[2]})")

if st.button("🗑️ Reset Survey"):
    st.session_state.votes = np.array([0, 0, 0])
    st.rerun()

st.divider()

# --- STEP 2: ESTIMATION (MLE) ---
st.header("2. MNL Model Estimation")

if total_students == 0:
    st.warning("Waiting for the first vote to estimate the model...")
else:
    # Maximum Likelihood Estimation function
    def neg_log_likelihood(gamma):
        V = gamma[0] * base_costs
        exp_V = np.exp(V - np.max(V)) # Subtract max V for numerical stability
        P = exp_V / np.sum(exp_V)
        # LL = Sum(N_i * ln(P_i))
        LL = np.sum(counts * np.log(P + 1e-10))
        return -LL

    # Run optimizer to find the best gamma
    res = minimize(neg_log_likelihood, x0=[-0.01], bounds=[(-1.0, 0.1)])
    estimated_gamma = res.x[0]
    
    st.success(f"**Live Estimated Cost Parameter (γ): {estimated_gamma:.4f}**")
    st.markdown("*The model just used Maximum Likelihood Estimation to find the exact parameter that best explains your real choices!*")

    st.divider()

    # --- STEP 3: FORECASTING ---
    st.header("3. Policy Forecast")
    st.markdown("Now that we have our mathematical model of *your* brains, what happens if the restaurant increases the price of Meatballs?")
    
    new_meatball_price = st.slider("New Price for Meatballs (SEK)", min_value=50, max_value=200, value=109, step=1)
    new_costs = np.array([new_meatball_price, 79, 129])
    
    # Calculate New Probabilities using the LIVE estimated gamma
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
