import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Lunch Choice Simulator", layout="centered")

st.title("🍔 Interactive Lunch Choice Forecast")
st.markdown("Adjust the prices in the sidebar to see how market shares shift based on the **Multinomial Logit (MNL)** model. You can also compare the analytical formula to a brute-force random utility simulation!")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Set Lunch Prices (SEK)")
cost_meatballs = st.sidebar.slider("Meatballs", min_value=50, max_value=200, value=109, step=1)
cost_salad = st.sidebar.slider("Salad", min_value=50, max_value=200, value=79, step=1)
cost_pho = st.sidebar.slider("Pho", min_value=50, max_value=200, value=129, step=1)

st.sidebar.header("2. Model Parameters")
gamma_cost = st.sidebar.slider("Cost Parameter (γ)", min_value=-0.10, max_value=0.00, value=-0.02, step=0.005)
num_draws = st.sidebar.select_slider("Random Gumbel Draws", options=[100, 500, 1000, 5000, 10000], value=1000)

# --- MATH & SIMULATION ---
# 1. Deterministic Utilities (V = gamma * cost)
V_meatballs = gamma_cost * cost_meatballs
V_salad = gamma_cost * cost_salad
V_pho = gamma_cost * cost_pho

V_array = np.array([V_meatballs, V_salad, V_pho])

# 2. Analytical MNL Probabilities
exp_V = np.exp(V_array)
analytical_probs = exp_V / np.sum(exp_V)

# 3. Random Utility Simulation (U = V + epsilon)
np.random.seed(42) # Keeps the random jumps stable visually
eps = -np.log(-np.log(np.random.rand(num_draws, 3))) # Gumbel draws
U = V_array + eps

# Find the maximum utility choice for each "person"
choices = np.argmax(U, axis=1)
simulated_probs = np.array([np.mean(choices == 0), np.mean(choices == 1), np.mean(choices == 2)])

# --- VISUALIZATION ---
labels = ['Meatballs', 'Salad', 'Pho']
costs = [cost_meatballs, cost_salad, cost_pho]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(labels))
width = 0.35

ax.bar(x - width/2, analytical_probs * 100, width, label='MNL Formula (%)', color='#2ca02c')
ax.bar(x + width/2, simulated_probs * 100, width, label=f'Simulated ({num_draws} draws) (%)', color='#ff7f0e')

ax.set_ylabel('Market Share (%)')
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n({c} SEK)" for l, c in zip(labels, costs)])
ax.set_ylim(0, 100)
ax.legend()
st.pyplot(fig)

# --- DATA TABLE ---
st.subheader("Data Summary")
df = pd.DataFrame({
    "Option": labels,
    "Cost (SEK)": costs,
    "Systematic Utility (V)": np.round(V_array, 2),
    "Analytical Share": [f"{p*100:.1f}%" for p in analytical_probs],
    "Simulated Share": [f"{p*100:.1f}%" for p in simulated_probs]
})
st.dataframe(df, use_container_width=True)
