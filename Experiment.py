import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Lunch Choice App", layout="centered")
st.title("🍔 Discrete Choice Experiment")

# --- INITIALIZE MEMORY ---
if 'votes' not in st.session_state:
    st.session_state.votes = np.array([0, 0, 0])

base_costs = np.array([109, 79, 129])
labels = ['Meatballs', 'Salad', 'Pho']

# Create 3 distinct pages/tabs
tab1, tab2, tab3 = st.tabs(["1. Data Collection", "2. Model Estimation", "3. Policy Forecast"])

counts = st.session_state.votes
total_students = np.sum(counts)

# ==========================================
# PAGE 1: MAKING CHOICES
# ==========================================
with tab1:
    st.header("Step 1: Cast Your Vote")
    st.markdown("What would you choose for lunch today at these current prices?")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(f"🧆 Meatballs\n({base_costs[0]} SEK)", use_container_width=True):
            st.session_state.votes[0] += 1
            st.rerun()
    with col2:
        if st.button(f"🥗 Salad\n({base_costs[1]} SEK)", use_container_width=True):
            st.session_state.votes[1] += 1
            st.rerun()
    with col3:
        if st.button(f"🍜 Pho\n({base_costs[2]} SEK)", use_container_width=True):
            st.session_state.votes[2] += 1
            st.rerun()

    st.info(f"**Total Votes Cast:** {total_students}")

    if st.button("🗑️ Reset Survey"):
        st.session_state.votes = np.array([0, 0, 0])
        st.rerun()

    # Frequency Bar Chart
    if total_students > 0:
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        bars = ax1.bar(labels, counts, color=['#8c564b', '#2ca02c', '#d62728'])
        ax1.set_ylabel('Number of Votes')
        ax1.set_title('Live Vote Frequency')
        
        # Add exact numbers on top of the bars
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')
            
        # Hide top and right spines for a cleaner look
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        st.pyplot(fig1)

# ==========================================
# PAGE 2: MODEL ESTIMATION
# ==========================================
with tab2:
    st.header("Step 2: Model Estimation")

    if total_students == 0:
        st.warning("Please cast at least one vote in the first tab to estimate the model.")
    else:
        # Maximum Likelihood Estimation function
        def neg_log_likelihood(gamma):
            V = gamma[0] * base_costs
            exp_V = np.exp(V - np.max(V)) # Stability
            P = exp_V / np.sum(exp_V)
            LL = np.sum(counts * np.log(P + 1e-10))
            return -LL

        # Run optimizer
        res = minimize(neg_log_likelihood, x0=[-0.01], bounds=[(-1.0, 0.1)])
        est_gamma = res.x[0]

        st.success(f"**Estimated Cost Parameter (γ): {est_gamma:.4f}**")
        st.markdown("The model estimated the parameter above. Let's see how well our mathematical predictions match the real-world observations.")

        # Calculate Predicted vs Observed Probabilities
        V_est = est_gamma * base_costs
        exp_V_est = np.exp(V_est - np.max(V_est))
        P_pred = exp_V_est / np.sum(exp_V_est)
        P_obs = counts / total_students

        # Bar Chart: Observed vs Predicted
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        x = np.arange(len(labels))
        width = 0.35

        ax2.bar(x - width/2, P_obs * 100, width, label='Observed (Real Votes)', color='#1f77b4')
        ax2.bar(x + width/2, P_pred * 100, width, label='Predicted (MNL Model)', color='#ff7f0e')

        ax2.set_ylabel('Market Share (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        # Scale y-axis dynamically based on the highest bar
        max_y = max(max(P_obs), max(P_pred)) * 100
        ax2.set_ylim(0, max_y + 15)
        ax2.legend()
        st.pyplot(fig2)

# ==========================================
# PAGE 3: POLICY FORECAST
# ==========================================
with tab3:
    st.header("Step 3: Policy Forecast")

    if total_students == 0:
        st.warning("Please cast at least one vote in the first tab to build the model.")
    else:
        st.markdown("Using our unique cost parameter, what happens if the cafeteria changes their prices?")

        new_m = st.slider("New Price for Meatballs (SEK)", 50, 200, 109, 1)
        new_s = st.slider("New Price for Salad (SEK)", 50, 200, 79, 1)
        new_p = st.slider("New Price for Pho (SEK)", 50, 200, 129, 1)

        new_costs = np.array([new_m, new_s, new_p])

        # 1. Analytical Forecast
        V_new = est_gamma * new_costs
        exp_V_new = np.exp(V_new - np.max(V_new))
        P_new = exp_V_new / np.sum(exp_V_new)

        # 2. Simulated Forecast (Random Utility)
        num_draws = 5000
        np.random.seed(42)
        eps = -np.log(-np.log(np.random.rand(num_draws, 3)))
        U = V_new + eps
        choices = np.argmax(U, axis=1)
        P_sim = np.array([np.mean(choices == 0), np.mean(choices == 1), np.mean(choices == 2)])

        # Visualization
        fig3, ax3 = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(labels))
        width = 0.35

        ax3.bar(x - width/2, P_new * 100, width, label='Analytical Forecast (%)', color='#2ca02c')
        ax3.bar(x + width/2, P_sim * 100, width, label=f'Simulated Forecast ({num_draws} draws)', color='#9467bd')

        ax3.set_ylabel('Forecasted Market Share (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"{l}\n({c} SEK)" for l, c in zip(labels, new_costs)])
        ax3.set_ylim(0, 100)
        ax3.legend()
        st.pyplot(fig3)
