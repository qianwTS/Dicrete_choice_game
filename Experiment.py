import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Lunch Choice App", layout="centered")

# --- COMPACT VERTICAL CSS ---
st.markdown("""
    <style>
    /* Reduce top padding of the whole app */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
    }
    /* Hide the default Streamlit header to save space */
    header {visibility: hidden;}
    /* Make buttons bold and slightly taller for easy tapping */
    .stButton button {
        min-height: 45px !important;
        font-size: 15px !important;
        font-weight: bold !important;
        margin-bottom: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🍔 Discrete Choice Experiment")

# --- 1. GLOBAL SHARED MEMORY ---
@st.cache_resource
def get_shared_votes():
    return [0, 0, 0] # [Meatballs, Salad, Beef Stew]

shared_votes = get_shared_votes()

# --- 2. LOCAL BROWSER MEMORY ---
if 'has_voted' not in st.session_state:
    st.session_state.has_voted = False

base_costs = np.array([109, 79, 129])
labels = ['Meatballs', 'Salad', 'Beef Stew']

tab1, tab2, tab3 = st.tabs(["1. Data Collection", "2. Model Estimation", "3. Policy Forecast"])

total_students = np.sum(shared_votes)

# ==========================================
# PAGE 1: MAKING CHOICES (Vertical List Layout)
# ==========================================
with tab1:
    if st.session_state.has_voted:
        st.success("✅ Vote recorded! Watch the live data update below.")
    else:
        st.markdown("#### What would you choose for lunch today?")

    # 100% Reliable Unsplash URLs (Cropped perfectly by their server)
    img_meatballs = "https://images.unsplash.com/photo-1529042410759-befb1204b468?auto=format&fit=crop&w=800&h=250&q=80"
    img_salad = "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?auto=format&fit=crop&w=800&h=250&q=80"
    # Permanent, high-reliability image of a hearty rustic stew
    img_stew = "https://images.unsplash.com/photo-1534939561126-855b8675edd7?auto=format&fit=crop&w=800&h=250&q=80](https://images.unsplash.com/photo-1534939561126-855b8675edd7?auto=format&fit=crop&w=800&h=250&q=80)"

    img_style = "width: 100%; height: 90px; object-fit: cover; border-radius: 8px; margin-bottom: 5px;"

    # --- ROW 1: MEATBALLS ---
    st.markdown(f'<img src="{img_meatballs}" style="{img_style}">', unsafe_allow_html=True)
    if st.button("🧆 Meatballs (109 SEK)", disabled=st.session_state.has_voted, use_container_width=True, key="btn_m"):
        shared_votes[0] += 1
        st.session_state.has_voted = True
        st.rerun()

    # --- ROW 2: SALAD ---
    st.markdown(f'<img src="{img_salad}" style="{img_style}">', unsafe_allow_html=True)
    if st.button("🥗 Fresh Salad (79 SEK)", disabled=st.session_state.has_voted, use_container_width=True, key="btn_s"):
        shared_votes[1] += 1
        st.session_state.has_voted = True
        st.rerun()

    # --- ROW 3: BEEF STEW ---
    st.markdown(f'<img src="{img_stew}" style="{img_style}">', unsafe_allow_html=True)
    if st.button("🥘 Beef Stew (129 SEK)", disabled=st.session_state.has_voted, use_container_width=True, key="btn_stew"):
        shared_votes[2] += 1
        st.session_state.has_voted = True
        st.rerun()

    # --- LIVE DATA CHART ---
    if total_students > 0:
        fig1, ax1 = plt.subplots(figsize=(6, 2.5))
        bars = ax1.bar(labels, shared_votes, color=['#8c564b', '#2ca02c', '#d62728'])
        ax1.set_ylabel('Votes')
        ax1.set_title(f'Live Class Votes (Total: {total_students})')
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        st.pyplot(fig1)

# ==========================================
# PAGE 2: MODEL ESTIMATION
# ==========================================
with tab2:
    st.header("Step 2: Model Estimation")

    if total_students == 0:
        st.warning("Waiting for class data...")
    else:
        def neg_log_likelihood(gamma):
            V = gamma[0] * base_costs
            exp_V = np.exp(V - np.max(V))
            P = exp_V / np.sum(exp_V)
            LL = np.sum(np.array(shared_votes) * np.log(P + 1e-10))
            return -LL

        res = minimize(neg_log_likelihood, x0=[-0.01], bounds=[(-1.0, 0.1)])
        est_gamma = res.x[0]

        st.success(f"**Estimated Cost Parameter (γ): {est_gamma:.4f}**")
        
        V_est = est_gamma * base_costs
        exp_V_est = np.exp(V_est - np.max(V_est))
        P_pred = exp_V_est / np.sum(exp_V_est)
        P_obs = np.array(shared_votes) / total_students

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        x = np.arange(len(labels))
        width = 0.35

        ax2.bar(x - width/2, P_obs * 100, width, label='Observed (Real Votes)', color='#1f77b4')
        ax2.bar(x + width/2, P_pred * 100, width, label='Predicted (MNL Model)', color='#ff7f0e')

        ax2.set_ylabel('Market Share (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
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
        st.warning("Waiting for class data...")
    else:
        new_m = st.slider("New Price for Meatballs (SEK)", 50, 200, 109, 1)
        new_s = st.slider("New Price for Salad (SEK)", 50, 200, 79, 1)
        new_stew = st.slider("New Price for Beef Stew (SEK)", 50, 200, 129, 1)

        new_costs = np.array([new_m, new_s, new_stew])

        V_new = est_gamma * new_costs
        exp_V_new = np.exp(V_new - np.max(V_new))
        P_new = exp_V_new / np.sum(exp_V_new)

        num_draws = 5000
        np.random.seed(42)
        eps = -np.log(-np.log(np.random.rand(num_draws, 3)))
        U = V_new + eps
        choices = np.argmax(U, axis=1)
        P_sim = np.array([np.mean(choices == 0), np.mean(choices == 1), np.mean(choices == 2)])

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        x = np.arange(len(labels))
        width = 0.35

        ax3.bar(x - width/2, P_new * 100, width, label='Analytical Forecast (%)', color='#2ca02c')
        ax3.bar(x + width/2, P_sim * 100, width, label=f'Simulated Forecast', color='#9467bd')

        ax3.set_ylabel('Forecasted Market Share (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"{l}\n({c} SEK)" for l, c in zip(labels, new_costs)])
        ax3.set_ylim(0, 100)
        ax3.legend()
        st.pyplot(fig3)

# ==========================================
# HIDDEN INSTRUCTOR TOOLS
# ==========================================
with st.expander("⚙️ Instructor Tools"):
    admin_password = st.text_input("Enter password", type="password")
    if admin_password == "admin123": 
        
        # New button to inject 25 fake votes for testing!
        if st.button("🧪 Inject 25 Test Votes"):
            shared_votes[0] = 12  # 12 for Meatballs
            shared_votes[1] = 8   # 8 for Salad
            shared_votes[2] = 5   # 5 for Beef Stew
            st.session_state.has_voted = True
            st.rerun()

        if st.button("🗑️ Reset All Class Votes"):
            shared_votes[0] = shared_votes[1] = shared_votes[2] = 0
            st.session_state.has_voted = False
            st.rerun()
            st.session_state.has_voted = False
            st.rerun()
