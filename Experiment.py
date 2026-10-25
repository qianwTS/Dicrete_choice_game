import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Lunch Choice App", layout="centered")

# --- COMPACT VERTICAL CSS ---
st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
    header {visibility: hidden;}
    .stButton button { min-height: 45px !important; font-size: 15px !important; font-weight: bold !important; margin-bottom: 10px !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🍔 Discrete Choice Experiment")

# --- 1. GLOBAL MICRODATA MEMORY ---
@st.cache_resource
def get_shared_data():
    return [] 

shared_data = get_shared_data()

# --- 2. LOCAL BROWSER MEMORY (4-TASK SP EXPERIMENT) ---
if 'has_voted' not in st.session_state:
    st.session_state.has_voted = False
    st.session_state.task_idx = 0
    
    price_levels = [79, 99, 119, 139, 159]
    st.session_state.scenarios = [
        {'m': int(np.random.choice(price_levels)),
         's': int(np.random.choice(price_levels)),
         'b': int(np.random.choice(price_levels))}
        for _ in range(4)
    ]

labels = ['Meatballs', 'Salad', 'Beef Stew']
total_obs = len(shared_data)
counts = [sum(1 for d in shared_data if d['choice'] == i) for i in range(3)] if total_obs > 0 else [0,0,0]

tab1, tab2, tab3 = st.tabs(["1. Survey", "2. Model Estimation", "3. Policy Forecast"])

# ==========================================
# PAGE 1: REPEATED SP EXPERIMENT (4 Tasks)
# ==========================================
with tab1:
    if st.session_state.has_voted:
        st.success("✅ All 4 scenarios completed! Watch the live data update below.")
    else:
        st.markdown(f"#### Scenario {st.session_state.task_idx + 1} of 4")
        st.caption("Prices vary. What is your choice for this scenario?")

        p_m = st.session_state.scenarios[st.session_state.task_idx]['m']
        p_s = st.session_state.scenarios[st.session_state.task_idx]['s']
        p_b = st.session_state.scenarios[st.session_state.task_idx]['b']

        img_meatballs = "https://images.unsplash.com/photo-1529042410759-befb1204b468?auto=format&fit=crop&w=800&h=250&q=80"
        img_salad = "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?auto=format&fit=crop&w=800&h=250&q=80"
        img_stew = "https://images.unsplash.com/photo-1534939561126-855b8675edd7?auto=format&fit=crop&w=800&h=250&q=80"
        img_style = "width: 100%; height: 90px; object-fit: cover; border-radius: 8px; margin-bottom: 5px;"

        st.markdown(f'<img src="{img_meatballs}" style="{img_style}">', unsafe_allow_html=True)
        if st.button(f"🧆 Meatballs ({p_m} SEK)", use_container_width=True, key="btn_m"):
            shared_data.append({'choice': 0, 'p_m': p_m, 'p_s': p_s, 'p_b': p_b})
            st.session_state.task_idx += 1
            if st.session_state.task_idx >= 4: st.session_state.has_voted = True
            st.rerun()

        st.markdown(f'<img src="{img_salad}" style="{img_style}">', unsafe_allow_html=True)
        if st.button(f"🥗 Fresh Salad ({p_s} SEK)", use_container_width=True, key="btn_s"):
            shared_data.append({'choice': 1, 'p_m': p_m, 'p_s': p_s, 'p_b': p_b})
            st.session_state.task_idx += 1
            if st.session_state.task_idx >= 4: st.session_state.has_voted = True
            st.rerun()

        st.markdown(f'<img src="{img_stew}" style="{img_style}">', unsafe_allow_html=True)
        if st.button(f"🥘 Beef Stew ({p_b} SEK)", use_container_width=True, key="btn_stew"):
            shared_data.append({'choice': 2, 'p_m': p_m, 'p_s': p_s, 'p_b': p_b})
            st.session_state.task_idx += 1
            if st.session_state.task_idx >= 4: st.session_state.has_voted = True
            st.rerun()

    st.divider()
    if total_obs > 0:
        fig1, ax1 = plt.subplots(figsize=(6, 2.5))
        bars = ax1.bar(labels, counts, color=['#8c564b', '#2ca02c', '#d62728'])
        ax1.set_ylabel('Choices Made')
        ax1.set_title(f'Live SP Data (Total Choices: {total_obs})')
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        st.pyplot(fig1)

# ==========================================
# MLE ESTIMATION
# ==========================================
est_gamma, est_asc_m, est_asc_s = 0.0, 0.0, 0.0

if total_obs > 0:
    def nll_mnl(params):
        gamma, a_m, a_s = params
        LL = 0
        for r in shared_data:
            v_m = gamma * r['p_m'] + a_m
            v_s = gamma * r['p_s'] + a_s
            v_b = gamma * r['p_b']
            exp_v = np.exp([v_m, v_s, v_b] - np.max([v_m, v_s, v_b]))
            p = exp_v / np.sum(exp_v)
            LL += np.log(p[r['choice']] + 1e-10)
        return -LL
    
    res = minimize(nll_mnl, x0=[-0.01, 0, 0], bounds=[(-1.0, 0.1), (None, None), (None, None)])
    est_gamma, est_asc_m, est_asc_s = res.x

# ==========================================
# PAGE 2: MODEL ESTIMATION (Monte Carlo)
# ==========================================
with tab2:
    st.header("MNL Model Estimation")
    if total_obs == 0:
        st.warning("Waiting for data...")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("γ (Cost)", f"{est_gamma:.4f}")
        c2.metric("ASC (Meatballs)", f"{est_asc_m:.2f}")
        c3.metric("ASC (Salad)", f"{est_asc_s:.2f}")
        
        st.markdown("#### Stochastic Microsimulation")
        st.markdown("Instead of analytical probabilities, we add a random Gumbel error to each student's utility to simulate a discrete choice.")
        
        # New Slider for simulation iterations
        num_sims = st.slider("Monte Carlo Iterations per Student", min_value=1, max_value=100, value=1, key="sim1")
        
        P_pred_counts = np.zeros(3)
        np.random.seed(42) # Keeps the charts from jittering randomly when you move the slider
        
        for r in shared_data:
            v_m = est_gamma * r['p_m'] + est_asc_m
            v_s = est_gamma * r['p_s'] + est_asc_s
            v_b = est_gamma * r['p_b']
            
            for _ in range(num_sims):
                # Add random Gumbel noise to deterministic utility
                eps = np.random.gumbel(loc=0, scale=1, size=3)
                U = np.array([v_m, v_s, v_b]) + eps
                
                # Agent strictly chooses the max utility
                choice = np.argmax(U)
                P_pred_counts[choice] += 1
                
        P_pred = P_pred_counts / (total_obs * num_sims)
        P_obs = np.array(counts) / total_obs

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        x = np.arange(len(labels))
        ax2.bar(x - 0.15, P_obs * 100, 0.3, label='Observed Shares', color='#1f77b4')
        ax2.bar(x + 0.15, P_pred * 100, 0.3, label='Predicted (Monte Carlo)', color='#ff7f0e')
        ax2.set_ylabel('Market Share (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()
        st.pyplot(fig2)

# ==========================================
# PAGE 3: POLICY FORECAST (Monte Carlo)
# ==========================================
with tab3:
    st.header("Policy Forecast")
    if total_obs == 0:
        st.warning("Waiting for data...")
    else:
        new_m = st.slider("New Price Meatballs (SEK)", 50, 200, 109, 1)
        new_s = st.slider("New Price Salad (SEK)", 50, 200, 79, 1)
        new_b = st.slider("New Price Beef Stew (SEK)", 50, 200, 129, 1)

        num_sims_forecast = st.slider("Monte Carlo Iterations per Student", min_value=1, max_value=100, value=10, key="sim2")

        P_forecast_counts = np.zeros(3)
        np.random.seed(123) 
        
        for r in shared_data:
            v_m = est_gamma * new_m + est_asc_m
            v_s = est_gamma * new_s + est_asc_s
            v_b = est_gamma * new_b
            
            for _ in range(num_sims_forecast):
                eps = np.random.gumbel(loc=0, scale=1, size=3)
                U = np.array([v_m, v_s, v_b]) + eps
                choice = np.argmax(U)
                P_forecast_counts[choice] += 1
                
        P_forecast = P_forecast_counts / (total_obs * num_sims_forecast)

        fig3, ax3 = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(labels))
        ax3.bar(x, P_forecast * 100, 0.4, color='#9467bd')
        ax3.set_ylabel('Forecasted Market Share (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"{l}\n({c} SEK)" for l, c in zip(labels, [new_m, new_s, new_b])])
        ax3.set_ylim(0, 100)
        st.pyplot(fig3)

# ==========================================
# HIDDEN INSTRUCTOR TOOLS
# ==========================================
with st.expander("⚙️ Instructor Tools"):
    admin_password = st.text_input("Enter password", type="password")
    if admin_password == "admin123": 
        
        st.markdown("#### Simulation Controls")
        st.caption("Underlying true preferences of fake students: γ=-0.015, ASC_m=1.2, ASC_s=0.5")
        
        num_individuals = st.selectbox("Select number of simulated students:", [25, 50, 75, 100, 125, 150, 175, 200])
        
        if st.button(f"🧪 Inject {num_individuals} Students ({num_individuals * 4} Choices)"):
            shared_data.clear()
            price_levels = [79, 99, 119, 139, 159]
            
            for _ in range(num_individuals):
                for _ in range(4):
                    p_m = np.random.choice(price_levels)
                    p_s = np.random.choice(price_levels)
                    p_b = np.random.choice(price_levels)
                    
                    v_m = -0.015 * p_m + 1.2 
                    v_s = -0.015 * p_s + 0.5 
                    v_b = -0.015 * p_b + 0
                    
                    probs = np.exp([v_m, v_s, v_b]) / np.sum(np.exp([v_m, v_s, v_b]))
                    choice = np.random.choice([0, 1, 2], p=probs)
                    shared_data.append({'choice': choice, 'p_m': p_m, 'p_s': p_s, 'p_b': p_b})
            
            st.session_state.has_voted = True
            st.rerun()

        st.divider()
        if st.button("🗑️ Reset All Class Votes"):
            shared_data.clear()
            st.session_state.has_voted = False
            st.session_state.task_idx = 0
            st.rerun()
