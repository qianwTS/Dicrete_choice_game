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
    return [] # Will store: {'choice': 0, 'veg': 1, 'swe': 0, 'p_m': 109, 'p_s': 79, 'p_b': 129}

shared_data = get_shared_data()

# --- 2. LOCAL BROWSER MEMORY (SP EXPERIMENT SETUP) ---
if 'has_voted' not in st.session_state:
    st.session_state.has_voted = False
    st.session_state.task_idx = 0 # Tracks which of the 3 questions they are on
    
    # Generate 3 randomized price scenarios for this specific student
    st.session_state.scenarios = [
        {'m': int(np.random.choice([89, 109, 129])),
         's': int(np.random.choice([59, 79, 99])),
         'b': int(np.random.choice([109, 129, 149]))}
        for _ in range(3)
    ]

labels = ['Meatballs', 'Salad', 'Beef Stew']
total_observations = len(shared_data)
counts = [sum(1 for d in shared_data if d['choice'] == i) for i in range(3)] if total_observations > 0 else [0,0,0]

tab1, tab2, tab3, tab4 = st.tabs(["1. Survey", "2. Basic MNL", "3. Advanced MNL", "4. Policy"])

# ==========================================
# PAGE 1: REPEATED SP EXPERIMENT
# ==========================================
with tab1:
    if st.session_state.has_voted:
        st.success("✅ All 3 scenarios completed! Watch the live data update below.")
    else:
        st.markdown(f"#### Step 1: About You")
        col_v, col_s = st.columns(2)
        with col_v:
            # Lock the demographics once they start clicking so they don't change it mid-survey
            ans_veg = st.radio("🌱 Vegetarian?", ["No", "Yes"], disabled=st.session_state.task_idx > 0)
            is_veg = 1 if ans_veg == "Yes" else 0
        with col_s:
            ans_swe = st.radio("🇸🇪 Swedish?", ["No", "Yes"], disabled=st.session_state.task_idx > 0)
            is_swe = 1 if ans_swe == "Yes" else 0
            
        st.markdown(f"#### Step 2: Scenario {st.session_state.task_idx + 1} of 3")
        st.caption("Prices have changed! What is your choice now?")

        # Fetch prices for the current task
        p_m = st.session_state.scenarios[st.session_state.task_idx]['m']
        p_s = st.session_state.scenarios[st.session_state.task_idx]['s']
        p_b = st.session_state.scenarios[st.session_state.task_idx]['b']

        img_meatballs = "https://images.unsplash.com/photo-1529042410759-befb1204b468?auto=format&fit=crop&w=800&h=250&q=80"
        img_salad = "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?auto=format&fit=crop&w=800&h=250&q=80"
        img_stew = "https://images.unsplash.com/photo-1534939561126-855b8675edd7?auto=format&fit=crop&w=800&h=250&q=80"
        img_style = "width: 100%; height: 90px; object-fit: cover; border-radius: 8px; margin-bottom: 5px;"

        st.markdown(f'<img src="{img_meatballs}" style="{img_style}">', unsafe_allow_html=True)
        if st.button(f"🧆 Meatballs ({p_m} SEK)", use_container_width=True, key="btn_m"):
            shared_data.append({'choice': 0, 'veg': is_veg, 'swe': is_swe, 'p_m': p_m, 'p_s': p_s, 'p_b': p_b})
            st.session_state.task_idx += 1
            if st.session_state.task_idx >= 3: st.session_state.has_voted = True
            st.rerun()

        st.markdown(f'<img src="{img_salad}" style="{img_style}">', unsafe_allow_html=True)
        if st.button(f"🥗 Fresh Salad ({p_s} SEK)", use_container_width=True, key="btn_s"):
            shared_data.append({'choice': 1, 'veg': is_veg, 'swe': is_swe, 'p_m': p_m, 'p_s': p_s, 'p_b': p_b})
            st.session_state.task_idx += 1
            if st.session_state.task_idx >= 3: st.session_state.has_voted = True
            st.rerun()

        st.markdown(f'<img src="{img_stew}" style="{img_style}">', unsafe_allow_html=True)
        if st.button(f"🥘 Beef Stew ({p_b} SEK)", use_container_width=True, key="btn_stew"):
            shared_data.append({'choice': 2, 'veg': is_veg, 'swe': is_swe, 'p_m': p_m, 'p_s': p_s, 'p_b': p_b})
            st.session_state.task_idx += 1
            if st.session_state.task_idx >= 3: st.session_state.has_voted = True
            st.rerun()

    st.divider()
    if total_observations > 0:
        fig1, ax1 = plt.subplots(figsize=(6, 2.5))
        bars = ax1.bar(labels, counts, color=['#8c564b', '#2ca02c', '#d62728'])
        ax1.set_ylabel('Choices Made')
        ax1.set_title(f'Live SP Data (Total Choices: {total_observations})')
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        st.pyplot(fig1)

# ==========================================
# SIMULTANEOUS ESTIMATION (Runs for Tabs 2, 3, 4)
# ==========================================
if total_observations > 0:
    # MODEL 1: Cost + ASCs simultaneously identified!
    def nll_basic(params):
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
    
    res1 = minimize(nll_basic, x0=[-0.01, 0, 0], bounds=[(-1.0, 0.1), (None, None), (None, None)])
    g1, asc_m1, asc_s1 = res1.x

    # MODEL 2: Cost + ASCs + Dummies
    def nll_adv(params):
        gamma, a_m, a_s, b_v, b_sw = params
        LL = 0
        for r in shared_data:
            v_m = gamma * r['p_m'] + a_m + b_sw * r['swe']
            v_s = gamma * r['p_s'] + a_s + b_v * r['veg']
            v_b = gamma * r['p_b']
            exp_v = np.exp([v_m, v_s, v_b] - np.max([v_m, v_s, v_b]))
            p = exp_v / np.sum(exp_v)
            LL += np.log(p[r['choice']] + 1e-10)
        return -LL
    
    res2 = minimize(nll_adv, x0=[-0.01, 0, 0, 0, 0], bounds=[(-1.0, 0.1), (None, None), (None, None), (None, None), (None, None)])
    g2, asc_m2, asc_s2, b_veg, b_swe = res2.x

# ==========================================
# PAGE 2: BASIC SP MODEL (Cost + ASCs)
# ==========================================
with tab2:
    st.header("Basic SP Model")
    if total_observations == 0:
        st.warning("Waiting for data...")
    else:
        st.markdown("*Because prices varied, we can now mathematically estimate Cost and ASCs simultaneously!*")
        c1, c2, c3 = st.columns(3)
        c1.metric("γ (Cost)", f"{g1:.4f}")
        c2.metric("ASC (Meatballs)", f"{asc_m1:.2f}")
        c3.metric("ASC (Salad)", f"{asc_s1:.2f}")
        
        # Calculate Average Predicted Probabilities over the sample
        P_pred1 = np.zeros(3)
        for r in shared_data:
            v_m = g1 * r['p_m'] + asc_m1
            v_s = g1 * r['p_s'] + asc_s1
            v_b = g1 * r['p_b']
            exp_v = np.exp([v_m, v_s, v_b] - np.max([v_m, v_s, v_b]))
            P_pred1 += exp_v / np.sum(exp_v)
        P_pred1 /= total_observations
        P_obs = np.array(counts) / total_observations

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        x = np.arange(len(labels))
        ax2.bar(x - 0.15, P_obs * 100, 0.3, label='Observed Shares', color='#1f77b4')
        ax2.bar(x + 0.15, P_pred1 * 100, 0.3, label='Predicted (Model 1)', color='#ff7f0e')
        ax2.set_ylabel('Market Share (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()
        st.pyplot(fig2)

# ==========================================
# PAGE 3: ADVANCED SP MODEL (Demographics)
# ==========================================
with tab3:
    st.header("Advanced SP Model")
    if total_observations == 0:
        st.warning("Waiting for data...")
    else:
        st.markdown("Adding individual demographics to explain *why* people have those ASC baseline preferences.")
        
        c1, c2 = st.columns(2)
        c1.metric("γ (Cost)", f"{g2:.4f}")
        c1.metric("ASC (Meatballs)", f"{asc_m2:.2f}")
        c1.metric("ASC (Salad)", f"{asc_s2:.2f}")
        c2.metric("β (Vegetarian ➔ Salad)", f"{b_veg:.2f}")
        c2.metric("β (Swedish ➔ Meatballs)", f"{b_swe:.2f}")

        P_pred2 = np.zeros(3)
        for r in shared_data:
            v_m = g2 * r['p_m'] + asc_m2 + b_swe * r['swe']
            v_s = g2 * r['p_s'] + asc_s2 + b_veg * r['veg']
            v_b = g2 * r['p_b']
            exp_v = np.exp([v_m, v_s, v_b] - np.max([v_m, v_s, v_b]))
            P_pred2 += exp_v / np.sum(exp_v)
        P_pred2 /= total_observations

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        x = np.arange(len(labels))
        ax3.bar(x - 0.15, P_obs * 100, 0.3, label='Observed Shares', color='#1f77b4')
        ax3.bar(x + 0.15, P_pred2 * 100, 0.3, label='Predicted (Advanced)', color='#2ca02c')
        ax3.set_ylabel('Market Share (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.legend()
        st.pyplot(fig3)

# ==========================================
# PAGE 4: POLICY FORECAST
# ==========================================
with tab4:
    st.header("Policy Forecast")
    if total_observations == 0:
        st.warning("Waiting for data...")
    else:
        st.markdown("Sample Enumeration using the Advanced Model.")
        new_m = st.slider("New Base Price Meatballs (SEK)", 50, 200, 109, 1)
        new_s = st.slider("New Base Price Salad (SEK)", 50, 200, 79, 1)
        new_b = st.slider("New Base Price Beef Stew (SEK)", 50, 200, 129, 1)

        P_forecast = np.zeros(3)
        for r in shared_data:
            v_m = g2 * new_m + asc_m2 + b_swe * r['swe']
            v_s = g2 * new_s + asc_s2 + b_veg * r['veg']
            v_b = g2 * new_b
            exp_v = np.exp([v_m, v_s, v_b] - np.max([v_m, v_s, v_b]))
            P_forecast += exp_v / np.sum(exp_v)
        P_forecast /= total_observations

        fig4, ax4 = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(labels))
        ax4.bar(x, P_forecast * 100, 0.4, color='#9467bd')
        ax4.set_ylabel('Forecasted Market Share (%)')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"{l}\n({c} SEK)" for l, c in zip(labels, [new_m, new_s, new_b])])
        ax4.set_ylim(0, 100)
        st.pyplot(fig4)

# ==========================================
# HIDDEN INSTRUCTOR TOOLS
# ==========================================
with st.expander("⚙️ Instructor Tools"):
    admin_password = st.text_input("Enter password", type="password")
    if admin_password == "admin123": 
        if st.button("🧪 Inject 30 Test SP Choices (10 Students)"):
            import random
            shared_data.clear()
            # Simulate 10 students making 3 choices each
            for _ in range(10):
                is_swe = np.random.choice([0, 1])
                is_veg = np.random.choice([0, 1]) if not is_swe else 0 # Swedes less likely veg here for data shape
                for _ in range(3):
                    p_m = np.random.choice([89, 109, 129])
                    p_s = np.random.choice([59, 79, 99])
                    p_b = np.random.choice([109, 129, 149])
                    
                    # Generative behavior based on true parameters to make charts look right
                    v_m = -0.015 * p_m + 1.2 + (1.5 * is_swe)
                    v_s = -0.015 * p_s + 0.5 + (2.5 * is_veg)
                    v_b = -0.015 * p_b + 0
                    
                    probs = np.exp([v_m, v_s, v_b]) / np.sum(np.exp([v_m, v_s, v_b]))
                    choice = np.random.choice([0, 1, 2], p=probs)
                    shared_data.append({'choice': choice, 'veg': is_veg, 'swe': is_swe, 'p_m': p_m, 'p_s': p_s, 'p_b': p_b})
            
            st.session_state.has_voted = True
            st.rerun()

        if st.button("🗑️ Reset All Class Votes"):
            shared_data.clear()
            st.session_state.has_voted = False
            st.session_state.task_idx = 0
            st.rerun()
