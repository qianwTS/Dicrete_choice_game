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
# Instead of aggregate votes, we now store individual student rows!
@st.cache_resource
def get_shared_data():
    return [] # List of dicts: [{'choice': 0, 'veg': 1, 'swe': 0}, ...]

shared_data = get_shared_data()

# --- 2. LOCAL BROWSER MEMORY ---
if 'has_voted' not in st.session_state:
    st.session_state.has_voted = False

base_costs = np.array([109, 79, 129])
labels = ['Meatballs', 'Salad', 'Beef Stew']

tab1, tab2, tab3, tab4 = st.tabs(["1. Survey", "2. Basic Model", "3. Advanced Model", "4. Policy Forecast"])

total_students = len(shared_data)

# Calculate aggregate counts for charts
counts = [sum(1 for d in shared_data if d['choice'] == i) for i in range(3)] if total_students > 0 else [0,0,0]

# ==========================================
# PAGE 1: MAKING CHOICES
# ==========================================
with tab1:
    if st.session_state.has_voted:
        st.success("✅ Vote recorded! Watch the live data update below.")
    else:
        st.markdown("#### 1. About You")
        col_v, col_s = st.columns(2)
        with col_v:
            is_veg = st.checkbox("🌱 I am Vegetarian")
        with col_s:
            is_swe = st.checkbox("🇸🇪 I am Swedish")
            
        st.markdown("#### 2. Choose your lunch")

    img_meatballs = "https://images.unsplash.com/photo-1529042410759-befb1204b468?auto=format&fit=crop&w=800&h=250&q=80"
    img_salad = "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?auto=format&fit=crop&w=800&h=250&q=80"
    img_stew = "https://images.unsplash.com/photo-1534939561126-855b8675edd7?auto=format&fit=crop&w=800&h=250&q=80"
    img_style = "width: 100%; height: 90px; object-fit: cover; border-radius: 8px; margin-bottom: 5px;"

    # --- ROW 1: MEATBALLS ---
    st.markdown(f'<img src="{img_meatballs}" style="{img_style}">', unsafe_allow_html=True)
    if st.button("🧆 Meatballs (109 SEK)", disabled=st.session_state.has_voted, use_container_width=True, key="btn_m"):
        shared_data.append({'choice': 0, 'veg': int(is_veg), 'swe': int(is_swe)})
        st.session_state.has_voted = True
        st.rerun()

    # --- ROW 2: SALAD ---
    st.markdown(f'<img src="{img_salad}" style="{img_style}">', unsafe_allow_html=True)
    if st.button("🥗 Fresh Salad (79 SEK)", disabled=st.session_state.has_voted, use_container_width=True, key="btn_s"):
        shared_data.append({'choice': 1, 'veg': int(is_veg), 'swe': int(is_swe)})
        st.session_state.has_voted = True
        st.rerun()

    # --- ROW 3: BEEF STEW ---
    st.markdown(f'<img src="{img_stew}" style="{img_style}">', unsafe_allow_html=True)
    if st.button("🥘 Beef Stew (129 SEK)", disabled=st.session_state.has_voted, use_container_width=True, key="btn_stew"):
        shared_data.append({'choice': 2, 'veg': int(is_veg), 'swe': int(is_swe)})
        st.session_state.has_voted = True
        st.rerun()

    if total_students > 0:
        fig1, ax1 = plt.subplots(figsize=(6, 2.5))
        bars = ax1.bar(labels, counts, color=['#8c564b', '#2ca02c', '#d62728'])
        ax1.set_ylabel('Votes')
        ax1.set_title(f'Live Class Votes (Total: {total_students})')
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        st.pyplot(fig1)

# ==========================================
# GLOBAL ESTIMATION (Runs silently for Tabs 2, 3, and 4)
# ==========================================
est_gamma = 0.0
asc_m, asc_s, b_veg, b_swe = 0.0, 0.0, 0.0, 0.0

if total_students > 0:
    # 1. Model 1 (Cost Only)
    def nll_m1(gamma):
        V = gamma[0] * base_costs
        exp_V = np.exp(V - np.max(V))
        P = exp_V / np.sum(exp_V)
        return -np.sum(np.array(counts) * np.log(P + 1e-10))
    
    res1 = minimize(nll_m1, x0=[-0.01], bounds=[(-1.0, 0.1)])
    est_gamma = res1.x[0]

    # 2. Model 2 (ASCs & Dummies, holding gamma fixed)
    def nll_m2(params):
        a_m, a_s, b_v, b_sw = params
        LL = 0
        for row in shared_data:
            v_m = est_gamma * 109 + a_m + b_sw * row['swe']
            v_s = est_gamma * 79 + a_s + b_v * row['veg']
            v_b = est_gamma * 129 # Beef Stew is Reference (ASC=0)
            
            exp_v = np.exp([v_m, v_s, v_b] - np.max([v_m, v_s, v_b]))
            p = exp_v / np.sum(exp_v)
            LL += np.log(p[row['choice']] + 1e-10)
        return -LL
    
    res2 = minimize(nll_m2, x0=[0, 0, 0, 0], method='L-BFGS-B')
    asc_m, asc_s, b_veg, b_swe = res2.x

# ==========================================
# PAGE 2: BASIC MODEL (Cost Only)
# ==========================================
with tab2:
    st.header("Basic Model (Cost Only)")
    if total_students == 0:
        st.warning("Waiting for class data...")
    else:
        st.success(f"**Estimated Cost Parameter (γ): {est_gamma:.4f}**")
        st.markdown("*Notice how the model overpredicts the cheapest option (Salad) because it only knows about price!*")
        
        V_est1 = est_gamma * base_costs
        P_pred1 = np.exp(V_est1) / np.sum(np.exp(V_est1))
        P_obs = np.array(counts) / total_students

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        x = np.arange(len(labels))
        ax2.bar(x - 0.15, P_obs * 100, 0.3, label='Observed (Real Votes)', color='#1f77b4')
        ax2.bar(x + 0.15, P_pred1 * 100, 0.3, label='Predicted (Basic MNL)', color='#ff7f0e')
        ax2.set_ylabel('Market Share (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()
        st.pyplot(fig2)

# ==========================================
# PAGE 3: ADVANCED MODEL (ASCs & Dummies)
# ==========================================
with tab3:
    st.header("Advanced Micro-Model")
    if total_students == 0:
        st.warning("Waiting for class data...")
    else:
        st.markdown("We lock the Cost parameter from Tab 2, and now estimate inherent preferences (ASCs) and demographics!")
        
        colA, colB = st.columns(2)
        colA.metric("ASC (Meatballs)", f"{asc_m:.2f}")
        colA.metric("ASC (Salad)", f"{asc_s:.2f}")
        colB.metric("β (Vegetarian ➔ Salad)", f"{b_veg:.2f}")
        colB.metric("β (Swedish ➔ Meatballs)", f"{b_swe:.2f}")

        # Calculate Predicted Probabilities using SAMPLE ENUMERATION!
        P_pred2 = np.zeros(3)
        for row in shared_data:
            v_m = est_gamma * 109 + asc_m + b_swe * row['swe']
            v_s = est_gamma * 79 + asc_s + b_veg * row['veg']
            v_b = est_gamma * 129
            exp_v = np.exp([v_m, v_s, v_b] - np.max([v_m, v_s, v_b]))
            P_pred2 += exp_v / np.sum(exp_v)
        P_pred2 /= total_students # Average across population

        st.markdown("### Model Fit: The ASC Effect")
        st.markdown("*Notice how adding ASCs forces the predicted shares to perfectly match reality!*")
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        x = np.arange(len(labels))
        ax3.bar(x - 0.15, P_obs * 100, 0.3, label='Observed', color='#1f77b4')
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
    if total_students == 0:
        st.warning("Waiting for class data...")
    else:
        st.markdown("Forecast using our advanced Sample Enumeration model:")
        new_m = st.slider("New Price for Meatballs (SEK)", 50, 200, 109, 1)
        new_s = st.slider("New Price for Salad (SEK)", 50, 200, 79, 1)
        new_stew = st.slider("New Price for Beef Stew (SEK)", 50, 200, 129, 1)

        # Forecasting using Sample Enumeration over the collected microdata
        P_forecast = np.zeros(3)
        for row in shared_data:
            v_m = est_gamma * new_m + asc_m + b_swe * row['swe']
            v_s = est_gamma * new_s + asc_s + b_veg * row['veg']
            v_b = est_gamma * new_stew
            exp_v = np.exp([v_m, v_s, v_b] - np.max([v_m, v_s, v_b]))
            P_forecast += exp_v / np.sum(exp_v)
        P_forecast /= total_students

        fig4, ax4 = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(labels))
        ax4.bar(x, P_forecast * 100, 0.4, color='#9467bd')
        ax4.set_ylabel('Forecasted Market Share (%)')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"{l}\n({c} SEK)" for l, c in zip(labels, [new_m, new_s, new_stew])])
        ax4.set_ylim(0, 100)
        st.pyplot(fig4)

# ==========================================
# HIDDEN INSTRUCTOR TOOLS
# ==========================================
with st.expander("⚙️ Instructor Tools"):
    admin_password = st.text_input("Enter password", type="password")
    if admin_password == "admin123": 
        if st.button("🧪 Inject 25 Test Votes"):
            # Creates realistic dummy microdata: 
            # Meatballs: 10 Swedes, 2 others. Salad: 7 Vegetarians, 1 other. Stew: 5 mixed.
            test_data = [{'choice': 0, 'veg': 0, 'swe': 1} for _ in range(10)] + \
                        [{'choice': 0, 'veg': 0, 'swe': 0} for _ in range(2)] + \
                        [{'choice': 1, 'veg': 1, 'swe': 1} for _ in range(2)] + \
                        [{'choice': 1, 'veg': 1, 'swe': 0} for _ in range(5)] + \
                        [{'choice': 1, 'veg': 0, 'swe': 0} for _ in range(1)] + \
                        [{'choice': 2, 'veg': 0, 'swe': 1} for _ in range(2)] + \
                        [{'choice': 2, 'veg': 0, 'swe': 0} for _ in range(3)]
            shared_data.clear()
            shared_data.extend(test_data)
            st.session_state.has_voted = True
            st.rerun()

        if st.button("🗑️ Reset All Class Votes"):
            shared_data.clear()
            st.session_state.has_voted = False
            st.rerun()
