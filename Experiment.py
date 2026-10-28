import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Lunch Choice App", layout="centered", page_icon="🍔")

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
        
        idx = st.session_state.task_idx 

        st.markdown(f'<img src="{img_meatballs}" style="{img_style}">', unsafe_allow_html=True)
        if st.button(f"🧆 Meatballs ({p_m} SEK)", use_container_width=True, key=f"btn_m_{idx}"):
            shared_data.append({'choice': 0, 'p_m': p_m, 'p_s': p_s, 'p_b': p_b})
            st.session_state.task_idx += 1
            if st.session_state.task_idx >= 4: st.session_state.has_voted = True
            st.rerun()

        st.markdown(f'<img src="{img_salad}" style="{img_style}">', unsafe_allow_html=True)
        if st.button(f"🥗 Fresh Salad ({p_s} SEK)", use_container_width=True, key=f"btn_s_{idx}"):
            shared_data.append({'choice': 1, 'p_m': p_m, 'p_s': p_s, 'p_b': p_b})
            st.session_state.task_idx += 1
            if st.session_state.task_idx >= 4: st.session_state.has_voted = True
            st.rerun()

        st.markdown(f'<img src="{img_stew}" style="{img_style}">', unsafe_allow_html=True)
        if st.button(f"🥘 Beef Stew ({p_b} SEK)", use_container_width=True, key=f"btn_stew_{idx}"):
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

# Global variables for parameters
est_gamma, est_asc_m, est_asc_s = 0.0, 0.0, 0.0
nl_gamma, nl_asc_m, nl_asc_s, nl_lambda = 0.0, 0.0, 0.0, 1.0

# ==========================================
# PAGE 2: MODEL ESTIMATION (MNL & NL)
# ==========================================
with tab2:
    st.header("Model Estimation")
    if total_obs == 0:
        st.warning("Waiting for data...")
    else:
        estimate_ascs = st.checkbox("Include Alternative Specific Constants (ASCs)", value=True)
        
        # --- 1. MNL ESTIMATION ---
        if estimate_ascs:
            def nll_mnl(params):
                gamma, a_m, a_s = params
                LL = 0
                for r in shared_data:
                    v = [gamma*r['p_m']+a_m, gamma*r['p_s']+a_s, gamma*r['p_b']]
                    p = np.exp(v - np.max(v)) / np.sum(np.exp(v - np.max(v)))
                    LL += np.log(p[r['choice']] + 1e-10)
                return -LL
            res_mnl = minimize(nll_mnl, x0=[-0.01, 0, 0], bounds=[(-1.0, 0.1), (None, None), (None, None)])
            est_gamma, est_asc_m, est_asc_s = res_mnl.x
        else:
            def nll_mnl(params):
                LL = 0
                for r in shared_data:
                    v = [params[0]*r['p_m'], params[0]*r['p_s'], params[0]*r['p_b']]
                    p = np.exp(v - np.max(v)) / np.sum(np.exp(v - np.max(v)))
                    LL += np.log(p[r['choice']] + 1e-10)
                return -LL
            res_mnl = minimize(nll_mnl, x0=[-0.01], bounds=[(-1.0, 0.1)])
            est_gamma, est_asc_m, est_asc_s = res_mnl.x[0], 0.0, 0.0

        # --- 2. NESTED LOGIT ESTIMATION (Meat Nest) ---
        def calc_nl_probs(gamma, a_m, a_s, lam, p_m, p_s, p_b):
            v_m, v_s, v_b = gamma*p_m + a_m, gamma*p_s + a_s, gamma*p_b
            max_meat = np.max([v_m/lam, v_b/lam])
            exp_meat = np.exp(v_m/lam - max_meat) + np.exp(v_b/lam - max_meat)
            iv_meat = max_meat + np.log(exp_meat)
            
            v_nest = [lam * iv_meat, v_s]
            p_nest = np.exp(v_nest - np.max(v_nest)) / np.sum(np.exp(v_nest - np.max(v_nest)))
            
            p_m_given = np.exp(v_m/lam - max_meat) / exp_meat
            p_b_given = np.exp(v_b/lam - max_meat) / exp_meat
            return [p_nest[0]*p_m_given, p_nest[1], p_nest[0]*p_b_given], v_m, v_s, v_b, v_nest[0], v_nest[1]

        if estimate_ascs:
            def nll_nl(params):
                g, am, as_, lam = params
                LL = 0
                for r in shared_data:
                    p_probs, _,_,_,_,_ = calc_nl_probs(g, am, as_, lam, r['p_m'], r['p_s'], r['p_b'])
                    LL += np.log(p_probs[r['choice']] + 1e-10)
                return -LL
            res_nl = minimize(nll_nl, x0=[-0.01, 0, 0, 0.5], bounds=[(-1.0, 0.1), (None, None), (None, None), (0.05, 1.0)])
            nl_gamma, nl_asc_m, nl_asc_s, nl_lambda = res_nl.x
        else:
            def nll_nl(params):
                g, lam = params
                LL = 0
                for r in shared_data:
                    p_probs, _,_,_,_,_ = calc_nl_probs(g, 0, 0, lam, r['p_m'], r['p_s'], r['p_b'])
                    LL += np.log(p_probs[r['choice']] + 1e-10)
                return -LL
            res_nl = minimize(nll_nl, x0=[-0.01, 0.5], bounds=[(-1.0, 0.1), (0.05, 1.0)])
            nl_gamma, nl_asc_m, nl_asc_s, nl_lambda = res_nl.x[0], 0.0, 0.0, res_nl.x[1]

        # Display Parameters
        st.markdown("##### 1. Multinomial Logit (MNL)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("γ (Cost)", f"{est_gamma:.4f}")
        c2.metric("ASC (Meat)", f"{est_asc_m:.2f}")
        c3.metric("ASC (Salad)", f"{est_asc_s:.2f}")
        c4.metric("λ (N/A)", "1.0")

        st.markdown("##### 2. Nested Logit (Meat Nest)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("γ (Cost)", f"{nl_gamma:.4f}")
        c2.metric("ASC (Meat)", f"{nl_asc_m:.2f}")
        c3.metric("ASC (Salad)", f"{nl_asc_s:.2f}")
        c4.metric("λ (Meat Nest)", f"{nl_lambda:.2f}")
        
        st.divider()
        st.markdown("#### Model Predictions")
        
        # Plotting Toggles
        tc1, tc2, tc3 = st.columns(3)
        show_nl_ana = tc1.checkbox("Include NL (Analytical)", value=False)
        show_mnl_mc = tc2.checkbox("Include MNL (Monte Carlo)", value=False)
        show_nl_mc = tc3.checkbox("Include NL (Monte Carlo)", value=False)
        
        num_sims = st.slider("Monte Carlo Iterations per Student", 1, 100, 1) if (show_mnl_mc or show_nl_mc) else 1
        
        P_ana_mnl, P_ana_nl = np.zeros(3), np.zeros(3)
        C_mc_mnl, C_mc_nl = np.zeros(3), np.zeros(3)
        
        np.random.seed(42) 
        
        for r in shared_data:
            # 1. MNL Analytical
            v_mnl = [est_gamma*r['p_m']+est_asc_m, est_gamma*r['p_s']+est_asc_s, est_gamma*r['p_b']]
            P_ana_mnl += np.exp(v_mnl - np.max(v_mnl)) / np.sum(np.exp(v_mnl - np.max(v_mnl)))
            
            # 2. NL Analytical
            nl_p, v_m, v_s, v_b, v_nest_meat, v_nest_veg = calc_nl_probs(nl_gamma, nl_asc_m, nl_asc_s, nl_lambda, r['p_m'], r['p_s'], r['p_b'])
            P_ana_nl += np.array(nl_p)
            
            for _ in range(num_sims):
                # 3. MNL Monte Carlo
                if show_mnl_mc:
                    eps = np.random.gumbel(0, 1, 3)
                    C_mc_mnl[np.argmax(np.array(v_mnl) + eps)] += 1
                
                # 4. NL Monte Carlo (Sequential Gumbel)
                if show_nl_mc:
                    eps_nest = np.random.gumbel(0, 1, 2)
                    U_nest = np.array([v_nest_meat, v_nest_veg]) + eps_nest
                    if np.argmax(U_nest) == 0: # Chose Meat Nest
                        eps_alt = np.random.gumbel(0, 1, 2)
                        U_alt = np.array([v_m/nl_lambda, v_b/nl_lambda]) + eps_alt
                        C_mc_nl[0 if np.argmax(U_alt) == 0 else 2] += 1
                    else:
                        C_mc_nl[1] += 1
                
        # Normalize
        P_ana_mnl /= total_obs
        P_ana_nl /= total_obs
        P_mc_mnl = C_mc_mnl / (total_obs * num_sims)
        P_mc_nl = C_mc_nl / (total_obs * num_sims)
        P_obs = np.array(counts) / total_obs

        # Dynamic Bar Plotting
        plot_data = [('Observed', P_obs, '#1f77b4'), ('MNL (Ana)', P_ana_mnl, '#2ca02c')]
        if show_nl_ana: plot_data.append(('NL (Ana)', P_ana_nl, '#9467bd'))
        if show_mnl_mc: plot_data.append(('MNL (MC)', P_mc_mnl, '#ff7f0e'))
        if show_nl_mc:  plot_data.append(('NL (MC)', P_mc_nl, '#e377c2'))

        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(labels))
        n_bars = len(plot_data)
        width = 0.8 / n_bars
        offsets = np.linspace(-width*(n_bars-1)/2, width*(n_bars-1)/2, n_bars)

        for i, (lbl, vals, col) in enumerate(plot_data):
            ax2.bar(x + offsets[i], vals * 100, width, label=lbl, color=col)
            
        ax2.set_ylabel('Market Share (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
        st.pyplot(fig2)

# ==========================================
# PAGE 3: POLICY FORECAST (IIA DEMO)
# ==========================================
with tab3:
    st.header("Policy Forecast (IIA Test)")
    st.markdown("Test an extreme price change to see how the two models route the substituting passengers. Notice who gets the lost Meatball customers!")
    
    if total_obs == 0:
        st.warning("Waiting for data...")
    else:
        new_m = st.slider("New Price Meatballs (SEK)", 50, 200, 109, 1)
        new_s = st.slider("New Price Salad (SEK)", 50, 200, 79, 1)
        new_b = st.slider("New Price Beef Stew (SEK)", 50, 200, 129, 1)

        P_fcst_mnl = np.zeros(3)
        P_fcst_nl = np.zeros(3)
        
        for r in shared_data:
            # Forecast MNL
            v_mnl = [est_gamma*new_m + est_asc_m, est_gamma*new_s + est_asc_s, est_gamma*new_b]
            P_fcst_mnl += np.exp(v_mnl - np.max(v_mnl)) / np.sum(np.exp(v_mnl - np.max(v_mnl)))
            
            # Forecast NL
            nl_p, _,_,_,_,_ = calc_nl_probs(nl_gamma, nl_asc_m, nl_asc_s, nl_lambda, new_m, new_s, new_b)
            P_fcst_nl += np.array(nl_p)
                
        P_fcst_mnl /= total_obs
        P_fcst_nl /= total_obs

        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(labels))
        width = 0.35
        
        ax3.bar(x - width/2, P_fcst_mnl * 100, width, label='MNL Forecast (Proportional)', color='#2ca02c')
        ax3.bar(x + width/2, P_fcst_nl * 100, width, label='NL Forecast (Correlated)', color='#9467bd')
        
        ax3.set_ylabel('Forecasted Market Share (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"{l}\n({c} SEK)" for l, c in zip(labels, [new_m, new_s, new_b])])
        ax3.set_ylim(0, 100)
        ax3.legend()
        st.pyplot(fig3)

# ==========================================
# HIDDEN INSTRUCTOR TOOLS (Adjustable DGP)
# ==========================================
with st.expander("⚙️ Instructor Tools"):
    admin_password = st.text_input("Enter password", type="password")
    if admin_password == "admin123": 
        
        st.markdown("#### Synthetic Population Injector")
        st.caption("True DGP: γ=-0.015, ASC_m=0.3, ASC_s= -0.1")
        
        num_individuals = st.selectbox("Select number of simulated students:", [25, 50, 75, 100, 125, 150, 175, 200])
        
        # Adjustable Lambda for Data Generation
        st.markdown("**Nesting Parameter (λ)**")
        st.caption("Set to 1.0 to generate pure MNL data. Set < 1.0 to inject 'Meat-Lover' correlation.")
        true_lambda = st.slider("True λ (Meatballs & Stew)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        
        if st.button(f"🧪 Inject {num_individuals} Students (True λ={true_lambda})"):
            shared_data.clear()
            price_levels = [79, 99, 119, 139, 159]
            
            true_gamma = -0.015
            true_asc_m = 0.3
            true_asc_s = -0.1
            
            for _ in range(num_individuals):
                for _ in range(4):
                    p_m = np.random.choice(price_levels)
                    p_s = np.random.choice(price_levels)
                    p_b = np.random.choice(price_levels)
                    
                    v_m = true_gamma * p_m + true_asc_m 
                    v_s = true_gamma * p_s + true_asc_s 
                    v_b = true_gamma * p_b + 0
                    
                    # Generate using Nested Logit logic
                    exp_m_lam = np.exp(v_m / true_lambda)
                    exp_b_lam = np.exp(v_b / true_lambda)
                    
                    iv_meat = np.log(exp_m_lam + exp_b_lam)
                    v_nest_meat = true_lambda * iv_meat
                    v_nest_veg = v_s
                    
                    p_nest_meat = np.exp(v_nest_meat) / (np.exp(v_nest_meat) + np.exp(v_nest_veg))
                    p_nest_veg  = np.exp(v_nest_veg) / (np.exp(v_nest_meat) + np.exp(v_nest_veg))
                    
                    p_m_given_meat = exp_m_lam / (exp_m_lam + exp_b_lam)
                    p_b_given_meat = exp_b_lam / (exp_m_lam + exp_b_lam)
                    
                    p_m_final = p_nest_meat * p_m_given_meat
                    p_b_final = p_nest_meat * p_b_given_meat
                    p_s_final = p_nest_veg
                    
                    choice = np.random.choice([0, 1, 2], p=[p_m_final, p_s_final, p_b_final])
                    shared_data.append({'choice': choice, 'p_m': p_m, 'p_s': p_s, 'p_b': p_b})
            
            st.session_state.has_voted = True
            st.session_state.task_idx = 4
            st.rerun()

        st.divider()
        if st.button("🗑️ Reset All Class Votes"):
            shared_data.clear()
            st.session_state.has_voted = False
            st.session_state.task_idx = 0
            st.rerun()
