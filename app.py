import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. SOLVER ENGINES ---
def rk4_solver(eq_strings, y0, t_points):
    try:
        num_vars = len(y0)
        y_res = [np.array(y0, dtype=float)]
        for i in range(len(t_points)-1):
            h = t_points[i+1] - t_points[i]
            t = t_points[i]
            curr_y = y_res[-1]
            def get_f(t_val, y_val):
                ctx = {"y": y_val, "t": t_val, "np": np}
                return np.array([eval(s.replace('^', '**'), {"__builtins__": None}, ctx) for s in eq_strings], dtype=float)
            k1 = get_f(t, curr_y); k2 = get_f(t + h/2, curr_y + h/2 * k1)
            k3 = get_f(t + h/2, curr_y + h/2 * k2); k4 = get_f(t + h, curr_y + h * k3)
            y_next = curr_y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            y_res.append(y_next)
        return np.array(y_res)
    except: return None

def picard_solvers(eq_strings, y0, t_points, num_iter, is_modified=True):
    try:
        num_vars = len(y0)
        y_history = [np.array(y0, dtype=float)]
        current_y_start = np.array(y0, dtype=float)
        for k in range(1, len(t_points)):
            h = t_points[k] - t_points[k-1]
            yn_k = current_y_start.copy()
            for n in range(num_iter):
                new_yn = []
                ctx = {"y": yn_k if is_modified else current_y_start, "t": t_points[k-1], "np": np}
                for i in range(num_vars):
                    f_val = eval(eq_strings[i].replace('^', '**'), {"__builtins__": None}, ctx)
                    new_yn.append(current_y_start[i] + f_val * h)
                yn_k = np.array(new_yn, dtype=float)
            current_y_start = yn_k
            y_history.append(current_y_start)
        return np.array(y_history)
    except: return None

# --- 2. NAVIGATION STATE ---
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'run_sim' not in st.session_state:
    st.session_state.run_sim = False

def next_step(): st.session_state.step += 1
def prev_step(): st.session_state.step -= 1

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="ODE Solver Pro", layout="wide")
st.title("🚀 ODE Solver: MSPI vs SPI vs RK4")

# Tabs diselaraskan dengan session_state step
tabs = st.tabs(["1️⃣ Configuration", "2️⃣ Equation Input", "3️⃣ Results & Analysis"])

# --- TAB 1: CONFIGURATION ---
with tabs:
    st.header("Sistem Konfigurasi")
    col_a, col_b = st.columns(2) # FIXED: Spec diletakkan
    with col_a:
        n_comp = st.number_input("Bilangan Kompartmen", 1, 10, 3, key="n_comp")
        t_max = st.number_input("Tempoh Masa (T)", 1, 100, 20, key="t_max")
    with col_b:
        k_stages = st.slider("Bilangan Sub-selang (k)", 5, 200, 50, key="k_val")
        n_iters = st.slider("Bilangan Iterasi Picard (n)", 1, 20, 5, key="n_val")
    
    st.write("---")
    st.button("Next: Equation Input ➡️", on_click=next_step, use_container_width=True)

# --- TAB 2: EQUATION INPUT ---
with tabs:
    st.header("Input Persamaan & Nilai Awal")
    u_eqs = []
    u_inits = []
    
    for i in range(int(st.session_state.n_comp)):
        st.subheader(f"Kompartmen y[{i}]")
        c1, c2 = st.columns() # FIXED: Spec diletakkan
        eq = c1.text_input(f"dy[{i}]/dt", value="1 - y**2" if i==0 else "0", key=f"eq{i}")
        init = c2.number_input(f"y[{i}] Initial Value", value=-0.5 if i==0 else 0.0, key=f"init{i}")
        u_eqs.append(eq)
        u_inits.append(init)
    
    st.write("---")
    col_nav = st.columns(2)
    col_nav.button("⬅️ Back", on_click=prev_step, use_container_width=True)
    if col_nav.button("🔥 Run & Next Result ➡️", use_container_width=True):
        st.session_state.run_sim = True
        st.session_state.u_eqs = u_eqs
        st.session_state.u_inits = u_inits
        st.session_state.step = 2
        st.rerun()

# --- TAB 3: RESULTS & ANALYSIS ---
with tabs:
    if st.session_state.run_sim:
        t_pts = np.linspace(0, st.session_state.t_max, st.session_state.k_val + 1)
        res_rk4 = rk4_solver(st.session_state.u_eqs, st.session_state.u_inits, t_pts)
        res_mspi = picard_solvers(st.session_state.u_eqs, st.session_state.u_inits, t_pts, st.session_state.n_val, True)
        res_spi = picard_solvers(st.session_state.u_eqs, st.session_state.u_inits, t_pts, st.session_state.n_val, False)
        
        if res_rk4 is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(len(st.session_state.u_inits)):
                ax.plot(t_pts, res_rk4[:, i], 'k-', alpha=0.3, label=f"RK4 y[{i}]")
                ax.plot(t_pts, res_spi[:, i], '--', label=f"SPI y[{i}]")
                ax.plot(t_pts, res_mspi[:, i], '-o', markersize=3, label=f"MSPI y[{i}]")
            ax.set_title("Analisis Perbandingan Numerikal")
            ax.legend()
            st.pyplot(fig)
            
            err_mspi = np.mean(np.abs(res_rk4 - res_mspi))
            err_spi = np.mean(np.abs(res_rk4 - res_spi))
            
            c_err = st.columns(2) # FIXED: Spec diletakkan
            c_err.metric("Ralat MSPI", f"{err_mspi:.6e}")
            c_err.metric("Ralat SPI", f"{err_spi:.6e}")
    else:
        st.warning("Sila lengkapkan Tab 2 dan tekan Run dahulu.")
