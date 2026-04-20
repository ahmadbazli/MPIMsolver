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
            k1 = get_f(t, curr_y)
            k2 = get_f(t + h/2, curr_y + h/2 * k1)
            k3 = get_f(t + h/2, curr_y + h/2 * k2)
            k4 = get_f(t + h, curr_y + h * k3)
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

# --- 2. STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'run_sim' not in st.session_state:
    st.session_state.run_sim = False

def move_page(page_num):
    st.session_state.page = page_num

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="MSPI Solver", layout="wide")
st.title("🚀 ODE Solver: MSPI vs SPI vs RK4")

# Progress Bar ringkas
st.progress(st.session_state.page / 3)

# --- PAGE 1: CONFIGURATION ---
if st.session_state.page == 1:
    st.header("1️⃣ Sistem Konfigurasi")
    col1, col2 = st.columns(2)
    with col1:
        n_comp = st.number_input("Bilangan Kompartmen", 1, 10, 3, key="n_comp_val")
        t_max = st.number_input("Tempoh Masa (T)", 1, 100, 20, key="t_max_val")
    with col2:
        k_stages = st.slider("Bilangan Sub-selang (k)", 5, 200, 50, key="k_val")
        n_iters = st.slider("Bilangan Iterasi Picard (n)", 1, 20, 5, key="n_val")
    
    st.write("---")
    st.button("Next: Equation Input ➡️", on_click=move_page, args=(2,), use_container_width=True)

# --- PAGE 2: EQUATION INPUT ---
elif st.session_state.page == 2:
    st.header("2️⃣ Input Persamaan & Nilai Awal")
    st.info("Gunakan y, y... untuk pembolehubah. Contoh: 1 - y**2")
    
    u_eqs = []
    u_inits = []
    
    for i in range(int(st.session_state.n_comp_val)):
        st.subheader(f"Kompartmen y[{i}]")
        c1, c2 = st.columns()
        eq = c1.text_input(f"dy[{i}]/dt", value="1 - y**2" if i==0 else "0", key=f"eq_in_{i}")
        init = c2.number_input(f"y[{i}] Initial Value", value=-0.5 if i==0 else 0.0, key=f"init_in_{i}")
        u_eqs.append(eq)
        u_inits.append(init)
    
    st.session_state.u_eqs = u_eqs
    st.session_state.u_inits = u_inits

    st.write("---")
    nav_col = st.columns(2)
    nav_col.button("⬅️ Back to Config", on_click=move_page, args=(1,), use_container_width=True)
    if nav_col.button("🔥 Run Simulation & See Results ➡️", use_container_width=True):
        st.session_state.run_sim = True
        st.session_state.page = 3
        st.rerun()

# --- PAGE 3: RESULTS & ANALYSIS ---
elif st.session_state.page == 3:
    st.header("3️⃣ Hasil Analisis & Discussion")
    
    if st.session_state.run_sim:
        t_pts = np.linspace(0, st.session_state.t_max_val, st.session_state.k_val + 1)
        
        # Pengiraan
        res_rk4 = rk4_solver(st.session_state.u_eqs, st.session_state.u_inits, t_pts)
        res_mspi = picard_solvers(st.session_state.u_eqs, st.session_state.u_inits, t_pts, st.session_state.n_val, True)
        res_spi = picard_solvers(st.session_state.u_eqs, st.session_state.u_inits, t_pts, st.session_state.n_val, False)
        
        if res_rk4 is not None:
            # Grafik
            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(len(st.session_state.u_inits)):
                ax.plot(t_pts, res_rk4[:, i], 'k-', alpha=0.3, label=f"RK4 Bench y[{i}]")
                ax.plot(t_pts, res_spi[:, i], '--', label=f"SPI y[{i}]")
                ax.plot(t_pts, res_mspi[:, i], '-o', markersize=3, label=f"MSPI y[{i}]")
            ax.set_xlabel("Masa (t)")
            ax.set_ylabel("Nilai")
            ax.legend()
            st.pyplot(fig)
            
            # Metrics
            err_mspi = np.mean(np.abs(res_rk4 - res_mspi))
            err_spi = np.mean(np.abs(res_rk4 - res_spi))
            
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Ralat MSPI (vs RK4)", f"{err_mspi:.6e}")
            m_col2.metric("Ralat SPI (vs RK4)", f"{err_spi:.6e}")
            
            st.success(f"Analisis: MSPI menunjukkan ralat {err_mspi:.6e} manakala SPI menunjukkan {err_spi:.6e}.")
        else:
            st.error("Ralat pengiraan. Sila semak format persamaan anda di Page 2.")
    
    st.write("---")
    st.button("⬅️ Back to Equations", on_click=move_page, args=(2,), use_container_width=True)
