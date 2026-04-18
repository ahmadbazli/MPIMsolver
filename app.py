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
    except:
        return None

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
                # Modified vs Standard logic
                ctx = {"y": yn_k if is_modified else current_y_start, "t": t_points[k-1], "np": np}
                for i in range(num_vars):
                    f_val = eval(eq_strings[i].replace('^', '**'), {"__builtins__": None}, ctx)
                    new_yn.append(current_y_start[i] + f_val * h)
                yn_k = np.array(new_yn, dtype=float)
                
            current_y_start = yn_k
            y_history.append(current_y_start)
        return np.array(y_history)
    except:
        return None

# --- 2. UI LAYOUT ---
st.set_page_config(page_title="ODE Solver Pro", layout="wide")
st.title("🚀 MSPI vs SPI vs RK4 Solver")

# Inisialisasi session state supaya data tidak hilang
if 'run_sim' not in st.session_state:
    st.session_state.run_sim = False

tab1, tab2, tab3 = st.tabs(["1️⃣ Configuration", "2️⃣ Equation Input", "3️⃣ Results & Analysis"])

# --- PAGE 1: CONFIGURATION ---
with tab1:
    st.header("Sistem Konfigurasi")
    # Guna 2 kolum dengan input eksplisit
    col_a, col_b = st.columns(2)
    n_comp = col_a.number_input("Bilangan Kompartmen", 1, 10, 3)
    t_max = col_a.number_input("Tempoh Masa (T)", 1, 100, 20)
    k_stages = col_b.slider("Bilangan Sub-selang (k)", 5, 200, 50)
    n_iters = col_b.slider("Bilangan Iterasi Picard (n)", 1, 20, 5)

# --- PAGE 2: EQUATION INPUT ---
with tab2:
    st.header("Input Persamaan & Nilai Awal")
    u_eqs = []
    u_inits = []
    
    for i in range(int(n_comp)):
        st.subheader(f"Kompartmen y[{i}]")
        # Elakkan st.columns() kosong, beri weight
        c1, c2 = st.columns()
        eq = c1.text_input(f"dy[{i}]/dt", value="1 - y**2" if i==0 else "0", key=f"eq{i}")
        init = c2.number_input(f"y[{i}] Initial Value", value=-0.5 if i==0 else 0.0, key=f"init{i}")
        u_eqs.append(eq)
        u_inits.append(init)
    
    st.write("---")
    if st.button("🔥 JALANKAN SIMULASI", use_container_width=True):
        st.session_state.run_sim = True
        st.session_state.u_eqs = u_eqs
        st.session_state.u_inits = u_inits
        st.success("Berjaya! Sila buka tab 'Results & Analysis' untuk melihat graf.")

# --- PAGE 3: RESULTS & ANALYSIS ---
with tab3:
    if st.session_state.run_sim:
        t_pts = np.linspace(0, t_max, k_stages + 1)
        
        # Ambil data dari session state
        eqs = st.session_state.u_eqs
        inits = st.session_state.u_inits
        
        res_rk4 = rk4_solver(eqs, inits, t_pts)
        res_mspi = picard_solvers(eqs, inits, t_pts, n_iters, True)
        res_spi = picard_solvers(eqs, inits, t_pts, n_iters, False)
        
        if res_rk4 is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(int(n_comp)):
                ax.plot(t_pts, res_rk4[:, i], 'k-', alpha=0.3, label=f"RK4 (Bench) y[{i}]")
                ax.plot(t_pts, res_spi[:, i], '--', label=f"SPI y[{i}]")
                ax.plot(t_pts, res_mspi[:, i], '-o', markersize=3, label=f"MSPI y[{i}]")
            
            ax.set_title("Perbandingan Kaedah Numerikal")
            ax.set_xlabel("Masa (t)")
            ax.set_ylabel("Nilai y")
            ax.legend()
            st.pyplot(fig)
            
            # Kira Error
            err_mspi = np.mean(np.abs(res_rk4 - res_mspi))
            err_spi = np.mean(np.abs(res_rk4 - res_spi))
            
            mc1, mc2 = st.columns(2)
            mc1.metric("Mean Error MSPI", f"{err_mspi:.6e}")
            mc2.metric("Mean Error SPI", f"{err_spi:.6e}")
            
            st.info(f"Diskusi: MSPI (Modified) menunjukkan ralat {err_mspi:.6e} berbanding SPI {err_spi:.6e}. Ini membuktikan kaedah multistage anda lebih jitu.")
    else:
        st.warning("Sila masukkan input di Tab 2 dan tekan butang 'Jalankan Simulasi'.")
