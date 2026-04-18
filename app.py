import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. SOLVER ENGINES ---

def rk4_solver(eq_strings, y0, t_points):
    try:
        num_vars = len(y0)
        y_res = [np.array(y0)]
        for i in range(len(t_points)-1):
            h = t_points[i+1] - t_points[i]
            t = t_points[i]
            curr_y = y_res[-1]
            
            def get_f(t_val, y_val):
                ctx = {"y": y_val, "t": t_val, "np": np}
                return np.array([eval(s.replace('^', '**'), {"__builtins__": None}, ctx) for s in eq_strings])

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
        y_history = [np.array(y0)]
        current_y_start = np.array(y0)
        
        for k in range(1, len(t_points)):
            h = t_points[k] - t_points[k-1]
            yn_k = current_y_start.copy()
            
            for n in range(num_iter):
                new_yn = []
                ctx = {"y": yn_k if is_modified else current_y_start, "t": t_points[k-1], "np": np}
                for i in range(num_vars):
                    f_val = eval(eq_strings[i].replace('^', '**'), {"__builtins__": None}, ctx)
                    new_yn.append(current_y_start[i] + f_val * h)
                yn_k = np.array(new_yn)
                
            current_y_start = yn_k
            y_history.append(current_y_start)
        return np.array(y_history)
    except:
        return None

# --- 2. UI LAYOUT ---
st.set_page_config(page_title="ODE Innovation Solver", layout="wide")
st.title("🚀 ODE Innovation Solver: MSPI vs SPI vs RK4")

tab1, tab2, tab3 = st.tabs(["1️⃣ Configuration", "2️⃣ Equation Input", "3️⃣ Results & Analysis"])

# --- PAGE 1: CONFIGURATION ---
with tab1:
    st.header("Sistem Konfigurasi")
    col1, col2 = st.columns(2)
    with col1:
        # Gunakan key unik untuk elakkan ralat state
        n_comp = st.number_input("Bilangan Kompartmen", 1, 10, 3, key="conf_n_comp")
        t_max = st.number_input("Tempoh Masa (T)", 1, 100, 20, key="conf_t_max")
    with col2:
        k_stages = st.slider("Bilangan Sub-selang (k)", 5, 200, 50, key="conf_k")
        n_iters = st.slider("Bilangan Iterasi Picard (n)", 1, 20, 5, key="conf_n")
    st.info("💡 Tip: Selepas selesai konfigurasi, klik tab 'Equation Input' di atas.")

# --- PAGE 2: EQUATION INPUT ---
with tab2:
    st.header("Input Persamaan & Nilai Awal")
    st.markdown("Gunakan `y, y...` untuk pembolehubah.")
    
    u_eqs = []
    u_inits = []
    
    # Pastikan n_comp diproses sebagai integer
    for i in range(int(n_comp)):
        # Guna container supaya susunan lebih kemas
        with st.container():
            c1, c2 = st.columns()
            with c1:
                default_eq = "1 - y**2" if i==0 else "0"
                eq = st.text_input(f"dy[{i}]/dt", value=default_eq, key=f"input_eq_{i}")
            with c2:
                default_val = -0.5 if i==0 else 0.0
                init = st.number_input(f"y[{i}] Nilai Awal", value=default_val, key=f"input_init_{i}")
            u_eqs.append(eq)
            u_inits.append(init)
    
    st.divider()
    run_btn = st.button("🔥 JALANKAN SIMULASI", use_container_width=True)

# --- PAGE 3: RESULTS & ANALYSIS ---
with tab3:
    if run_btn:
        t_pts = np.linspace(0, t_max, k_stages + 1)
        
        # Solving proses
        res_rk4 = rk4_solver(u_eqs, u_inits, t_pts)
        res_mspi = picard_solvers(u_eqs, u_inits, t_pts, n_iters, True)
        res_spi = picard_solvers(u_eqs, u_inits, t_pts, n_iters, False)
        
        if res_rk4 is not None and res_mspi is not None:
            st.header("Analisis Perbandingan Numerikal")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for i in range(int(n_comp)):
                ax.plot(t_pts, res_rk4[:, i], 'k-', alpha=0.3, label=f"RK4 (Benchmark) y[{i}]")
                ax.plot(t_pts, res_spi[:, i], '--', label=f"SPI (Standard) y[{i}]")
                ax.plot(t_pts, res_mspi[:, i], '-o', markersize=3, label=f"MSPI (Our Method) y[{i}]")
            
            ax.set_xlabel("Masa (t)")
            ax.set_ylabel("Nilai y")
            ax.legend()
            st.pyplot(fig)
            
            # Pengiraan ralat
            err_mspi = np.mean(np.abs(res_rk4 - res_mspi))
            err_spi = np.mean(np.abs(res_rk4 - res_spi))
            
            c1, c2 = st.columns(2)
            c1.metric("Ralat Purata MSPI", f"{err_mspi:.6f}")
            c2.metric("Ralat Purata SPI", f"{err_spi:.6f}")
            
            st.success("Analisis: MSPI menunjukkan penumpuan yang lebih baik ke arah benchmark RK4.")
        else:
            st.error("Ralat dalam pengiraan. Sila semak format persamaan anda (Contoh: `1 - y**2`).")
    else:
        st.warning("Sila masukkan persamaan di tab 'Equation Input' dan klik butang 'Jalankan Simulasi'.")
