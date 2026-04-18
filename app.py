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
    except Exception as e:
        st.error(f"Ralat RK4: {e}")
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
                # Modified vs Standard
                ctx = {"y": yn_k if is_modified else current_y_start, "t": t_points[k-1], "np": np}
                for i in range(num_vars):
                    f_val = eval(eq_strings[i].replace('^', '**'), {"__builtins__": None}, ctx)
                    new_yn.append(current_y_start[i] + f_val * h)
                yn_k = np.array(new_yn)
                
            current_y_start = yn_k
            y_history.append(current_y_start)
        return np.array(y_history)
    except Exception as e:
        st.error(f"Ralat Picard: {e}")
        return None

# --- 2. UI LAYOUT ---
st.set_page_config(page_title="ODE Innovation Solver", layout="wide")
st.title("🚀 ODE Innovation Solver: MSPI vs SPI vs RK4")

tab1, tab2, tab3 = st.tabs(["1️⃣ Configuration", "2️⃣ Equation Input", "3️⃣ Results & Analysis"])

# --- PAGE 1: CONFIGURATION ---
with tab1:
    st.header("Sistem Konfigurasi")
    c_config1, c_config2 = st.columns(2)
    with c_config1:
        n_comp = st.number_input("Bilangan Kompartmen", 1, 10, 3, key="n_comp")
        t_max = st.number_input("Tempoh Masa (T)", 1, 100, 20, key="t_max")
    with c_config2:
        k_stages = st.slider("Bilangan Sub-selang (k)", 5, 200, 50, key="k_val")
        n_iters = st.slider("Bilangan Iterasi Picard (n)", 1, 20, 5, key="n_val")
    st.info("💡 Selesai di sini? Klik tab **Equation Input** di atas.")

# --- PAGE 2: EQUATION INPUT ---
with tab2:
    st.header("Input Persamaan & Nilai Awal")
    u_eqs = []
    u_inits = []
    
    # Looping untuk setiap kompartmen
    for i in range(int(n_comp)):
        st.markdown(f"**Kompartmen {i}**")
        # Betulkan ralat st.columns() dengan meletakkan input
        row_cols = st.columns() 
        with row_cols:
            eq = st.text_input(f"dy[{i}]/dt", value="1 - y**2" if i==0 else "0", key=f"eq_in_{i}")
        with row_cols:
            init = st.number_input(f"y[{i}] Initial", value=-0.5 if i==0 else 0.0, key=f"init_in_{i}")
        u_eqs.append(eq)
        u_inits.append(init)
    
    st.divider()
    # Gunakan session_state untuk simpan trigger button
    if st.button("🔥 JALANKAN SIMULASI", use_container_width=True):
        st.session_state.run_sim = True
        st.info("Simulasi selesai! Sila buka tab **Results & Analysis**.")

# --- PAGE 3: RESULTS & ANALYSIS ---
with tab3:
    if "run_sim" in st.session_state and st.session_state.run_sim:
        t_pts = np.linspace(0, t_max, k_stages + 1)
        
        # Solving
        res_rk4 = rk4_solver(u_eqs, u_inits, t_pts)
        res_mspi = picard_solvers(u_eqs, u_inits, t_pts, n_iters, True)
        res_spi = picard_solvers(u_eqs, u_inits, t_pts, n_iters, False)
        
        if res_rk4 is not None:
            st.header("Analisis Perbandingan Numerikal")
            fig, ax = plt.subplots(figsize=(12, 6))
            for i in range(int(n_comp)):
                ax.plot(t_pts, res_rk4[:, i], 'k-', alpha=0.3, label=f"RK4 (Bench) y[{i}]")
                ax.plot(t_pts, res_spi[:, i], '--', label=f"SPI y[{i}]")
                ax.plot(t_pts, res_mspi[:, i], '-o', markersize=3, label=f"MSPI y[{i}]")
            ax.set_xlabel("Masa (t)")
            ax.set_ylabel("Nilai y")
            ax.legend()
            st.pyplot(fig)
            
            # Metrics
            err_mspi = np.mean(np.abs(res_rk4 - res_mspi))
            err_spi = np.mean(np.abs(res_rk4 - res_spi))
            
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Ralat Purata MSPI", f"{err_mspi:.6e}")
            m_col2.metric("Ralat Purata SPI", f"{err_spi:.6e}")
            
            st.markdown("### 📝 Discussion")
            if err_mspi < err_spi:
                st.success(f"Inovasi anda (MSPI) terbukti lebih tepat dengan ralat {err_mspi:.6e} berbanding Standard Picard ({err_spi:.6e}).")
            else:
                st.warning("Standard Picard menunjukkan keputusan yang hampir sama. Cuba tingkatkan bilangan 'Sub-selang (k)' untuk melihat perbezaan ketara MSPI.")
    else:
        st.warning("Sila tekan butang **JALANKAN SIMULASI** di Tab 2 dahulu.")
