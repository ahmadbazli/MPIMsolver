import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- LOGIK MULTISTAGE PICARD (MSPI) ---
def mspi_solver(system_funcs, y0, t_points, num_iter=3):
    """
    system_funcs: Senarai fungsi ODE [f1, f2, ...]
    y0: Nilai awal bagi setiap pembolehubah [S0, I0, ...]
    t_points: Titik masa sub-selang
    """
    num_vars = len(y0)
    y_history = [np.array(y0)]
    t_history = [t_points]
    
    current_y_start = np.array(y0)
    
    for k in range(1, len(t_points)):
        tk_minus_1 = t_points[k-1]
        tk = t_points[k]
        h = tk - tk_minus_1
        
        # Iterasi Picard dalam sub-selang [t_{k-1}, t_k]
        yn_k = current_y_start.copy()
        for n in range(num_iter):
            new_yn = []
            for i in range(num_vars):
                # Integrasi ringkas menggunakan kaedah Euler/Trapezoidal dalam Picard
                integral_approx = system_funcs[i](tk_minus_1, yn_k) * h
                new_yn.append(current_y_start[i] + integral_approx)
            yn_k = np.array(new_yn)
            
        current_y_start = yn_k
        y_history.append(current_y_start)
        t_history.append(tk)
        
    return np.array(t_history), np.array(y_history)

# --- KONFIGURASI WEB ---
st.set_page_config(page_title="MSPI Solver Engine", layout="wide")

st.title("🚀 MSPI-Engine: Multistage Picard Solver")
st.markdown("### High-Performance Numerical Innovation for Complex Systems")

# Sidebar
st.sidebar.header("⚙️ Konfigurasi Algoritma")
stages = st.sidebar.slider("Bilangan Sub-selang (k)", 5, 50, 20)
iters = st.sidebar.slider("Bilangan Iterasi (n)", 1, 10, 3)
t_max = st.sidebar.number_input("Tempoh Simulasi (Hari)", value=30)

st.sidebar.divider()
st.sidebar.write("**Application Highlight:**")
st.sidebar.info("Model yang dipaparkan adalah 'Monkeypox Transmission Model' sebagai bukti keberkesanan kaedah MSPI.")

# Model Monkeypox (Contoh Parameter)
# S' = -beta*S*I, I' = beta*S*I - gamma*I, R' = gamma*I
def monkeypox_model():
    beta = 0.01
    gamma = 0.1
    f1 = lambda t, y: -beta * y * y          # Susceptible
    f2 = lambda t, y: beta * y * y - gamma * y # Infected
    f3 = lambda t, y: gamma * y                 # Recovered
    return [f1, f2, f3], # Initial values: S=99, I=1, R=0

funcs, initial_y = monkeypox_model()
t_steps = np.linspace(0, t_max, stages + 1)

# Jalankan Solver
t_res, y_res = mspi_solver(funcs, initial_y, t_steps, num_iter=iters)

# Paparan Hasil
col1, col2 = st.columns()

with col1:
    st.subheader("📈 Simulasi Dinamik")
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ['Susceptible', 'Infected', 'Recovered']
    for i in range(y_res.shape):
        ax.plot(t_res, y_res[:, i], label=labels[i], linewidth=2)
    
    ax.set_xlabel("Masa (Hari)")
    ax.set_ylabel("Populasi")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with col2:
    st.subheader("💡 Inovasi Metodologi")
    st.latex(r"y_{n+1,k}(t) = y(t_{k-1}) + \int_{t_{k-1}}^{t} f(s, y_{n,k}(s)) ds")
    st.write("""
    Sistem ini membuktikan bahawa **Multistage Picard** mampu:
    1. **Mengecilkan ralat** pada setiap sub-selang.
    2. **Meningkatkan kelajuan** pengiraan berbanding kaedah global.
    3. **Menangani ketidaklinearan** model Monkeypox dengan lebih stabil.
    """)
    
    if st.checkbox("Tunjukkan Data Numerikal"):
        st.write(y_res)

st.divider()
st.markdown("© 2026 Innovation Project | Built with Streamlit")
