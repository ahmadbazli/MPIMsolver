import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- ENGINE: MULTISTAGE PICARD SOLVER (MSPI) ---
def mspi_solver(eq_strings, y0, t_points, num_iter=3):
    num_vars = len(y0)
    y_history = [np.array(y0)]
    t_history = [t_points]
    current_y_start = np.array(y0)
    
    # Pre-compile strings to code for speed
    for k in range(1, len(t_points)):
        tk_minus_1 = t_points[k-1]
        tk = t_points[k]
        h = tk - tk_minus_1
        
        yn_k = current_y_start.copy()
        for n in range(num_iter):
            new_yn = []
            # Sediakan context untuk eval (y, y...)
            context = {"y": yn_k, "np": np}
            for i in range(num_vars):
                try:
                    # EVALUATE persamaan daripada input user
                    f_val = eval(eq_strings[i], {"__builtins__": None}, context)
                    new_yn.append(current_y_start[i] + f_val * h)
                except Exception as e:
                    st.error(f"Ralat pada Persamaan {i+1}: {e}")
                    return None, None
            yn_k = np.array(new_yn)
            
        current_y_start = yn_k
        y_history.append(current_y_start)
        t_history.append(tk)
        
    return np.array(t_history), np.array(y_history)

# --- UI WEB ---
st.set_page_config(page_title="Universal MSPI Solver", layout="wide")

st.title("🧪 Universal MSPI: Dynamic ODE Engine")
st.markdown("Sistem ini membolehkan penyelesaian **Multistage Picard** untuk sebarang model kompartmen.")

# Sidebar: Konfigurasi Algoritma
st.sidebar.header("⚙️ Konfigurasi Algoritma")
num_compartments = st.sidebar.number_input("Bilangan Kompartmen (Persamaan)", 1, 10, 3)
stages = st.sidebar.slider("Bilangan Sub-selang (k)", 5, 100, 50)
iters = st.sidebar.slider("Bilangan Iterasi (n)", 1, 15, 5)
t_max = st.sidebar.number_input("Tempoh Masa (T)", value=20)

st.divider()

# Input Persamaan & Initial Values
st.subheader("📝 Input Model ODE")
st.info("Gunakan **y, y, y...** untuk mewakili pembolehubah (cth: S, I, R).")

cols = st.columns(num_compartments)
user_eqs = []
user_inits = []

for i in range(num_compartments):
    with cols[i]:
        st.markdown(f"**Kompartmen y[{i}]**")
        eq = st.text_input(f"dy[{i}]/dt =", value="-0.1 * y * y" if i==0 else "0.1 * y * y" if i==1 else "0")
        init = st.number_input(f"Nilai Awal y[{i}]", value=0.9 if i==0 else 0.1 if i==1 else 0.0, key=f"init_{i}")
        user_eqs.append(eq)
        user_inits.append(init)

# Butang Solve
if st.button("🚀 Jalankan Simulasi MSPI"):
    t_steps = np.linspace(0, t_max, stages + 1)
    t_res, y_res = mspi_solver(user_eqs, user_inits, t_steps, num_iter=iters)
    
    if t_res is not None:
        col_res1, col_res2 = st.columns()
        
        with col_res1:
            st.subheader("📈 Graf Simulasi")
            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(num_compartments):
                ax.plot(t_res, y_res[:, i], label=f"y[{i}]", linewidth=2)
            ax.set_xlabel("Masa (t)")
            ax.set_ylabel("Nilai")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
        with col_res2:
            st.subheader("📊 Data Result")
            st.dataframe(y_res)
            
st.divider()
st.caption("Developed for INDES 2026 - Multistage Picard Iterative Method Innovation")
