import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. LOGIK PENGIRAAN (MSPI) ---
def mspi_solver(eq_strings, y0, t_points, num_iter):
    try:
        y_history = [np.array(y0, dtype=float)]
        curr_y = np.array(y0, dtype=float)
        for k in range(1, len(t_points)):
            h = t_points[k] - t_points[k-1]
            yn_k = curr_y.copy()
            for n in range(num_iter):
                new_yn = []
                ctx = {"y": yn_k, "t": t_points[k-1], "np": np}
                for i in range(len(y0)):
                    # Tukar ^ kepada ** secara automatik untuk elak ralat math
                    f_val = eval(eq_strings[i].replace('^', '**'), {"__builtins__": None}, ctx)
                    new_yn.append(curr_y[i] + f_val * h)
                yn_k = np.array(new_yn, dtype=float)
            curr_y = yn_k
            y_history.append(curr_y)
        return np.array(y_history)
    except Exception as e:
        st.error(f"Ralat Matematik: {e}")
        return None

# --- 2. PENGURUSAN HALAMAN (NAVIGASI) ---
if 'page' not in st.session_state:
    st.session_state.page = 1

def change_page(page_num):
    st.session_state.page = page_num

# --- 3. ANTARAMUKA PENGGUNA (UI) ---
st.set_page_config(page_title="MSPI Solver Pro", layout="wide")
st.title("🚀 ODE Solver: Multistage Picard Iteration")

# Halaman 1: Konfigurasi
if st.session_state.page == 1:
    st.header("1️⃣ Konfigurasi Algoritma")
    col1, col2 = st.columns(2) # PEMBETULAN: Letak angka 2
    with col1:
        n_comp = st.number_input("Bilangan Kompartmen", 1, 10, 3, key="n_comp")
        t_max = st.number_input("Tempoh Masa (T)", 1, 100, 20, key="t_max")
    with col2:
        k_val = st.slider("Bilangan Sub-selang (k)", 5, 200, 50, key="k_val")
        n_iter = st.slider("Bilangan Iterasi (n)", 1, 20, 5, key="n_iter")
    
    st.divider()
    st.button("Seterusnya: Input Persamaan ➡️", on_click=change_page, args=(2,), use_container_width=True)

# Halaman 2: Input Persamaan
elif st.session_state.page == 2:
    st.header("2️⃣ Input Persamaan & Nilai Awal")
    u_eqs = []
    u_inits = []
    
    for i in range(st.session_state.n_comp):
        st.subheader(f"Kompartmen y[{i}]")
        c1, c2 = st.columns() # PEMBETULAN: Guna nisbah
        eq = c1.text_input(f"dy[{i}]/dt =", value="1 - y**2" if i==0 else "0", key=f"eq_{i}")
        init = c2.number_input(f"Nilai Awal y[{i}]", value=-0.5 if i==0 else 0.0, key=f"init_{i}")
        u_eqs.append(eq)
        u_inits.append(init)
    
    st.session_state.u_eqs = u_eqs
    st.session_state.u_inits = u_inits

    st.divider()
    b1, b2 = st.columns(2)
    b1.button("⬅️ Kembali", on_click=change_page, args=(1,), use_container_width=True)
    if b2.button("Jalankan Simulasi & Lihat Hasil ➡️", use_container_width=True):
        st.session_state.page = 3
        st.rerun()

# Halaman 3: Keputusan
elif st.session_state.page == 3:
    st.header("3️⃣ Keputusan & Analisis Graf")
    t_pts = np.linspace(0, st.session_state.t_max, st.session_state.k_val + 1)
    results = mspi_solver(st.session_state.u_eqs, st.session_state.u_inits, t_pts, st.session_state.n_iter)
    
    if results is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(len(st.session_state.u_inits)):
            ax.plot(t_pts, results[:, i], label=f"y[{i}]")
        ax.set_xlabel("Masa (t)")
        ax.set_ylabel("Nilai y")
        ax.legend()
        st.pyplot(fig)
    
    st.divider()
    st.button("⬅️ Edit Persamaan", on_click=change_page, args=(2,), use_container_width=True)
