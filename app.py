import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. ENGINE SOLVER ---
def mspi_solver(eq_strings, y0, t_points, num_iter):
    try:
        y_history = [np.array(y0, dtype=float)]
        curr_y = np.array(y0, dtype=float)
        for k in range(1, len(t_points)):
            h = t_points[k] - t_points[k-1]
            yn_k = curr_y.copy()
            for n in range(num_iter):
                new_yn = []
                # Konteks untuk pengiraan: tukar ^ ke ** secara automatik
                ctx = {"y": yn_k, "t": t_points[k-1], "np": np}
                for i in range(len(y0)):
                    clean_eq = eq_strings[i].replace('^', '**')
                    f_val = eval(clean_eq, {"__builtins__": None}, ctx)
                    new_yn.append(curr_y[i] + f_val * h)
                yn_k = np.array(new_yn, dtype=float)
            curr_y = yn_k
            y_history.append(curr_y)
        return np.array(y_history)
    except Exception as e:
        st.error(f"Ralat Matematik: {e}")
        return None

# --- 2. NAVIGASI ---
if 'page' not in st.session_state:
    st.session_state.page = 1

# --- 3. UI UTAMA ---
st.set_page_config(page_title="MSPI Solver", layout="wide")
st.title("🚀 MSPI-Engine: Multistage Picard Solver")

# --- PAGE 1: CONFIG ---
if st.session_state.page == 1:
    st.header("1️⃣ Konfigurasi")
    # PEMBETULAN: Mesti ada angka 2 di dalam kurungan!
    c1, c2 = st.columns(2) 
    n_comp = c1.number_input("Bilangan Kompartmen", 1, 5, 1)
    t_max = c1.number_input("Tempoh Masa", 1, 100, 20)
    k_val = c2.slider("Bilangan Sub-selang (k)", 5, 100, 50)
    n_iter = c2.slider("Bilangan Iterasi (n)", 1, 10, 5)
    
    if st.button("Next: Masukkan Persamaan ➡️", use_container_width=True):
        st.session_state.n_comp = n_comp
        st.session_state.t_max = t_max
        st.session_state.k_val = k_val
        st.session_state.n_iter = n_iter
        st.session_state.page = 2
        st.rerun()

# --- PAGE 2: INPUT PERSAMAAN ---
elif st.session_state.page == 2:
    st.header("2️⃣ Input Model ODE")
    st.info("Gunakan y, y... Contoh: 1 - y**2")
    
    eqs = []
    inits = []
    for i in range(int(st.session_state.n_comp)):
        # PEMBETULAN: Mesti ada senarai di dalam kurungan!
        cols = st.columns() 
        e = cols.text_input(f"dy[{i}]/dt =", value="1 - y**2" if i==0 else "0")
        v = cols.number_input(f"Nilai Awal y[{i}]", value=-0.5 if i==0 else 0.0)
        eqs.append(e)
        inits.append(v)
    
    col_nav = st.columns(2) # PEMBETULAN: Letak angka 2
    if col_nav.button("⬅️ Kembali"):
        st.session_state.page = 1
        st.rerun()
    if col_nav.button("Jalankan Simulasi 🚀", use_container_width=True):
        st.session_state.u_eqs = eqs
        st.session_state.u_inits = inits
        st.session_state.page = 3
        st.rerun()

# --- PAGE 3: KEPUTUSAN ---
elif st.session_state.page == 3:
    st.header("3️⃣ Hasil Simulasi")
    t_pts = np.linspace(0, st.session_state.t_max, st.session_state.k_val + 1)
    res = mspi_solver(st.session_state.u_eqs, st.session_state.u_inits, t_pts, st.session_state.n_iter)
    
    if res is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(len(st.session_state.u_inits)):
            ax.plot(t_pts, res[:, i], label=f"y[{i}]")
        ax.set_xlabel("Masa (t)")
        ax.set_ylabel("Nilai y")
        ax.legend()
        st.pyplot(fig)
        st.success("Simulasi Berjaya!")
    
    if st.button("⬅️ Edit Persamaan"):
        st.session_state.page = 2
        st.rerun()
