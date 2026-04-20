import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. SOLVER ENGINE ---
def mspi_solver(eq_strings, y0, t_points, num_iter):
    try:
        y_history = [np.array(y0, dtype=float)]
        curr_y = np.array(y0, dtype=float)
        for k in range(1, len(t_points)):
            h = t_points[k] - t_points[k-1]
            yn_k = curr_y.copy()
            for n in range(num_iter):
                new_yn = []
                # Konteks untuk eval: menyokong 'y' sebagai array
                ctx = {"y": yn_k, "t": t_points[k-1], "np": np}
                for i in range(len(y0)):
                    # Tukar ^ kepada ** untuk mengelakkan ralat bitwise_xor
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

# --- 2. STATE & NAVIGATION ---
if 'page' not in st.session_state:
    st.session_state.page = 1

def move_to(p):
    st.session_state.page = p

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="MSPI Solver", layout="wide")
st.title("🚀 ODE Solver: Multistage Picard Iteration")

# PAGE 1: CONFIGURATION
if st.session_state.page == 1:
    st.header("1️⃣ Konfigurasi Algoritma")
    c1, c2 = st.columns(2) # FIXED: Diletakkan nilai 2
    with c1:
        n_comp = st.number_input("Bilangan Kompartmen", 1, 10, 3, key="n_comp")
        t_max = st.number_input("Tempoh Masa (T)", 1, 100, 20, key="t_max")
    with c2:
        k_val = st.slider("Bilangan Sub-selang (k)", 5, 200, 50, key="k_val")
        n_iter = st.slider("Bilangan Iterasi (n)", 1, 20, 5, key="n_iter")
    
    st.divider()
    st.button("Seterusnya: Input Persamaan ➡️", on_click=move_to, args=(2,), use_container_width=True)

# PAGE 2: EQUATION INPUT
elif st.session_state.page == 2:
    st.header("2️⃣ Input Persamaan & Nilai Awal")
    st.info("Gunakan y, y untuk mewakili pembolehubah.")
    
    u_eqs = []
    u_inits = []
    
    for i in range(st.session_state.n_comp):
        st.subheader(f"Kompartmen y[{i}]")
        # FIXED: Menambah nisbah untuk mengelakkan TypeError
        c1, c2 = st.columns() 
        eq = c1.text_input(f"dy[{i}]/dt =", value="1 - y**2" if i==0 else "0", key=f"eq_{i}")
        init = c2.number_input(f"Nilai Awal y[{i}]", value=-0.5 if i==0 else 0.0, key=f"init_{i}")
        u_eqs.append(eq)
        u_inits.append(init)
    
    st.session_state.u_eqs = u_eqs
    st.session_state.u_inits = u_inits

    st.divider()
    b_col1, b_col2 = st.columns(2) # FIXED: Diletakkan nilai 2
    b_col1.button("⬅️ Kembali", on_click=move_to, args=(1,), use_container_width=True)
    if b_col2.button("Jalankan Simulasi & Hasil ➡️", use_container_width=True):
        st.session_state.page = 3
        st.rerun()

# PAGE 3: RESULTS
elif st.session_state.page == 3:
    st.header("3️⃣ Hasil Simulasi")
    
    t_pts = np.linspace(0, st.session_state.t_max, st.session_state.k_val + 1)
    results = mspi_solver(st.session_state.u_eqs, st.session_state.u_inits, t_pts, st.session_state.n_iter)
    
    if results is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(len(st.session_state.u_inits)):
            ax.plot(t_pts, results[:, i], label=f"y[{i}] (MSPI)")
        ax.set_xlabel("Masa (t)")
        ax.set_ylabel("Nilai y")
        ax.legend()
        st.pyplot(fig)
        
        st.success("Simulasi selesai dengan jayanya!")
    
    st.divider()
    st.button("⬅️ Edit Persamaan", on_click=move_to, args=(2,), use_container_width=True)
