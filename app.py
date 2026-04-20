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
                ctx = {"y": yn_k, "t": t_points[k-1], "np": np}
                for i in range(len(y0)):
                    f_val = eval(eq_strings[i].replace('^', '**'), {"__builtins__": None}, ctx)
                    new_yn.append(curr_y[i] + f_val * h)
                yn_k = np.array(new_yn, dtype=float)
            curr_y = yn_k
            y_history.append(curr_y)
        return np.array(y_history)
    except: return None

# --- 2. STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = 1

def gopage(n):
    st.session_state.page = n

# --- 3. UI ---
st.set_page_config(page_title="MSPI Solver", layout="wide")
st.title("🚀 ODE Solver: MSPI")

# PAGE 1: CONFIG
if st.session_state.page == 1:
    st.header("Step 1: Konfigurasi")
    # WAJIB LETAK NOMBOR 2 DALAM COLUMNS
    c1, c2 = st.columns(2) 
    n_comp = c1.number_input("Bilangan Kompartmen", 1, 10, 3, key="n_comp")
    t_max = c1.number_input("Tempoh Masa", 1, 100, 20, key="t_max")
    k_val = c2.slider("Sub-selang (k)", 5, 200, 50, key="k_val")
    n_iter = c2.slider("Iterasi (n)", 1, 20, 5, key="n_iter")
    
    st.button("Next ➡️", on_click=gopage, args=(2,), use_container_width=True)

# PAGE 2: INPUT
elif st.session_state.page == 2:
    st.header("Step 2: Input Persamaan")
    u_eqs = []
    u_inits = []
    for i in range(int(st.session_state.n_comp)):
        # WAJIB LETAK LIST DALAM COLUMNS
        cols = st.columns() 
        e = cols.text_input(f"dy[{i}]/dt", value="1 - y**2" if i==0 else "0", key=f"e{i}")
        v = cols.number_input(f"y[{i}] Mula", value=-0.5 if i==0 else 0.0, key=f"v{i}")
        u_eqs.append(e)
        u_inits.append(v)
    
    st.session_state.u_eqs = u_eqs
    st.session_state.u_inits = u_inits

    st.button("Back ⬅️", on_click=gopage, args=(1,))
    if st.button("Run Simulation ➡️", use_container_width=True):
        st.session_state.page = 3
        st.rerun()

# PAGE 3: RESULTS
elif st.session_state.page == 3:
    st.header("Step 3: Keputusan")
    t_pts = np.linspace(0, st.session_state.t_max, st.session_state.k_val + 1)
    res = mspi_solver(st.session_state.u_eqs, st.session_state.u_inits, t_pts, st.session_state.n_iter)
    
    if res is not None:
        fig, ax = plt.subplots()
        for i in range(len(st.session_state.u_inits)):
            ax.plot(t_pts, res[:, i], label=f"y[{i}]")
        ax.legend()
        st.pyplot(fig)
        st.success("Berjaya!")
    
    st.button("Back ⬅️", on_click=gopage, args=(2,), use_container_width=True)
