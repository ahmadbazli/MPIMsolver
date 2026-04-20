from textwrap import dedent

app_code = dedent(r'''
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="MPIM Prediction System", layout="wide")

# =========================
# Utility Functions
# =========================
def initialize_state():
    defaults = {
        "page": 1,
        "n_comp": 8,
        "n_init": 8,
        "n_iter": 5,
        "t0": 0.0,
        "t_end": 10.0,
        "steps": 200,
        "equations": [""] * 8,
        "initials": [""] * 8,
        "run_done": False,
        "results": None,
        "error_table": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def resize_inputs(n_comp, n_init):
    old_eq = st.session_state.equations
    old_iv = st.session_state.initials
    st.session_state.equations = [old_eq[i] if i < len(old_eq) else "" for i in range(n_comp)]
    st.session_state.initials = [old_iv[i] if i < len(old_iv) else "" for i in range(n_init)]

def safe_eval(expr, t, state):
    allowed = {
        "t": t,
        "np": np,
        "math": math,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "log": math.log,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
    }
    for i, val in enumerate(state, start=1):
        allowed[f"x{i}"] = float(val)
    return eval(expr, {"__builtins__": {}}, allowed)

def system_func_factory(equations):
    def f(t, y):
        vals = []
        for eq in equations:
            vals.append(float(safe_eval(eq, t, y)))
        return np.array(vals, dtype=float)
    return f

def rk4_system(f, t0, t_end, y0, steps):
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    t = np.linspace(t0, t_end, steps + 1)
    h = (t_end - t0) / steps
    y = np.zeros((steps + 1, n), dtype=float)
    y[0] = y0

    for i in range(steps):
        ti = t[i]
        yi = y[i]
        k1 = f(ti, yi)
        k2 = f(ti + h/2, yi + h*k1/2)
        k3 = f(ti + h/2, yi + h*k2/2)
        k4 = f(ti + h, yi + h*k3)
        y[i+1] = yi + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return t, y

def picard_system(f, t0, t_end, y0, steps, iterations):
    """
    Numerical Picard iteration over the whole interval.
    x^{k+1}(t_i) = x0 + integral_{t0}^{t_i} f(s, x^k(s)) ds
    Integral approximated by cumulative rectangle rule.
    """
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    t = np.linspace(t0, t_end, steps + 1)
    h = (t_end - t0) / steps

    guess = np.zeros((steps + 1, n), dtype=float)
    guess[:] = y0  # initial guess constant on the interval

    for _ in range(iterations):
        new_guess = np.zeros_like(guess)
        new_guess[0] = y0
        for i in range(1, steps + 1):
            fi_minus_1 = f(t[i-1], guess[i-1])
            new_guess[i] = new_guess[i-1] + h * fi_minus_1
        guess = new_guess.copy()

    return t, guess

def mpim_system(f, t0, t_end, y0, steps, iterations):
    """
    Multistage Picard Iterative Method:
    Apply Picard iteration locally on each small subinterval and pass
    the last value as the initial condition for the next stage.
    """
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    t = np.linspace(t0, t_end, steps + 1)
    h = (t_end - t0) / steps
    y = np.zeros((steps + 1, n), dtype=float)
    y[0] = y0

    for m in range(steps):
        tm = t[m]
        ym = y[m].copy()
        stage_val = ym.copy()

        for _ in range(iterations):
            stage_val = ym + h * f(tm, stage_val)

        y[m+1] = stage_val

    return t, y

def build_plot_dataframe(t, y_picard, y_mpim, y_rk4):
    df = pd.DataFrame({"time": t})
    n = y_picard.shape[1]
    for i in range(n):
        df[f"PIM_C{i+1}"] = y_picard[:, i]
        df[f"MPIM_C{i+1}"] = y_mpim[:, i]
        df[f"RK4_C{i+1}"] = y_rk4[:, i]
    return df

def build_error_table(y_picard, y_mpim, y_rk4):
    n = y_rk4.shape[1]
    rows = []
    for i in range(n):
        picard_abs = np.abs(y_picard[:, i] - y_rk4[:, i])
        mpim_abs = np.abs(y_mpim[:, i] - y_rk4[:, i])

        rows.append({
            "Compartment": f"C{i+1}",
            "PIM Final Absolute Error": float(picard_abs[-1]),
            "MPIM Final Absolute Error": float(mpim_abs[-1]),
            "PIM Mean Absolute Error": float(np.mean(picard_abs)),
            "MPIM Mean Absolute Error": float(np.mean(mpim_abs)),
        })
    return pd.DataFrame(rows)

def validate_inputs():
    if any(not eq.strip() for eq in st.session_state.equations):
        return False, "Please fill in all compartment equations."
    if any(not iv.strip() for iv in st.session_state.initials):
        return False, "Please fill in all initial values."
    return True, ""

# =========================
# Page Rendering
# =========================
initialize_state()

st.title("Multistage Picard Iterative Method Prediction System")
st.caption("Methods included: Original Picard Iterative Method, Multistage Picard Iterative Method, and Fourth Order Runge-Kutta Method")

progress_map = {1: 25, 2: 50, 3: 75, 4: 100}
st.progress(progress_map[st.session_state.page])

# -------------------------
# PAGE 1
# -------------------------
if st.session_state.page == 1:
    st.subheader("Welcome Page")

    st.markdown("""
    This system is developed for **prediction using Ordinary Differential Equation (ODE) models**.
    
    The system compares three methods:
    - **Original Picard Iterative Method**
    - **Multistage Picard Iterative Method (MPIM)**
    - **Fourth Order Runge-Kutta Method (RK4)**
    """)

    st.info("To use this system, prepare your ODE model and initial conditions first.")

    if st.button("Start"):
        st.session_state.page = 2
        st.rerun()

# -------------------------
# PAGE 2
# -------------------------
elif st.session_state.page == 2:
    st.subheader("Page 2: System Configuration")

    col1, col2 = st.columns(2)

    with col1:
        n_comp = st.number_input("Number of Compartments", min_value=1, max_value=20, value=int(st.session_state.n_comp), step=1)
        n_iter = st.number_input("Number of Iterations", min_value=1, max_value=50, value=int(st.session_state.n_iter), step=1)
        t0 = st.number_input("Initial Time", value=float(st.session_state.t0), step=0.1)

    with col2:
        n_init = st.number_input("Number of Initial Values", min_value=1, max_value=20, value=int(st.session_state.n_init), step=1)
        t_end = st.number_input("Final Time", value=float(st.session_state.t_end), step=0.1)
        steps = st.number_input("Number of Time Steps", min_value=10, max_value=5000, value=int(st.session_state.steps), step=10)

    st.session_state.n_comp = int(n_comp)
    st.session_state.n_init = int(n_init)
    st.session_state.n_iter = int(n_iter)
    st.session_state.t0 = float(t0)
    st.session_state.t_end = float(t_end)
    st.session_state.steps = int(steps)

    resize_inputs(st.session_state.n_comp, st.session_state.n_init)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Back"):
            st.session_state.page = 1
            st.rerun()
    with c2:
        if st.button("Next"):
            if st.session_state.n_comp != st.session_state.n_init:
                st.warning("For this system, it is recommended that the number of compartments equals the number of initial values.")
            st.session_state.page = 3
            st.rerun()

# -------------------------
# PAGE 3
# -------------------------
elif st.session_state.page == 3:
    st.subheader("Page 3: Enter ODE Compartments and Initial Conditions")

    st.markdown("Use variables like `x1`, `x2`, ..., `xn` and `t` in the equations.")
    st.code("Example:\n-delta*x1 + 0.1*x2\n0.5*x1 - 0.2*x2", language="python")
    st.caption("For constants such as beta or delta, directly replace them with numeric values, for example `-0.2*x1 + 0.1*x2`.")

    st.markdown("### Compartment Equations")
    for i in range(st.session_state.n_comp):
        st.session_state.equations[i] = st.text_input(
            f"Compartment {i+1} ODE",
            value=st.session_state.equations[i],
            placeholder=f"Example: -0.2*x{i+1} + 0.1*x1"
        )

    st.markdown("### Initial Conditions")
    for i in range(st.session_state.n_init):
        st.session_state.initials[i] = st.text_input(
            f"Initial Value {i+1}",
            value=st.session_state.initials[i],
            placeholder=f"Example: 10"
        )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Back"):
            st.session_state.page = 2
            st.rerun()

    with c2:
        if st.button("Run"):
            ok, msg = validate_inputs()
            if not ok:
                st.error(msg)
            elif st.session_state.n_comp != st.session_state.n_init:
                st.error("The number of compartments must be equal to the number of initial values.")
            else:
                try:
                    equations = st.session_state.equations
                    initials = [float(eval(iv, {"__builtins__": {}}, {})) for iv in st.session_state.initials]

                    f = system_func_factory(equations)

                    t_picard, y_picard = picard_system(
                        f, st.session_state.t0, st.session_state.t_end,
                        initials, st.session_state.steps, st.session_state.n_iter
                    )
                    t_mpim, y_mpim = mpim_system(
                        f, st.session_state.t0, st.session_state.t_end,
                        initials, st.session_state.steps, st.session_state.n_iter
                    )
                    t_rk4, y_rk4 = rk4_system(
                        f, st.session_state.t0, st.session_state.t_end,
                        initials, st.session_state.steps
                    )

                    result_df = build_plot_dataframe(t_rk4, y_picard, y_mpim, y_rk4)
                    error_df = build_error_table(y_picard, y_mpim, y_rk4)

                    st.session_state.results = result_df
                    st.session_state.error_table = error_df
                    st.session_state.run_done = True
                    st.session_state.page = 4
                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred while running the system: {e}")

# -------------------------
# PAGE 4
# -------------------------
elif st.session_state.page == 4:
    st.subheader("Page 4: Result of the System")

    if st.session_state.results is None or st.session_state.error_table is None:
        st.warning("No results found. Please run the system first.")
    else:
        df = st.session_state.results
        err_df = st.session_state.error_table
        n = st.session_state.n_comp

        st.markdown("## Graphical Comparison")
        selected_comp = st.selectbox(
            "Select compartment to display",
            options=[f"C{i+1}" for i in range(n)],
            index=0
        )
        idx = int(selected_comp[1:])

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["time"], df[f"PIM_C{idx}"], label="Original Picard Iterative Method")
        ax.plot(df["time"], df[f"MPIM_C{idx}"], label="Multistage Picard Iterative Method")
        ax.plot(df["time"], df[f"RK4_C{idx}"], label="Fourth Order Runge-Kutta Method")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(f"Prediction Graph for {selected_comp}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.markdown("## Error Table")
        st.caption("Absolute errors are computed using RK4 as the benchmark/reference solution.")
        st.dataframe(err_df.style.format({
            "PIM Final Absolute Error": "{:.6f}",
            "MPIM Final Absolute Error": "{:.6f}",
            "PIM Mean Absolute Error": "{:.6f}",
            "MPIM Mean Absolute Error": "{:.6f}",
        }), use_container_width=True)

        csv = err_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Error Table (CSV)",
            data=csv,
            file_name="error_table.csv",
            mime="text/csv"
        )

        st.markdown("## Result Summary")
        better_counts = (err_df["MPIM Mean Absolute Error"] < err_df["PIM Mean Absolute Error"]).sum()
        if better_counts > 0:
            st.success(f"MPIM gives smaller mean absolute error than Original Picard for {better_counts} compartment(s).")
        else:
            st.info("For this run, MPIM does not outperform Original Picard in mean absolute error.")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Back"):
            st.session_state.page = 3
            st.rerun()
    with c2:
        if st.button("Return to Welcome Page"):
            st.session_state.page = 1
            st.rerun()
''')
