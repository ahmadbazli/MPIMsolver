import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="MPIM Prediction System",
    layout="wide"
)

# =====================================
# SESSION STATE
# =====================================
if "page" not in st.session_state:
    st.session_state.page = 1

if "run_result" not in st.session_state:
    st.session_state.run_result = None

# =====================================
# FUNCTIONS
# =====================================

def eval_equation(eq, t, vals, params, var_names):
    env = {
        "t": t,
        "np": np,
        "math": math,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
        "sqrt": math.sqrt,
        "log": math.log
    }

    # Compartment variables ikut nama yang user isi
    for name, value in zip(var_names, vals):
        env[name] = value

    # Parameters
    for name, value in params.items():
        env[name] = value

    return eval(eq, {"__builtins__": {}}, env)


def model_function(equations, params, var_names):
    def f(t, y):
        result = []
        for eq in equations:
            result.append(eval_equation(eq, t, y, params, var_names))
        return np.array(result, dtype=float)
    return f


# =====================================
# RK4 METHOD
# =====================================
def rk4_solver(f, t0, tf, y0, nstep):
    h = (tf - t0) / nstep
    t = np.linspace(t0, tf, nstep + 1)

    y = np.zeros((nstep + 1, len(y0)))
    y[0] = y0

    for i in range(nstep):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h*k1/2)
        k3 = f(t[i] + h/2, y[i] + h*k2/2)
        k4 = f(t[i] + h, y[i] + h*k3)

        y[i+1] = y[i] + h*(k1 + 2*k2 + 2*k3 + k4)/6

    return t, y


# =====================================
# ORIGINAL PICARD
# =====================================
def picard_solver(f, t0, tf, y0, nstep, niter):
    h = (tf - t0) / nstep
    t = np.linspace(t0, tf, nstep + 1)

    y = np.zeros((nstep + 1, len(y0)))
    y[:] = y0

    for _ in range(niter):
        new_y = np.zeros_like(y)
        new_y[0] = y0

        for i in range(1, nstep + 1):
            val = f(t[i-1], y[i-1])
            new_y[i] = new_y[i-1] + h * val

        y = new_y.copy()

    return t, y


# =====================================
# MULTISTAGE PICARD
# =====================================
def mpim_solver(f, t0, tf, y0, nstep, niter):
    h = (tf - t0) / nstep
    t = np.linspace(t0, tf, nstep + 1)

    y = np.zeros((nstep + 1, len(y0)))
    y[0] = y0

    for i in range(nstep):
        stage = y[i].copy()

        for _ in range(niter):
            stage = y[i] + h * f(t[i], stage)

        y[i+1] = stage

    return t, y


# =====================================
# ERROR TABLE
# =====================================
def create_error_table(pim, mpim, rk4, var_names):
    rows = []

    for i in range(rk4.shape[1]):
        err_pim = np.abs(pim[:, i] - rk4[:, i])
        err_mpim = np.abs(mpim[:, i] - rk4[:, i])

        rows.append({
            "Compartment": var_names[i],
            "PIM Final Error": err_pim[-1],
            "MPIM Final Error": err_mpim[-1],
            "PIM Mean Error": np.mean(err_pim),
            "MPIM Mean Error": np.mean(err_mpim)
        })

    return pd.DataFrame(rows)


# =====================================
# PAGE 1
# =====================================
if st.session_state.page == 1:

    st.title("Multistage Picard Iterative Method Prediction System")

    st.markdown("""
    ### Welcome
    The Multistage Picard Iterative Method Prediction System is a specialized computational platform developed to solve and predict the behaviour of first-order systems of Ordinary Differential Equations (ODEs). Many real-world mathematical models such as disease transmission, population interaction, chemical processes, engineering systems, and dynamic forecasting problems can be expressed in the form of first-order differential equation systems. This system was created to simplify the analysis and numerical solution of such models in an efficient and user-friendly environment.
    
    This platform is especially useful for students, lecturers, researchers, engineers, scientists, and analysts who work with mathematical modelling or prediction problems. Instead of solving complex equations manually, users can input their differential equation models together with initial conditions and obtain numerical approximations quickly and efficiently.
    
    This system compares three important numerical approaches:

    - Original Picard Iterative Method – a classical iterative technique for approximate solutions.
    - Multistage Picard Iterative Method – an improved version designed for better long-term stability and higher prediction accuracy.
    - Fourth Order Runge-Kutta Method – a well-known benchmark numerical solver widely used in science and engineering.

    The main purpose of this system is to help users solve, simulate, predict, and compare ODE models over time. It allows users to observe solution behaviour, evaluate accuracy between methods, and generate graphical outputs for analysis and research purposes.

    This system is suitable for applications in:

    - Epidemic and disease modelling
    - Population dynamics
    - Physics and engineering systems
    - Finance and economic forecasting
    - Biological and environmental models
    - Academic learning and mathematical research
    
    Click Start to begin exploring your ODE prediction model using powerful iterative numerical methods.
    """)

    if st.button("Start", key="start_btn"):
        st.session_state.page = 2
        st.rerun()


# =====================================
# PAGE 2
# =====================================
elif st.session_state.page == 2:

    st.title("System Configuration")

    ncomp = st.number_input("Number of Compartments", 1, 20, 8)

    # Lock initial values ikut number of compartments
    ninit = ncomp
    st.number_input(
        "Number of Initial Values",
        value=int(ninit),
        disabled=True,
        key="locked_initial_values"
    )

    # Number of parameters
    nparam = st.number_input("Number of Parameters", 0, 20, 6)

    niter = st.number_input("Number of Iterations", 1, 50, 5)
    t0 = st.number_input("Initial Time", value=0.0)
    tf = st.number_input("Final Time", value=10.0)

    # Fixed step size
    h = 0.01

    # Auto compute number of steps
    if tf > t0:
        nstep = int((tf - t0) / h)
    else:
        nstep = 1

    st.session_state.ncomp = ncomp
    st.session_state.ninit = ninit
    st.session_state.nparam = nparam
    st.session_state.niter = niter
    st.session_state.t0 = t0
    st.session_state.tf = tf
    st.session_state.h = h
    st.session_state.nstep = nstep

    st.info(f"Fixed Step Size Used: {h}")
    st.caption(f"Auto computed number of steps: {nstep}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back", key="page2_back"):
            st.session_state.page = 1
            st.rerun()

    with col2:
        if st.button("Next", key="page2_next"):
            if tf <= t0:
                st.error("Final Time must be greater than Initial Time.")
            else:
                st.session_state.page = 3
                st.rerun()


# =====================================
# PAGE 3
# =====================================
elif st.session_state.page == 3:

    st.title("Enter ODE, Initial Conditions and Parameters")

    st.write("You may use custom variable names such as x, y, z, s, i, r, u, v.")
    st.write("You may also use parameters such as alpha, beta, gamma, epsilon, lambda, rho.")

    var_names = []
    equations = []
    initials = []
    param_names = []
    param_values = []

    st.subheader("Compartment Variable Names")

    for i in range(st.session_state.ncomp):
        vname = st.text_input(
            f"Compartment Variable {i+1}",
            placeholder="Example: x or s or i",
            key=f"vname_{i}"
        )
        var_names.append(vname.strip())

    st.subheader("ODE Compartments")

    for i in range(st.session_state.ncomp):
        eq = st.text_input(
            f"Equation {i+1}",
            placeholder="Example: -beta*s*i or -alpha*x + beta*y",
            key=f"eq_{i}"
        )
        equations.append(eq)

    st.subheader("Initial Values")

    for i in range(st.session_state.ninit):
        label_name = var_names[i] if i < len(var_names) and var_names[i] != "" else f"Variable {i+1}"
        val = st.text_input(
            f"Initial Value for {label_name}",
            placeholder="Example: 10",
            key=f"iv_{i}"
        )
        initials.append(val)

    st.subheader("Parameters")

    if st.session_state.nparam == 0:
        st.caption("No parameter input is required.")
    else:
        for i in range(st.session_state.nparam):
            colp1, colp2 = st.columns(2)

            with colp1:
                pname = st.text_input(
                    f"Parameter Name {i+1}",
                    placeholder="Example: alpha",
                    key=f"pname_{i}"
                )

            with colp2:
                pvalue = st.text_input(
                    f"Parameter Value {i+1}",
                    placeholder="Example: 0.25",
                    key=f"pvalue_{i}"
                )

            param_names.append(pname)
            param_values.append(pvalue)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back", key="page3_back"):
            st.session_state.page = 2
            st.rerun()

    with col2:
        if st.button("Run", key="page3_run"):
            try:
                for i, name in enumerate(var_names, start=1):
                    if name == "":
                        st.error(f"Compartment Variable {i} cannot be empty.")
                        st.stop()

                if len(set(var_names)) != len(var_names):
                    st.error("Compartment variable names must be unique.")
                    st.stop()

                for i, eq in enumerate(equations, start=1):
                    if eq.strip() == "":
                        st.error(f"Equation {i} cannot be empty.")
                        st.stop()

                for i, val in enumerate(initials, start=1):
                    if val.strip() == "":
                        st.error(f"Initial Value {i} cannot be empty.")
                        st.stop()

                y0 = [float(v) for v in initials]

                params = {}
                for i, (name, value) in enumerate(zip(param_names, param_values), start=1):
                    if name.strip() == "":
                        st.error(f"Parameter Name {i} cannot be empty.")
                        st.stop()
                    if value.strip() == "":
                        st.error(f"Parameter Value {i} cannot be empty.")
                        st.stop()
                    params[name.strip()] = float(value)

                f = model_function(equations, params, var_names)

                t1, pim = picard_solver(
                    f,
                    st.session_state.t0,
                    st.session_state.tf,
                    y0,
                    st.session_state.nstep,
                    st.session_state.niter
                )

                t2, mpim = mpim_solver(
                    f,
                    st.session_state.t0,
                    st.session_state.tf,
                    y0,
                    st.session_state.nstep,
                    st.session_state.niter
                )

                t3, rk4 = rk4_solver(
                    f,
                    st.session_state.t0,
                    st.session_state.tf,
                    y0,
                    st.session_state.nstep
                )

                err_df = create_error_table(pim, mpim, rk4, var_names)

                st.session_state.run_result = {
                    "t": t1,
                    "pim": pim,
                    "mpim": mpim,
                    "rk4": rk4,
                    "error": err_df,
                    "params": params,
                    "var_names": var_names
                }

                st.session_state.page = 4
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")


# =====================================
# PAGE 4
# =====================================
elif st.session_state.page == 4:

    st.title("Result of the System")

    result = st.session_state.run_result

    t = result["t"]
    pim = result["pim"]
    mpim = result["mpim"]
    rk4 = result["rk4"]
    err = result["error"]
    var_names = result["var_names"]

    comp = st.selectbox(
        "Select Compartment",
        var_names,
        key="page4_compartment"
    )

    idx = var_names.index(comp)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        t,
        pim[:, idx],
        color="red",
        linestyle="--",
        linewidth=1.8,
        label="Original Picard (PIM)"
    )

    ax.plot(
        t,
        mpim[:, idx],
        color="blue",
        linestyle="-",
        linewidth=2.0,
        label="MPIM"
    )

    ax.plot(
        t,
        rk4[:, idx],
        color="black",
        linestyle="None",
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="RK4"
    )

    y_focus = np.concatenate([mpim[:, idx], rk4[:, idx]])
    y_min = np.min(y_focus)
    y_max = np.max(y_focus)

    if abs(y_max - y_min) < 1e-8:
        margin = 0.1
    else:
        margin = 0.1 * (y_max - y_min)

    ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(f"{comp} vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    st.subheader("Error Table")
    st.dataframe(err)

    st.write("RK4 is used as benchmark solution.")
    st.write("The graph is zoomed based on MPIM and RK4 so the curve behavior can be seen more clearly.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back", key="page4_back"):
            st.session_state.page = 3
            st.rerun()

    with col2:
        if st.button("Home", key="page4_home"):
            st.session_state.page = 1
            st.rerun()
