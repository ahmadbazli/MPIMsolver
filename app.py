import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- LOGIK MATEMATIK (Ubah suai ikut tesis anda) ---
def standard_picard(func, y0, t_span, iterations=5):
    # Contoh implementasi ringkas
    results = [y0]
    # ... proses iterasi standard ...
    return results

def modified_picard(func, y0, t_span, iterations=5):
    # MASUKKAN FORMULA MODIFIED ANDA DI SINI
    # Contoh: Anda tambah penambahbaikan pada langkah integrasi atau weighting
    results = [y0]
    # ... proses iterasi modified ...
    return results

# --- INTERFACE WEB STREAMLIT ---
st.set_page_config(page_title="MPIM Solver", layout="wide")

st.title("🚀 MPIM-Solver: High-Speed Numerical Engine")
st.markdown("""
Sistem ini mempamerkan kecekapan **Modified Picard Iterative Method (MPIM)** berbanding kaedah konvensional dalam menyelesaikan model dinamik kompleks.
""")

# Sidebar untuk Input
st.sidebar.header("Parameter Model")
y0 = st.sidebar.number_input("Nilai Awal (Initial Guess)", value=0.1)
iter_count = st.sidebar.slider("Bilangan Iterasi", 1, 20, 5)
test_case = st.sidebar.selectbox("Pilih Model Ujian", ["Monkeypox Model", "SIR Model", "General ODE"])

# Layout Utama
col1, col2 = st.columns()

with col1:
    st.subheader("📊 Analisis Penumpuan (Convergence)")
    if st.button("Jalankan Simulasi"):
        start_time = time.time()
        # Panggil fungsi matematik anda
        # Contoh data dummy untuk demonstrasi:
        t = np.linspace(0, 10, 100)
        error_std = [0.5, 0.3, 0.2, 0.15, 0.12] # Ralat makin sikit
        error_mod = [0.5, 0.2, 0.05, 0.01, 0.001] # MPIM menjunam lebih laju
        
        duration = time.time() - start_time

        # Plot Graf Ralat
        fig, ax = plt.subplots()
        ax.plot(range(1, 6), error_std, 'r--o', label='Standard Picard')
        ax.plot(range(1, 6), error_mod, 'g-s', label='Modified Picard (MPIM)')
        ax.set_xlabel("Bilangan Iterasi")
        ax.set_ylabel("Ralat (Error)")
        ax.set_yscale('log')
        ax.legend()
        st.pyplot(fig)
        
        st.success(f"Simulasi tamat dalam {duration:.4f} saat")

with col2:
    st.subheader("💡 Kelebihan MPIM")
    st.info("""
    - **Pantas:** Mencapai ketepatan (tolerance) dalam iterasi yang lebih sedikit.
    - **Stabil:** Lebih berkesan untuk model non-linear.
    - **ESG Compliant:** Pengiraan lebih efisien mengurangkan beban pemprosesan server.
    """)
    
    # Tampilkan Formula (LaTeX)
    st.latex(r'''
    y_{n+1}(t) = y_0 + \int_{t_0}^{t} f(s, y_n(s)) ds
    ''')
    st.caption("Asas Kaedah Picard - Modifikasi anda meningkatkan kadar penumpuan.")

# Footer
st.divider()
st.markdown("Developed for **INDES 2026** | Focused on Numerical Innovation")
