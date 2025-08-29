import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gamma, chi2
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# Set page config as the first Streamlit command
st.set_page_config(layout="wide", page_title="Sistem Bonus-Malus Optimal", page_icon="📊")

# Increase Pandas Styler max elements limit
pd.set_option("styler.render.max_elements", 746416)

# Custom CSS untuk styling
st.markdown(
    """
    <style>
        /* Background utama jadi hitam */
        .stApp {
            background-color: #000000;
            color: white; /* teks default putih */
        }

        /* Table/Dataframe styling */
        table.dataframe, .dataframe th, .dataframe td {
            border: 1px solid white !important; /* border putih */
            color: white !important;            /* teks putih */
        }

        /* Hilangkan striping default agar lebih clean */
        .dataframe tbody tr:nth-child(odd) {
            background-color: #000000 !important;
        }
        .dataframe tbody tr:nth-child(even) {
            background-color: #000000 !important;
        }

        /* Scrollbar hitam */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: #555;
        }
        ::-webkit-scrollbar-track {
            background: #000;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar untuk unggah file dan pengaturan
with st.sidebar:
    st.header("⚙️ Pengaturan")
    uploaded_file = st.file_uploader("📂 Unggah file CSV/XLSX", type=["csv", "xlsx"], help="Unggah file data klaim (CSV atau XLSX).")
    st.markdown("---")
    st.info("Gunakan sidebar untuk mengunggah data dan atur parameter dasar.")

# Judul aplikasi
st.title("📊 Sistem Bonus-Malus Optimal (Estimasi Bayesian)")
st.markdown("Aplikasi ini membantu dalam menghitung premi bonus-malus optimal berdasarkan data frekuensi klaim dengan distribusi negatif binomial.")

# Konten utama
if uploaded_file is not None:
    try:
        # Membaca file dengan penanganan error yang lebih ketat
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Validasi bahwa df adalah DataFrame
        if not isinstance(df, pd.DataFrame):
            st.error(f"❌ File tidak menghasilkan DataFrame yang valid. Tipe data: {type(df)}")
            st.stop()

        # Validasi bahwa DataFrame tidak kosong dan memiliki kolom
        if df.empty or len(df.columns) == 0:
            st.error("❌ DataFrame kosong atau tidak memiliki kolom. Pastikan file berisi data tabular.")
            st.stop()

        # Bagian 1: Tampilkan Data
        with st.container():
            st.header("📋 Data yang Diunggah")
            # Display only the first 100 rows to improve performance
            st.write("**Pratinjau (100 baris pertama):**")
            st.dataframe(df.head(100).style.highlight_null(props='background-color:red').set_caption("Pratinjau Data (100 Baris Pertama)"), use_container_width=True)
            st.write(f"**Total Baris:** {len(df)} | **Total Kolom:** {len(df.columns)} | **Total Sel:** {len(df) * len(df.columns)}")
            # Option to download full DataFrame
            csv = df.to_csv(index=False)
            st.download_button("📥 Unduh DataFrame Lengkap (CSV)", csv, "data_full.csv", "text/csv")

        # Bagian 2: Pilih Kolom Frekuensi
        with st.container():
            st.header("🔍 Pilih Kolom Frekuensi")
            col1, col2 = st.columns([1, 2])
            with col1:
                freq_column = st.selectbox("Kolom frekuensi:", df.columns, help="Pilih kolom yang berisi jumlah klaim.")
            with col2:
                st.write("**Deskripsi:** Kolom ini digunakan untuk menghitung distribusi klaim.")

        # Estimasi Parameter
        if freq_column:
            freq_data = df[freq_column].dropna()
            if freq_data.empty:
                st.error("❌ Kolom frekuensi kosong setelah menghapus nilai null.")
                st.stop()

            xbar = freq_data.mean()
            skuadrat = freq_data.var()

            if skuadrat <= xbar:
                st.error("❌ Variansi harus lebih besar dari rata-rata untuk distribusi negatif binomial.")
                st.stop()

            tau = xbar / (skuadrat - xbar)
            aa = (xbar ** 2) / (skuadrat - xbar)

            with st.container():
                st.header("📈 Estimasi Parameter")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Rata-rata (x̄)", f"{xbar:.4f}")
                col2.metric("Variansi (s²)", f"{skuadrat:.4f}")
                col3.metric("Tau (τ)", f"{tau:.4f}")
                col4.metric("Alpha (α)", f"{aa:.4f}")

            # Uji Hipotesis Chi-Square
            with st.expander("🔬 Uji Hipotesis Chi-Square (Goodness of Fit)"):
                st.write("**Perhitungan untuk Distribusi Negatif Binomial**")
                unique_categories = sorted(freq_data.unique())
                n_categories = len(unique_categories)
                observed = freq_data.value_counts().sort_index().reindex(unique_categories, fill_value=0).values

                P = np.zeros(n_categories)
                P[0] = (tau / (1 + tau)) ** aa
                for k in range(n_categories - 1):
                    P[k + 1] = ((k + aa) / ((k + 1) * (1 + tau))) * P[k]
                P = P / np.sum(P)

                n = len(freq_data)
                expected = P * n
                chisquare = np.sum((observed - expected) ** 2 / expected)
                df_chi = n_categories - 1 - 2
                critical_value = chi2.ppf(1 - 0.05, df_chi)

                st.metric("Chi-Square", f"{chisquare:.4f}")
                st.metric("Degrees of Freedom", df_chi)
                st.metric("Nilai Kritis (α=0.05)", f"{critical_value:.4f}")

                if chisquare < critical_value:
                    st.success("✅ Data cocok dengan distribusi negatif binomial.")
                else:
                    st.warning("⚠️ Data tidak cocok. Lanjutkan dengan distribusi negatif binomial?")
                    if st.button("Lanjutkan"):
                        st.info("Anda memilih untuk melanjutkan.")

            # Bagian 3: Pilih Loss Function
            with st.container():
                st.header("⚖️ Pilih Loss Function")
                loss_function = st.radio("Jenis loss function:", ("Squared-Error Loss", "Absolute Loss Function"), horizontal=True)

            # Bagian 4: Pengaturan Premi
            with st.container():
                st.header("💰 Pengaturan Premi")
                col1, col2 = st.columns(2)
                with col1:
                    premium_in_data = st.radio("Premi ada di data?", ("Ya", "Tidak"), horizontal=True)
                if premium_in_data == "Ya":
                    with col2:
                        premium_column = st.selectbox("Kolom premi:", df.columns)
                        selected_row = st.number_input("Indeks polis:", min_value=0, max_value=len(df)-1, step=1)
                        premium_value = df[premium_column].iloc[selected_row]
                else:
                    with col2:
                        premium_value = st.number_input("Masukkan premi:", min_value=0.0, value=100.0, step=1.0)
                st.metric("Nilai Premi", f"{premium_value:.2f}")

            # Bagian 5: Input k dan t
            with st.container():
                st.header("🔢 Input Nilai Maksimum k dan t")
                col1, col2 = st.columns(2)
                with col1:
                    max_k = st.number_input("Maksimum k:", min_value=0, max_value=25, step=1, value=7)
                with col2:
                    max_t = st.number_input("Maksimum t:", min_value=0, max_value=25, step=1, value=7)

            # Bagian 6: Hasil Bonus-Malus
            with st.container():
                st.header("🏆 Premi Sistem Bonus-Malus Optimal")
                with st.spinner("Menghitung premi bonus-malus..."):
                    if loss_function == "Squared-Error Loss":
                        t_vals = np.arange(0, max_t + 1)
                        k_vals = np.arange(0, max_k + 1)
                        result = np.zeros((len(t_vals), len(k_vals)))
                        for t_idx, t in enumerate(t_vals):
                            for k_idx, k in enumerate(k_vals):
                                if t == 0 and k != 0:
                                    result[t_idx, k_idx] = np.nan
                                else:
                                    result[t_idx, k_idx] = premium_value * ((tau * (aa + k)) / (aa * (tau + t)))
                        result_df = pd.DataFrame(result, index=[f"t={t}" for t in t_vals], columns=[f"k={k}" for k in k_vals]).round(2)
                        st.write("**Premi (Squared-Error Loss)**")
                    else:
                        baseline_median = gamma.ppf(0.5, a=aa, scale=1.0/tau)
                        t_vals = np.arange(0, max_t + 1)
                        k_vals = np.arange(0, max_k + 1)
                        result = np.zeros((len(t_vals), len(k_vals)))
                        for t_idx, t in enumerate(t_vals):
                            for k_idx, k in enumerate(k_vals):
                                if t == 0 and k != 0:
                                    result[t_idx, k_idx] = np.nan
                                else:
                                    alpha = aa + k
                                    rate = tau + t
                                    post_median = gamma.ppf(0.5, a=alpha, scale=1.0/rate)
                                    result[t_idx, k_idx] = premium_value * (post_median / baseline_median)
                        result_df = pd.DataFrame(result, index=[f"t={t}" for t in t_vals], columns=[f"k={k}" for k in k_vals]).round(2)
                        st.write("**Premi (Absolute Loss Function)**")

                    # Tampilkan tabel
                    st.dataframe(result_df.style.format(na_rep="").format(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")).background_gradient(cmap='Blues', axis=None).set_caption("Tabel Premi Bonus-Malus"), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {str(e)}")
        st.stop()
else:
    st.info("⬆️ Silakan unggah file CSV atau XLSX untuk memulai.")
