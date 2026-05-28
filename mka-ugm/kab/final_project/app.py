import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Intelligent Habit Development", page_icon="💡", layout="wide")

st.title("Intelligent Habit Development")
st.markdown("Sistem rekomendasi habit cerdas untuk membangun kebiasaan baru secara personal.")

with st.expander("1. Data yang Akan Digunakan", expanded=True):
    st.markdown("**Data**")
    st.write(
        "- goals pengguna: kebugaran, produktivitas, stop merokok\n"
        "- riwayat habit: checklist / rekaman waktu habit harian\n"
        "- kepribadian\n"
        "- post-challenge self-assessment\n"
        "- ketersediaan jadwal\n"
        "- waktu luang mingguan"
    )

with st.expander("2. Proses Pengolahan Data Menjadi Informasi", expanded=False):
    st.subheader("Riwayat habit")
    st.write("Event habit → Rekayasa fitur → Model regresi logistik → Estimasi probabilitas keberhasilan")

    st.code(
        "P(berhasil) = σ(β0 + β1*waktu + β2*tidur + β3*rentetan + β4*hari kerja)",
        language="python",
    )
    st.write(
        "Menggabungkan data kebiasaan historis, mengekstraksi fitur kontekstual, melatih model prediksi, "
        "dan memperbarui model setiap minggu dengan data pengguna baru."
    )

    st.subheader("Notifikasi pengingat")
    st.write(
        "State = waktu pengingat yang berbeda; Action = klik notifikasi; Reward = pengisian habit. "
        "Algoritma RL mempelajari jadwal optimal, menguji beberapa jadwal, mengukur imbalan penyelesaian, "
        "dan memperbarui kebijakan hingga konvergensi ke konfigurasi terbaik."
    )

    st.subheader("Event habit terlewat")
    st.write(
        "Pattern mining + clustering pada event yang terlewatkan untuk mendeteksi pola kegagalan dan menghasilkan hipotesis penyebab."
    )

    st.subheader("On Board & Post-Challenge Assessment")
    st.write(
        "Heuristic planning, rule-based expert system, dan decision support system membantu menyusun rencana habit dan evaluasi post-challenge."
    )

with st.expander("3. Input Pengguna", expanded=True):
    st.subheader("Masukkan data pengguna")
    col1, col2 = st.columns(2)

    with col1:
        goals = st.multiselect(
            "Goals pengguna", ["Kebugaran", "Produktivitas", "Stop Merokok", "Belajar", "Tidur Lebih Baik"]
        )
        personality = st.selectbox(
            "Kepribadian", ["Terstruktur", "Fleksibel", "Ambivert", "Introvert", "Ekstrovert"]
        )
        schedule = st.text_area(
            "Ketersediaan jadwal (hari dan jam)", "Senin-Jumat 06:00-08:00, Sabtu 09:00-11:00"
        )

    with col2:
        habit_history = st.text_area(
            "Riwayat habit / checklist harian",
            "Push-up: 4/7 hari, Tidur 7 jam: 5/7 hari, Membaca: 3/7 hari",
        )
        weekly_free_time = st.slider("Waktu luang mingguan (jam)", 0, 30, 10)
        post_assessment = st.select_slider(
            "Post-challenge self-assessment", ["Kurang", "Cukup", "Baik", "Sangat Baik"]
        )

    if st.button("Proses Data & Buat Rekomendasi"):
        st.session_state.run = True

if st.session_state.get("run", False):
    st.markdown("---")
    st.header("Hasil Rekomendasi Habit")

    # contoh rekomendasi sederhana
    goal_label = goals[0] if goals else "Kebugaran"
    st.subheader(f"Tujuan: {goal_label}")

    recommendations = [
        {"Minggu": "1–2", "Habit": "5 push-up / hari"},
        {"Minggu": "3–4", "Habit": "10 push-up / hari"},
        {"Minggu": "5+", "Habit": "15 push-up / hari"},
    ]
    rec_df = pd.DataFrame(recommendations)
    st.table(rec_df)
    st.write("**Waktu optimal:** 07:10 pagi")
    st.info("Pengguna lebih mudah membangun habit baru dan onboarding aplikasi menjadi lebih efektif.")

    st.subheader("Prediksi Keberhasilan Habit")
    # simulate logistic probability
    x = min(max(0.1 * weekly_free_time + 0.2 * len(goals), 0), 1)
    prob = int(60 + 20 * x)
    st.metric("Probabilitas menyelesaikan habit olahraga hari ini", f"{prob}%")
    st.write("Saran: pindahkan pengingat ke pagi hari jika probabilitas < 70%.")
    st.success("Pengguna dapat mengantisipasi kegagalan dan meningkatkan konsistensi habit.")

    st.subheader("Rekomendasi Jadwal Pengingat Optimal")
    st.write("Waktu pengingat optimal: 07:10")
    st.write("Peningkatan tingkat keberhasilan: +18%")
    st.success("Meningkatkan engagement pengguna dan membuat notifikasi lebih efektif.")

    st.subheader("Analisis Pola Kegagalan Habit")
    st.write("Habit 'membaca' sering gagal pada:")
    st.write("- Jumat malam\n- akhir pekan")
    st.write("Saran: ubah jadwal menjadi pukul 19:00 hari kerja")
    st.success("Pengguna memahami penyebab kegagalan dan dapat memperbaiki strategi habit.")

    st.markdown("---")
    st.header("Estimasi Pendapatan & Model Bisnis")
    st.write("Freemium + langganan fitur AI.")

    business = pd.DataFrame(
        [
            {"Paket": "Free", "Fitur": "habit tracking dasar", "Harga": "Gratis"},
            {"Paket": "Pro", "Fitur": "AI insight + analytics", "Harga": "$4/bulan"},
        ]
    )
    st.table(business)

    st.subheader("Estimasi Pendapatan")
    active_users = 50000
    conversion = 0.05
    price = 4
    revenue_month = active_users * conversion * price
    revenue_year = revenue_month * 12
    st.write(f"Asumsi pengguna aktif: {active_users}")
    st.write(f"Konversi premium: {int(conversion*100)}%")
    st.write(f"Pendapatan: ${revenue_month:,.0f} per bulan ≈ ${revenue_year:,.0f} per tahun")
    st.write("Jika pengguna mencapai 200.000: $40.000 per bulan")

    st.markdown("---")
    st.header("Nilai Strategis Produk")
    st.write("Aplikasi habit biasa: Habit Tracking")
    st.write("Produk ini: Habit Intelligence System — Track → Analyze → Optimize Habit")

    st.write(
        "Sistem ini menggabungkan analisis prediktif, rekomendasi jadwal optimal, dan pola kegagalan untuk membantu pengguna membangun habit yang lebih konsisten."
    )
