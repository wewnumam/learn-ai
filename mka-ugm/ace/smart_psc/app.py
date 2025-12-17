import streamlit as st
import subprocess

st.title("Smart Public Service Center (MAS Demo)")

st.write("Demo cara kerja Multi-Agent System")

if st.button("Jalankan Simulasi Layanan"):
    st.info("Menjalankan agent...")
    subprocess.Popen(["python", "run_agents.py"])
    st.success("Simulasi berjalan. Cek terminal log.")
