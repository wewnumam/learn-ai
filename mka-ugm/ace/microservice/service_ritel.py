import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(page_title="Ritel System", page_icon="🏢")

def get_db_connection():
    return sqlite3.connect('ban_system.db')

st.title("🏢 Dashboard Ritel (Inventory Service)")
st.caption("Aplikasi ini berjalan di Server Ritel (Port 8501)")

# Pilih Login Ritel
ritel_id = st.sidebar.selectbox("Login Sebagai:", ["R1", "R2"])

# --- MONITORING STOK ---
st.subheader(f"Gudang Stok: {ritel_id}")
conn = get_db_connection()

# Query Stok
df_stok = pd.read_sql_query(f"SELECT * FROM stok WHERE ritel_id = '{ritel_id}'", conn)
st.dataframe(df_stok, use_container_width=True)

# Form Update Stok
with st.expander("Update Stok Manual"):
    with st.form("update"):
        tipe = st.selectbox("Tipe Ban", ["Ban_Sedan", "Ban_SUV"])
        qty_baru = st.number_input("Jumlah Baru", min_value=0, value=50)
        
        if st.form_submit_button("Update Database"):
            c = conn.cursor()
            c.execute("UPDATE stok SET jumlah = ? WHERE ritel_id = ? AND tipe_ban = ?", 
                      (qty_baru, ritel_id, tipe))
            conn.commit()
            st.success("Stok berhasil diupdate!")
            st.rerun()

# --- MONITORING ORDER ---
st.divider()
st.subheader("Pesanan Masuk")
df_order = pd.read_sql_query(f"SELECT * FROM pesanan WHERE ritel_id = '{ritel_id}'", conn)

if not df_order.empty:
    st.dataframe(df_order, use_container_width=True)
else:
    st.info("Belum ada pesanan dari Pembeli.")

conn.close()
