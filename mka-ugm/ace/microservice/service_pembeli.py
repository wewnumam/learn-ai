import streamlit as st
import sqlite3
import pandas as pd
import uuid

st.set_page_config(page_title="Aplikasi Pembeli", page_icon="🛒")

def get_db_connection():
    return sqlite3.connect('ban_system.db')

st.title("🛒 Cari Ban Online")
st.caption("Aplikasi ini berjalan di HP Pembeli (Port 8502)")

# --- SERVICE: SEARCH ---
tipe_dicari = st.selectbox("Saya mencari ban untuk:", ["Ban_Sedan", "Ban_SUV"])

if st.button("🔍 Cari Ketersediaan"):
    conn = get_db_connection()
    # Mengambil data dari 'API' (Database)
    query = f"SELECT * FROM stok WHERE tipe_ban = '{tipe_dicari}' AND jumlah > 0"
    hasil = pd.read_sql_query(query, conn)
    conn.close()
    
    if not hasil.empty:
        st.session_state['hasil_cari'] = hasil
        st.success(f"Ditemukan {len(hasil)} toko yang memiliki stok!")
    else:
        st.session_state['hasil_cari'] = pd.DataFrame()
        st.error("Stok kosong di semua ritel.")

# --- SERVICE: ORDER ---
if 'hasil_cari' in st.session_state and not st.session_state['hasil_cari'].empty:
    st.write("### Hasil Pencarian")
    
    for index, row in st.session_state['hasil_cari'].iterrows():
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**{row['ritel_nama']}** ({row['ritel_id']})")
            col1.caption(f"Sisa Stok: {row['jumlah']} unit")
            
            # Tombol Beli
            if col2.button("Beli 1 Unit", key=f"btn_{row['ritel_id']}"):
                conn = get_db_connection()
                c = conn.cursor()
                
                # 1. Cek Stok Lagi (Concurrency Check)
                c.execute("SELECT jumlah FROM stok WHERE ritel_id=? AND tipe_ban=?", 
                          (row['ritel_id'], tipe_dicari))
                stok_sekarang = c.fetchone()[0]
                
                if stok_sekarang > 0:
                    # 2. Kurangi Stok
                    c.execute("UPDATE stok SET jumlah = jumlah - 1 WHERE ritel_id=? AND tipe_ban=?", 
                              (row['ritel_id'], tipe_dicari))
                    
                    # 3. Buat Invoice
                    order_id = str(uuid.uuid4())[:8]
                    c.execute("INSERT INTO pesanan VALUES (?, ?, ?, ?, ?)", 
                              (order_id, "User_Guest", row['ritel_id'], tipe_dicari, "LUNAS"))
                    
                    conn.commit()
                    st.toast(f"✅ Berhasil membeli dari {row['ritel_nama']}!")
                    # Refresh halaman
                    del st.session_state['hasil_cari']
                    st.rerun()
                else:
                    st.error("Yah, keduluan orang lain! Stok habis.")
                conn.close()
